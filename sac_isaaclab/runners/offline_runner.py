# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline Runner for training SAC with IsaacLab environments
Compatible with skrl's Runner interface
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Dict, Optional, Callable
import time
import os
from datetime import datetime

from skrl.agents.torch import Agent
from skrl.utils.runner.torch import Runner

from ..modules import SAC
from ..storage import ReplayBuffer, PrioritizedReplayBuffer


class OfflineRunner:
    """
    Offline RL Runner for SAC training with IsaacLab.
    
    Handles:
    - Environment interaction and data collection
    - Replay buffer management
    - Training loop with gradient updates
    - Checkpointing and logging
    - Video recording
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent: SAC,
        cfg: Optional[Dict] = None
    ):
        """
        Initialize Offline Runner.
        
        Args:
            env: IsaacLab environment (wrapped for skrl)
            agent: SAC agent
            cfg: Configuration dictionary
        """
        self.env = env
        self.agent = agent
        self.cfg = cfg or {}
        
        # Training configuration
        self.timesteps = self.cfg.get("timesteps", 1000000)
        self.warmup_timesteps = self.cfg.get("warmup_timesteps", 10000)
        self.eval_interval = self.cfg.get("eval_interval", 5000)
        self.eval_episodes = self.cfg.get("eval_episodes", 10)
        self.save_interval = self.cfg.get("save_interval", 50000)
        self.log_interval = self.cfg.get("log_interval", 1000)
        
        # Environment info
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = agent.device
        
        # Logging
        self.log_dir = self.cfg.get("log_dir", "logs/sac")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Tracking
        self.current_timestep = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Metrics
        self.metrics = {
            "train/episode_reward": [],
            "train/episode_length": [],
            "eval/mean_reward": [],
            "eval/mean_length": [],
        }
    
    def train(self) -> None:
        """
        Main training loop.
        """
        print(f"[INFO] Starting SAC training for {self.timesteps} timesteps")
        print(f"[INFO] Warmup timesteps: {self.warmup_timesteps}")
        print(f"[INFO] Number of parallel environments: {self.num_envs}")
        
        # Reset environment
        states, infos = self.env.reset()
        
        episode_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs)
        
        start_time = time.time()
        last_log_time = start_time
        
        while self.current_timestep < self.timesteps:
            # Select actions
            if self.current_timestep < self.warmup_timesteps:
                # Random actions during warmup
                actions = self._sample_random_actions()
            else:
                # Policy actions
                with torch.no_grad():
                    actions = self.agent.act(
                        states,
                        timestep=self.current_timestep,
                        eval_mode=False
                    )
            
            # Step environment
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            print("Size of variables during evaluation step:")
            print(next_states.size(), rewards.size(), terminated.size(), truncated.size())
            # Track episode statistics
            # Handle reward shape: [batch_size, 1] -> [batch_size]
            if torch.is_tensor(rewards):
                rewards_np = rewards.detach().squeeze(-1).cpu().numpy()
            else:
                rewards_np = np.asarray(rewards).squeeze()
            episode_rewards += rewards_np
            episode_lengths += 1
            
            # Record transition
            self.agent.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=self.current_timestep,
                timesteps=self.timesteps
            )
            
            # Handle episode completion
            done = np.logical_or(
                terminated.cpu().numpy() if torch.is_tensor(terminated) else terminated,
                truncated.cpu().numpy() if torch.is_tensor(truncated) else truncated
            )
            
            for i in range(self.num_envs):
                if done[i]:
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
            
            # Update state
            states = next_states
            self.current_timestep += self.num_envs
            
            # Logging
            if self.current_timestep % self.log_interval < self.num_envs:
                self._log_progress(start_time, last_log_time)
                last_log_time = time.time()
            
            # Evaluation
            if self.current_timestep % self.eval_interval < self.num_envs:
                if self.current_timestep >= self.warmup_timesteps:
                    self._evaluate()
            
            # Save checkpoint
            if self.current_timestep % self.save_interval < self.num_envs:
                self._save_checkpoint()
        
        # Final evaluation and save
        print("[INFO] Training complete. Running final evaluation...")
        self._evaluate()
        self._save_checkpoint(suffix="final")
        
        total_time = time.time() - start_time
        print(f"[INFO] Total training time: {total_time / 3600:.2f} hours")
    
    def _sample_random_actions(self) -> torch.Tensor:
        """Sample random actions from action space."""
        actions = []
        for _ in range(self.num_envs):
            action = self.env.action_space.sample()
            actions.append(torch.tensor(action, dtype=torch.float32, device=self.device))
        return torch.stack(actions)
    
    def _log_progress(self, start_time: float, last_log_time: float) -> None:
        """Log training progress."""
        elapsed = time.time() - start_time
        step_rate = self.log_interval / (time.time() - last_log_time + 1e-6)
        
        # Compute statistics
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]
        
        mean_reward = np.mean(recent_rewards)
        mean_length = np.mean(recent_lengths)
        
        # Get agent metrics
        agent_metrics = getattr(self.agent, "_track_metrics", {})
        
        print(
            f"Step {self.current_timestep}/{self.timesteps} | "
            f"Episodes: {len(self.episode_rewards)} | "
            f"Reward: {mean_reward:.2f} | "
            f"Length: {mean_length:.1f} | "
            f"Steps/s: {step_rate:.1f} | "
            f"Time: {elapsed / 3600:.2f}h"
        )
        
        # Log to file
        if agent_metrics:
            with open(os.path.join(self.log_dir, "training_log.txt"), "a") as f:
                f.write(f"Step {self.current_timestep}: {agent_metrics}\n")
    
    def _evaluate(self) -> None:
        """Run evaluation episodes."""
        print(f"\n[EVAL] Running evaluation at step {self.current_timestep}")
        
        eval_rewards = []
        eval_lengths = []
        
        # Reset for evaluation
        states, _ = self.env.reset()
        
        for ep in range(self.eval_episodes):
            ep_reward = 0
            ep_length = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    actions = self.agent.act(
                        states,
                        timestep=self.current_timestep,
                        eval_mode=True
                    )
                
                states, rewards, terminated, truncated, _ = self.env.step(actions)
                
                reward_val = rewards.cpu().item() if torch.is_tensor(rewards) else rewards
                done_val = terminated.cpu().item() if torch.is_tensor(terminated) else terminated
                
                ep_reward += reward_val
                ep_length += 1
                done = done_val
            
            eval_rewards.append(ep_reward)
            eval_lengths.append(ep_length)
        
        mean_reward = np.mean(eval_rewards)
        mean_length = np.mean(eval_lengths)
        
        self.metrics["eval/mean_reward"].append(mean_reward)
        self.metrics["eval/mean_length"].append(mean_length)
        
        print(f"[EVAL] Mean Reward: {mean_reward:.2f} (+/- {np.std(eval_rewards):.2f})")
        print(f"[EVAL] Mean Length: {mean_length:.1f}\n")
        
        # Save evaluation results
        with open(os.path.join(self.log_dir, "eval_log.txt"), "a") as f:
            f.write(f"Step {self.current_timestep}: Reward={mean_reward:.2f}, Length={mean_length:.1f}\n")
    
    def _save_checkpoint(self, suffix: str = "") -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.log_dir,
            f"checkpoint_{self.current_timestep}{f'_{suffix}' if suffix else ''}.pt"
        )
        
        self.agent.write_checkpoint(checkpoint_path, self.current_timestep)
        print(f"[INFO] Saved checkpoint to {checkpoint_path}")


class SkrlCompatibleRunner(Runner):
    """
    Wrapper to make our SAC compatible with skrl's Runner interface.
    This allows using the standard skrl training script.
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent_cfg: Dict,
        custom_sac_agent: Optional[SAC] = None
    ):
        """
        Initialize runner compatible with skrl.
        
        Args:
            env: Environment
            agent_cfg: Agent configuration (from YAML)
            custom_sac_agent: Optional pre-configured SAC agent
        """
        # Import model instantiators from skrl
        from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model
        from skrl.memories.torch import RandomMemory
        
        # Extract configuration
        models_cfg = agent_cfg.get("models", {})
        memory_cfg = agent_cfg.get("memory", {})
        agent_specific_cfg = agent_cfg.get("agent", {})
        
        # Create models
        if models_cfg.get("separate", False):
            models = {}
            
            # Policy model
            if "policy" in models_cfg:
                policy_cfg = models_cfg["policy"]
                if policy_cfg.get("class") == "GaussianMixin":
                    models["policy"] = gaussian_model(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        device=env.device,
                        **{k: v for k, v in policy_cfg.items() if k != "class" and k != "network"},
                        network=policy_cfg.get("network", [])
                    )
            
            # Critic models
            for critic_name in ["critic_1", "critic_2"]:
                if critic_name in models_cfg:
                    critic_cfg = models_cfg[critic_name]
                    models[critic_name] = deterministic_model(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        device=env.device,
                        **{k: v for k, v in critic_cfg.items() if k != "class" and k != "network"},
                        network=critic_cfg.get("network", [])
                    )
            
            # Target critics
            for target_name in ["target_critic_1", "target_critic_2"]:
                if target_name in models_cfg:
                    target_cfg = models_cfg[target_name]
                    models[target_name] = deterministic_model(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        device=env.device,
                        **{k: v for k, v in target_cfg.items() if k != "class" and k != "network"},
                        network=target_cfg.get("network", [])
                    )
        
        # Create memory
        memory = None
        if memory_cfg.get("class") == "RandomMemory":
            from skrl.memories.torch import RandomMemory
            memory = RandomMemory(
                memory_size=memory_cfg.get("memory_size", 100000),
                num_envs=env.num_envs,
                device=env.device
            )
        
        # Create SAC agent
        if custom_sac_agent is not None:
            self.agent = custom_sac_agent
        else:
            self.agent = SAC(
                models=models,
                memory=memory,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                cfg=agent_specific_cfg
            )
        
        self.env = env
        self.agent_cfg = agent_cfg
        
        # Initialize agent
        self.agent.init()
    
    def run(self) -> None:
        """Run training using the compatible runner."""
        trainer_cfg = self.agent_cfg.get("trainer", {})
        
        runner = OfflineRunner(
            env=self.env,
            agent=self.agent,
            cfg={
                "timesteps": trainer_cfg.get("timesteps", 1000000),
                "warmup_timesteps": self.agent_cfg.get("agent", {}).get("learning_starts", 10000),
                "eval_interval": trainer_cfg.get("eval_interval", 5000),
                "eval_episodes": trainer_cfg.get("eval_episodes", 10),
                "save_interval": trainer_cfg.get("save_interval", 50000),
                "log_interval": trainer_cfg.get("log_interval", 1000),
                "log_dir": os.path.join(
                    self.agent_cfg.get("agent", {}).get("experiment", {}).get("directory", "logs"),
                    self.agent_cfg.get("agent", {}).get("experiment", {}).get("experiment_name", "sac")
                )
            }
        )
        
        runner.train()
