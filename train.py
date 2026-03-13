#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone SAC Training Script for IsaacLab
No skrl dependencies except environment wrapper

Usage:
    python train.py --task Isaac-Ant-Direct-v0 --num_envs 4096 --headless
"""

import argparse
import sys
import os

debug = False
debug_tensors = False
def debug_allocated(i: int):
    if debug:
        import torch
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        print(f"Step {i} | Allocated: {allocated / 1e6:.2f} MB | Reserved: {reserved / 1e6:.2f} MB")
# Add sac_isaaclab to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train SAC agent with IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--use_per",
    action="store_true",
    help="Use Prioritized Experience Replay (PER) if set; otherwise use random replay buffer."
)
# Note: --headless and --device are handled by AppLauncher/IsaacLab

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports
import gymnasium as gym
import torch
torch.cuda.memory._record_memory_history(max_entries=100000)
import random
import numpy as np
from datetime import datetime

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from sac_isaaclab import GaussianActor, TwinQCritic, make_replay_buffer
from sac_isaaclab.modules import SAC, SAC_DEFAULT_CONFIG

def merge_dict(base: dict, override: dict):
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_dict(base[k], v)
        else:
            base[k] = v

def train_sac(env, agent, buffer, agent_cfg, log_dir):
    """
    Main SAC training loop.
    """

    import time
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    except ImportError:
        writer = None
        print("Warning: tensorboard not available")

    # Initialize
    obs, _ = env.reset()
    obs = torch.from_numpy(obs).float() if not isinstance(obs, torch.Tensor) else obs
    
    timestep = 0
    log_timestep = 0

    episode = 0
    episode_rewards = []
    episode_lengths = []
    
    num_envs = env.num_envs
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs)
    
    train_cfg = agent_cfg["training"]
    batch_size = train_cfg["batch_size"]
    warmup_steps = train_cfg["warmup_steps"]
    gradient_steps = train_cfg["updates_per_step"]
    max_iterations = train_cfg["max_iterations"]

    total_timesteps = max_iterations * env.num_envs * train_cfg["num_steps_per_env"]

    print(f"Starting SAC training for {total_timesteps} timesteps")
    print(f"Environment: {num_envs} parallel environments")
    
    start_time = time.time()


    

    while timestep < total_timesteps:
        # Select action
        debug_allocated(-3) 
        if timestep < warmup_steps:
            action_dim = env.action_space.shape[0]

            low = getattr(env.action_space, "low", None)
            high = getattr(env.action_space, "high", None)

            if low is None or high is None or np.any(np.isinf(low)) or np.any(np.isinf(high)):
                actions = 2 * torch.rand((num_envs, action_dim), device=agent.device) - 1
            else:
                low = torch.tensor(low, dtype=torch.float32, device=agent.device)
                high = torch.tensor(high, dtype=torch.float32, device=agent.device)
                actions = low + (high - low) * torch.rand((num_envs, action_dim), device=agent.device)
            obs_device = obs.to(agent.device) if not obs.is_cuda else obs
        else:
            with torch.no_grad():
                obs_device = obs.to(agent.device)
                actions = agent.select_action(obs_device, deterministic=False)
                # print(f"[Training] actions_list: {actions[:5].cpu().numpy()}")
                # actions = torch.zeros((num_envs, 8), dtype=torch.float32, device=agent.device)
                # Ensure actions is torch.Tensor on correct device
                if not isinstance(actions, torch.Tensor):
                    actions = torch.from_numpy(actions).float()
                actions = actions.to(agent.device)
                
        
        # Step environment with torch.Tensor (shape: [num_envs, action_dim])
        debug_allocated(-2)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        #print("Size of variables during evaluation step:")
        #print(next_obs.size(), rewards.size(), terminated.size(), truncated.size())
        # Prepare for buffer storage
        rewards_t = rewards.view(-1, 1) if rewards.dim() == 1 else rewards
        dones_t = (terminated.bool() | truncated.bool()).float().view(-1, 1)
        debug_allocated(-1)
        # Store transitions

        # print("Size of variables before buffer storage:")
        # print(obs.size(), actions.size(), rewards.size(), next_obs.size(), terminated.size(), truncated.size())
        # Convert environment outputs to tensors if needed
        debug_allocated(0)
        for i in range(num_envs):
            buffer.add(
                obs[i],
                actions[i],
                rewards_t[i],
                next_obs[i],
                dones_t[i]
            )
            current_rewards[i] += rewards_t[i].detach().cpu().numpy() if isinstance(rewards_t[i], torch.Tensor) else rewards_t[i]
            current_lengths[i] += 1
            
            if terminated[i] or truncated[i]:
                episode += 1
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                current_rewards[i] = 0
                current_lengths[i] = 0

        obs = next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs

        timestep += num_envs
        log_timestep += num_envs

        debug_allocated(1)
        # Train agent
        if timestep >= warmup_steps and len(buffer) >= batch_size:
            for _ in range(gradient_steps):
                # print(f"agent use per:{agent.use_per}")
                if agent.use_per:
                    batch, indices, weights = buffer.sample(batch_size)
                    # print(f"data type of batch: {type(batch)}, indices: {type(indices)}, weights: {type(weights)}")
                    metrics = agent.update(batch, weights)  #!!!!!!!Key for memory leak debugging: check if batch is properly released after update
                    if "td_error" in metrics:
                        td_errors = metrics["td_error"]
                        buffer.update_priorities(indices, td_errors)
                    del batch, indices, weights
                else:
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    del batch
                # torch.cuda.empty_cache()
                
                if timestep % 1000 == 0 and writer is not None:
                     for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"train/{key}", value, timestep)
        
        debug_allocated(2)
        # print(f"Buffer size: {len(buffer)}, timestep: {timestep}, episode: {episode}")
        # Logging
        if timestep % 1000 == 0 and timestep > 0:
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(
                max_entries=100000,
                context="all",   # 记录所有上下文
                stacks="all"     # 同时记录 C++ 和 Python 调用栈
            )
            # torch.cuda.empty_cache()
            if timestep >= 0:
                allocated = torch.cuda.memory_allocated(agent.device)
                reserved = torch.cuda.memory_reserved(agent.device)
                print(f"Step {timestep} | Allocated: {allocated / 1e6:.2f} MB | Reserved: {reserved / 1e6:.2f} MB")
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            if timestep % 10000 == 0 and debug_tensors:
                import gc
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                            if obj.is_cuda:
                                with open(f"tensors{timestep}", "a") as f:
                                    f.write(f"Tensor: {obj.shape}\n")
                    except Exception as e:
                        pass
            if len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            else:
                mean_reward = 0
                mean_length = 0
            

            print(f"Step {timestep}/{total_timesteps} | Episodes: {episode} | "
                  f"Reward: {mean_reward:.2f} | Length: {mean_length:.1f} | "
                  f"Buffer: {len(buffer)} | FPS: {fps:.0f}")
            
            if writer is not None:
                writer.add_scalar("train/mean_reward", mean_reward, timestep)
                writer.add_scalar("train/mean_length", mean_length, timestep)
                writer.add_scalar("train/buffer_size", len(buffer), timestep)
        
        debug_allocated(3)
        # Save checkpoint
        if log_timestep >= 500000000:
            log_timestep -= 500000000
            checkpoint_path = os.path.join(log_dir, f"checkpoint_{timestep}.pt")
            agent.save(checkpoint_path)
            buffer.save(os.path.join(log_dir, f"buffer_{timestep}.pt"))
            print(f"Saved checkpoint at timestep {timestep}")
        debug_allocated(4)
        debug_allocated(5)
        del actions, rewards_t, next_obs, dones_t, terminated, truncated, infos, rewards
        # torch.cuda.empty_cache()
        debug_allocated(6)
    # Final save
    agent.save(os.path.join(log_dir, "final_model.pt"))

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")
    
    if writer is not None:
        writer.close()


@hydra_task_config(args_cli.task, "sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg_override: dict):
    """Main training function."""
    # an default configuration for the agent, can be overridden by agent.yaml.
    agent_cfg = {
        "seed": 42,

        "network": {
            "actor": {
                "hidden_dims": [512, 256],
                "activation": "relu",
                "log_std_min": -20.0,
                "log_std_max": 2.0,
            },
            "critic": {
                "hidden_dims": [512, 256],
                "activation": "relu",
            },
        },

        "buffer": {
            "class": "ReplayBuffer",
            "buffer_size": 1_000_000,
            "prioritized": False,
            "alpha": 0.6,
            "beta": 0.4,
            "beta_increment": 0.001,
        },

        "agent": {
            "class": "SAC",
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,

            "gamma": 0.99,
            "tau": 0.005,

            "alpha": 0.2,
            "automatic_entropy_tuning": True,
            "target_entropy": None,
        },

        "training": {
            "batch_size": 2048,
            "warmup_steps": 10000,
            "updates_per_step": 1,
            "num_steps_per_env": 100,

            "max_iterations": 1000,

            "grad_norm_clip": 1.0,
        },

        "experiment": {
            "directory": "logs/sac",
            "experiment_name": "isaaclab",
            "write_interval": 100,
            "checkpoint_interval": 5000,
            "save_interval": 50,
            "log_interval": 10,
        },
    }
    if agent_cfg_override is not None:
        merge_dict(agent_cfg, agent_cfg_override)

    print(f"agent_cfg: {agent_cfg}")
    # Override configurations
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    
    # Set seed
    if args_cli.seed >= 0:
        random.seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        env_cfg.seed = args_cli.seed
    
    # Create log directory
    log_root_path = os.path.join("logs", "sac_isaaclab")
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed{args_cli.seed}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    
    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = args_cli.device
    print(f"[INFO] Environment created with {env.num_envs} parallel environments")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    
    # Create networks
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_bounds = (env.action_space.low[0], env.action_space.high[0])

    actor_cfg = agent_cfg["network"]["actor"]
    critic_cfg = agent_cfg["network"]["critic"]
    
    actor = GaussianActor(
        obs_dim=obs_shape[0],
        action_dim=action_shape[0],
        action_bounds=action_bounds,
        hidden_dims=actor_cfg["hidden_dims"],
        activation=actor_cfg["activation"],
        log_std_min=-20.0,
        log_std_max=2.0,
        use_tanh_squashing=True,
        device=device
    )
    
    critic = TwinQCritic(
        obs_dim=obs_shape[0],
        action_dim=action_shape[0],
        hidden_dims=critic_cfg["hidden_dims"],
        activation=critic_cfg["activation"],
        use_layer_norm=False,
        share_features=False,
        device=device
    )
    
    target_critic = TwinQCritic(
        obs_dim=obs_shape[0],
        action_dim=action_shape[0],
        hidden_dims=critic_cfg["hidden_dims"],
        activation=critic_cfg["activation"],
        use_layer_norm=False,
        share_features=False,
        device=device
    )
    
    # Create replay buffer using factory function
    use_per = agent_cfg.get("buffer", {}).get("prioritized", False) or getattr(args_cli, "use_per", False)
    agent_cfg["agent"]["use_per"] = use_per
    buffer_type = "priority" if use_per else "random"
    
    buffer_cfg = agent_cfg["buffer"]

    buffer = make_replay_buffer(
        buffer_type="priority" if buffer_cfg["prioritized"] else "random",
        buffer_size=buffer_cfg["buffer_size"],
        obs=obs_shape,
        action_shape=action_shape,
        device=device,
        alpha=buffer_cfg["alpha"],
        beta=buffer_cfg["beta"],
    )
    print(f"[INFO] Buffer size: {agent_cfg['buffer']['buffer_size']}")
    print(f"[INFO] Batch size: {agent_cfg['training']['batch_size']}")
    print(f"[INFO] Using {buffer_type} replay buffer")
    
    agent = SAC(
        actor,
        critic,
        target_critic,
        env.observation_space,
        env.action_space,
        device,
        agent_cfg["agent"]
    )


    agent_yaml_path = os.path.join(log_dir, "agent.yaml")
    dump_yaml(agent_yaml_path, agent_cfg)
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)
    
    # Train
    try:
        train_sac(env, agent, buffer, agent_cfg, log_dir)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted")
        agent.save(os.path.join(log_dir, "interrupted_model.pt"))
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
