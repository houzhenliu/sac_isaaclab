# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom SAC (Soft Actor-Critic) implementation for IsaacLab
Standalone version - no skrl dependencies except for environment wrapping
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import copy


class SAC:
    """
    Soft Actor-Critic (SAC) agent for continuous action spaces.
    Standalone implementation - no skrl dependencies.
    
    Based on "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    (Haarnoja et al., 2018)
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cuda:0",
        cfg: Optional[Dict[str, Any]] = None,
        use_per: bool = False
    ):
        """
        Initialize SAC agent.
        
        Args:
            actor: Policy network (GaussianActor)
            critic: Twin Q-networks (TwinQCritic)
            target_critic: Target twin Q-networks
            observation_space: Environment observation space
            action_space: Environment action space
            device: Computation device
            cfg: Configuration dictionary
        """
        self.cfg = cfg or SAC_DEFAULT_CONFIG
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Configuration parameters
        self.gamma = self.cfg.get("gamma", SAC_DEFAULT_CONFIG["gamma"])
        self.tau = self.cfg.get("tau", SAC_DEFAULT_CONFIG["tau"])
        self.lr_actor = self.cfg.get("actor_lr", SAC_DEFAULT_CONFIG["lr_actor"])
        self.lr_critic = self.cfg.get("critic_lr", SAC_DEFAULT_CONFIG["lr_critic"])
        self.lr_alpha = self.cfg.get("alpha_lr", SAC_DEFAULT_CONFIG["lr_alpha"])
        self.grad_norm_clip = self.cfg.get("grad_norm_clip", SAC_DEFAULT_CONFIG["grad_norm_clip"])
        self.batch_size = self.cfg.get("batch_size", SAC_DEFAULT_CONFIG["batch_size"])
        print(f"SAC Configuration: gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, lr_actor={self.lr_actor}, lr_critic={self.lr_critic}, lr_alpha={self.lr_alpha}")
 
        # Entropy parameters
        self.auto_entropy_tuning = self.cfg.get("auto_entropy_tuning", True)
        self.target_entropy = self.cfg.get("target_entropy", None)
        self.use_per = use_per
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).item()
        
        # Initialize alpha (entropy temperature)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        else:
            self.alpha = torch.tensor([self.cfg.get("alpha", 0.2)], device=device)
            self.log_alpha = None
            self.alpha_optimizer = None
        
        # Networks
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = target_critic.to(device)
        
        # Copy parameters to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self._update_cnt = 0
        # Freeze target network
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Tracking
        self._timestep = 0
        self._track_metrics = {}

        if self.use_per:
            print("[SAC]:Using Prioritized Experience Replay (PER)")
    
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action given observation.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return deterministic action
            
        Returns:
            Action tensor
        """
        with torch.no_grad():
            action, _ = self.actor.sample(obs, deterministic=deterministic)
            action = action.detach()
        return action
    
    def update(self, batch: Dict[str, torch.Tensor], weights: torch.Tensor = None) -> Dict[str, float]:
        """
        Update agent with a batch of transitions.
        
        Args:
            batch: Dictionary containing:
                - observations: [batch_size, obs_dim]
                - actions: [batch_size, action_dim]
                - rewards: [batch_size, 1]
                - next_observations: [batch_size, obs_dim]
                - dones: [batch_size, 1]
                
        Returns:
            Dictionary of metrics
        """

        # print(f"weights shape: {weights.shape if weights is not None else None}, batch size: {batch['observations'].shape[0]}")
        if weights is not None:
            weights = weights.unsqueeze(-1)
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        dones = batch["dones"]
        
        # Update critics
        critic_loss, q1_value, q2_value, target_value = self._update_critics(
            states, actions, rewards, next_states, dones, weights
        )
        
        # Update actor
        actor_loss, alpha_loss = self._update_actor(states)
        
        # Soft update target networks
        self._soft_update_target()
        
        with torch.no_grad():
            td_error = torch.max(
                (q1_value - target_value).abs(),
                (q2_value - target_value).abs()
            )
        metrics = {
            "loss/critic": critic_loss,
            "loss/actor": actor_loss,
            "loss/alpha": alpha_loss if alpha_loss is not None else 0.0,
            "alpha": self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            "q_value/mean": ((q1_value + q2_value) / 2).mean().item(),
            "q_value/std": ((q1_value + q2_value) / 2).std().item(),
            "td_error/mean": td_error.mean().item(),
            "td_error/std": td_error.std().item(),
            "td_error": td_error.cpu().detach().numpy(),
        }
        self._track_metrics = metrics
        return metrics
    
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Update Q-networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            target_q1, target_q2 = self.target_critic.get_both(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            target_q = target_q - self.alpha * next_log_probs
            
            # Compute target value
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic.get_both(states, actions)
        
        if self._update_cnt > 1000:
            print(f"Current Q1: {current_q1.mean().item():.4f} ± {current_q1.std().item():.4f}, Current Q2: {current_q2.mean().item():.4f} ± {current_q2.std().item():.4f}, Target Value: {target_value.mean().item():.4f} ± {target_value.std().item():.4f}")
            self._update_cnt = 0

        self._update_cnt += 1

        td_error1 = current_q1 - target_value
        td_error2 = current_q2 - target_value

        # Compute critic loss
        if weights is not None:
            critic_loss_1 = (td_error1.pow(2) * weights).mean()
            critic_loss_2 = (td_error2.pow(2) * weights).mean()
        else:
            critic_loss_1 = td_error1.pow(2).mean()
            critic_loss_2 = td_error2.pow(2).mean()

        critic_loss = critic_loss_1 + critic_loss_2
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()
        
        return critic_loss.item(), current_q1.detach(), current_q2.detach(), target_value.detach()
    
    def _update_actor(self, states: torch.Tensor) -> Tuple[float, Optional[float]]:
        """Update policy network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)
        
        # Compute Q-values for sampled actions
        q1, q2 = self.critic.get_both(states, actions)
        q = torch.min(q1, q2)
        
        # Compute actor loss
        alpha = self.alpha.detach()
        actor_loss = (alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        self.actor_optimizer.step()
        
        # Update alpha if using auto entropy tuning
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().detach()
        
        return actor_loss.item(), alpha_loss.item() if alpha_loss is not None else None
    
    def _soft_update_target(self) -> None:
        """Soft update target network parameters."""
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha": self.alpha if not self.auto_entropy_tuning else self.log_alpha,
            "timestep": self._timestep,
        }
        
        if self.auto_entropy_tuning:
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        if self.auto_entropy_tuning:
            self.log_alpha = checkpoint["alpha"]
            self.alpha = self.log_alpha.exp()
            if "alpha_optimizer_state_dict" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        else:
            self.alpha = checkpoint["alpha"]
        
        self._timestep = checkpoint.get("timestep", 0)
    
    def set_mode(self, mode: str) -> None:
        """Set network mode (train/eval)."""
        if mode == "train":
            self.actor.train()
            self.critic.train()
            self.target_critic.train()
        elif mode == "eval":
            self.actor.eval()
            self.critic.eval()
            self.target_critic.eval()


# Default SAC configuration
SAC_DEFAULT_CONFIG = {
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "grad_norm_clip": 1.0,
    "auto_entropy_tuning": True,
    "target_entropy": None,
    "alpha": 0.2,
}
