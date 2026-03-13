# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Actor network for SAC - Stochastic policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

from .mlp import MLP
from .base import FeatureExtractor


class GaussianActor(nn.Module):
    """
    Stochastic actor using Gaussian distribution with reparameterization trick.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std: float = 0.0,
        # Additional parameters for train.py compatibility
        action_bounds: tuple = (-1.0, 1.0),
        use_tanh_squashing: bool = True,
        device: str = "cuda:0",
    ):
        """
        Initialize Gaussian Actor.
        
        Args:
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            init_log_std: Initial log standard deviation
            action_bounds: (low, high) bounds for actions
            use_tanh_squashing: Whether to use tanh squashing for bounded actions
            device: Device for tensors
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh_squashing = use_tanh_squashing
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            obs_dim,
            hidden_dims,
            activation,
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # Mean and log_std layers
        self.mean_layer = nn.Linear(feature_dim, action_dim)
        self.log_std_layer = nn.Linear(feature_dim, action_dim)
        
        # Initialize
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.constant_(self.log_std_layer.bias, init_log_std)
        
        # Action bounds for scaling
        low, high = action_bounds
        self.register_buffer("action_scale", torch.tensor(
            (high - low) / 2.0, dtype=torch.float32, device=device
        ))
        self.register_buffer("action_bias", torch.tensor(
            (high + low) / 2.0, dtype=torch.float32, device=device
        ))
        
        self.to(device)
    
    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get mean and log_std.
        
        Args:
            observation: Observations
            deterministic: If True, return deterministic actions
            
        Returns:
            Tuple of (mean, log_std)
        """
        features = self.feature_extractor(observation)
        # with open("features_debug.txt", "a") as f:
        #     f.write(f"[Features]features shape: {features.shape}\n")
        #     f.write(f"[Features]features finite? {torch.isfinite(features).all()}\n")
        #     f.write(f"[Features]features sample:\n{features[:1]}\n")
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            observation: Observations
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        # print(f"[SAC] Sampling action for observation shape: {observation.shape}")
        # print(f"[SAC] Observation sample:\n{observation[:1]}")
        mean, log_std = self.forward(observation)
        log_std = torch.clamp(log_std, min=-20, max=2)
        if deterministic:
            if self.use_tanh_squashing:
                action = torch.tanh(mean)
            else:
                action = mean
            # Scale to action bounds
            action = action * self.action_scale + self.action_bias
            log_prob = None
        else:
            # Reparameterization trick
            std = log_std.exp()
            noise = torch.randn_like(mean)
            # print(f"[SAC] mean first 5 samples:\n{mean[:5]}")
            # print(f"[SAC] log_std first 5 samples:\n{log_std[:5]}")
            # print(f"[SAC] std first 5 samples:\n{std[:5]}")
            # print(f"[SAC] noise first 5 samples:\n{noise[:5]}")
            z = mean + std * noise
                # print(f"[SAC] z first 5 samples:\n{z[:5]}")
                # print(f"[SAC] z finite? {torch.isfinite(z).all()}")
            if self.use_tanh_squashing:
                action = torch.tanh(z)
                # print(f"[SAC] action(tanh) first 5 samples:\n{z[:5]}")
                # Compute log probability with tanh correction
                log_prob = self._compute_log_prob(mean, log_std, z)
            else:
                action = z
                # Standard Gaussian log prob
                log_prob = -0.5 * (((z - mean) / std).pow(2) + 2 * log_std + np.log(2 * np.pi))
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            # Scale to action bounds
            action = action * self.action_scale + self.action_bias
        
        return action, log_prob
    
    def _compute_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of action with tanh correction.
        
        Args:
            mean: Mean of Gaussian
            log_std: Log std of Gaussian
            pre_tanh: Pre-tanh value
            
        Returns:
            Log probability
        """
        std = log_std.exp()
        
        # Log prob of Gaussian
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std).pow(2) +
            2 * log_std +
            np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        
        # Tanh correction
        log_prob -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)
        
        return log_prob
        # Log prob of Gaussian
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std).pow(2) +
            2 * log_std +
            np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        
        # Tanh correction
        log_prob -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)
        
        return log_prob
    
    def get_entropy(self, observation: torch.Tensor) -> torch.Tensor:
        """Get entropy of the policy."""
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        
        # Entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2)
        entropy = 0.5 * (log_std.shape[-1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1))
        
        return entropy


class SimpleActor(nn.Module):
    """
    Simple deterministic or stochastic actor.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        stochastic: bool = True,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.stochastic = stochastic
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        
        if stochastic:
            self.log_std_layer = nn.Linear(prev_dim, action_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Forward pass."""
        features = self.network(observation)
        mean = self.mean_layer(features)
        
        if self.stochastic:
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            return torch.tanh(mean)
    
    def sample(self, observation: torch.Tensor, deterministic: bool = False):
        """Sample action."""
        output = self.forward(observation)
        
        if self.stochastic:
            mean, log_std = output
            
            if deterministic:
                return torch.tanh(mean), None
            
            std = log_std.exp()
            noise = torch.randn_like(mean)
            action = torch.tanh(mean + std * noise)
            
            return action, None  # Simplified, can compute log_prob if needed
        else:
            return output, None


class SharedActorCritic(nn.Module):
    """
    Actor and Critic with shared feature extractor.
    Can be used for parameter sharing.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared_features = FeatureExtractor(
            obs_dim,
            hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims,
            activation,
        )
        
        feature_dim = self.shared_features.output_dim
        
        # Actor head
        self.actor_mean = nn.Linear(feature_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (state value)
        self.critic = nn.Linear(feature_dim, 1)
    
    def forward_actor(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through actor."""
        features = self.shared_features(observation)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        return mean, log_std
    
    def forward_critic(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward through critic."""
        features = self.shared_features(observation)
        value = self.critic(features)
        return value
    
    def forward(self, observation: torch.Tensor):
        """Forward through both."""
        features = self.shared_features(observation)
        
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        value = self.critic(features)
        
        return mean, log_std, value
