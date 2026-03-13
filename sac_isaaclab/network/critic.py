# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Critic network for SAC - Q-function
"""

import torch
import torch.nn as nn
from typing import Tuple

from .mlp import MLP
from .base import FeatureExtractor


class QCritic(nn.Module):
    """
    Q-function critic that estimates Q(s, a).
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        """
        Initialize Q-Critic.
        
        Args:
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        # Concatenate observation and action as input
        input_dim = obs_dim + action_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(activation_fn)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value.
        
        Args:
            observation: Observations
            action: Actions
            
        Returns:
            Q-values
        """
        x = torch.cat([observation, action], dim=-1)
        return self.network(x)


class StateCritic(nn.Module):
    """
    State-value critic V(s) for use with deterministic policy.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute state value."""
        return self.network(observation)


class TwinQCritic(nn.Module):
    """
    Twin Q-critics for SAC.
    Returns minimum of two Q-values to reduce overestimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = False,
        share_features: bool = False,  # Added for compatibility
        device: str = "cuda:0",
        **kwargs
    ):
        super().__init__()
        
        self.q1 = QCritic(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            use_layer_norm,
        )
        
        self.q2 = QCritic(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            use_layer_norm,
        )
        
        self.to(device)
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_both: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both critics.
        
        Args:
            observation: Observations
            action: Actions
            return_both: If True, return both Q-values
            
        Returns:
            Q-values (min or both)
        """
        q1 = self.q1(observation, action)
        q2 = self.q2(observation, action)
        
        if return_both:
            return q1, q2
        else:
            return torch.min(q1, q2)
    
    def get_both(self, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both Q-values."""
        return self.q1(observation, action), self.q2(observation, action)
    
    def q1_forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward through Q1 only."""
        return self.q1(observation, action)
    
    def q2_forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward through Q2 only."""
        return self.q2(observation, action)


class EnsembleCritic(nn.Module):
    """
    Ensemble of Q-critics for uncertainty estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_critics: int = 5,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.num_critics = num_critics
        
        self.critics = nn.ModuleList([
            QCritic(obs_dim, action_dim, hidden_dims, activation)
            for _ in range(num_critics)
        ])
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward through ensemble.
        
        Args:
            observation: Observations
            action: Actions
            return_all: If True, return all Q-values
            
        Returns:
            Q-values (mean or all)
        """
        q_values = torch.stack([
            critic(observation, action) for critic in self.critics
        ], dim=0)
        
        if return_all:
            return q_values
        else:
            return q_values.mean(dim=0)
    
    def get_min(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value across ensemble."""
        q_values = self.forward(observation, action, return_all=True)
        return q_values.min(dim=0)[0]
    
    def get_std(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get standard deviation of Q-values."""
        q_values = self.forward(observation, action, return_all=True)
        return q_values.std(dim=0)


class FeatureCritic(nn.Module):
    """
    Critic with feature extraction shared with actor.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_extractor: nn.Module,
        hidden_dim: int = 256,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        
        # Q-value head
        feature_dim = feature_extractor.output_dim + action_dim
        
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.q_head = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value."""
        features = self.feature_extractor(observation)
        x = torch.cat([features, action], dim=-1)
        return self.q_head(x)
