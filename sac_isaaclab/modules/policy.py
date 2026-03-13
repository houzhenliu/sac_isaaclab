# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Policy network for SAC - Stochastic Gaussian Policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

from skrl.models.torch import GaussianMixin, MultivariateGaussianMixin


class SACPolicy(GaussianMixin, nn.Module):
    """
    Stochastic policy for SAC using Gaussian distribution.
    
    Outputs mean and log_std for continuous actions, with reparameterization trick
    for gradient estimation through sampling.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network_features=[256, 256],
        activation="relu",
        **kwargs
    ):
        """
        Initialize SAC policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space  
            device: Computation device
            clip_actions: Whether to clip actions to action space bounds
            clip_log_std: Whether to clip log standard deviations
            min_log_std: Minimum log standard deviation
            max_log_std: Maximum log standard deviation
            initial_log_std: Initial log standard deviation
            network_features: Hidden layer sizes
            activation: Activation function name
        """
        nn.Module.__init__(self)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.clip_log_std = clip_log_std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # Build network
        self.network = self._build_network(
            self.input_dim, 
            network_features, 
            activation
        )
        
        # Output layers
        self.mean_layer = nn.Linear(network_features[-1], self.action_dim)
        self.log_std_layer = nn.Linear(network_features[-1], self.action_dim)
        
        # Initialize log_std
        nn.init.constant_(self.log_std_layer.weight, 0)
        nn.init.constant_(self.log_std_layer.bias, initial_log_std)
        
        # Action bounds
        self.register_buffer("action_scale", torch.tensor(
            (action_space.high - action_space.low) / 2.0, 
            dtype=torch.float32, device=device
        ))
        self.register_buffer("action_bias", torch.tensor(
            (action_space.high + action_space.low) / 2.0,
            dtype=torch.float32, device=device
        ))
        
        self.to(device)
    
    def _build_network(self, input_dim, features, activation):
        """Build the feature extraction network."""
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for feature in features:
            layers.append(nn.Linear(prev_dim, feature))
            layers.append(activation_fn)
            prev_dim = feature
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def compute(self, inputs: Dict, role: str = "") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute policy outputs.
        
        Args:
            inputs: Dictionary containing 'states'
            role: Role identifier
            
        Returns:
            Tuple of (actions, log_prob, outputs_dict)
        """
        states = inputs["states"]
        
        # Forward pass through network
        features = self.network(states)
        
        # Compute mean and log_std
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clip log_std if configured
        if self.clip_log_std:
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        # For SAC, we output mean and log_std
        # The GaussianMixin will handle sampling and log_prob computation
        return mean, log_std, {}
    
    def get_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.
        
        Args:
            states: Input states
            
        Returns:
            Entropy values
        """
        features = self.network(states)
        log_std = self.log_std_layer(features)
        
        if self.clip_log_std:
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        # Entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2)
        entropy = 0.5 * (log_std.shape[-1] * (1.0 + math.log(2 * math.pi)) + log_std.sum(dim=-1))
        return entropy


class TanhSACPolicy(GaussianMixin, nn.Module):
    """
    SAC Policy with tanh squashing for bounded actions.
    
    This is the standard SAC policy that applies tanh to the sampled actions
    to keep them within the action space bounds.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network_features=[256, 256],
        activation="relu",
        **kwargs
    ):
        nn.Module.__init__(self)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.clip_log_std = clip_log_std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        self.input_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # Build network
        self.network = self._build_network(self.input_dim, network_features, activation)
        
        # Output layers
        self.mean_layer = nn.Linear(network_features[-1], self.action_dim)
        self.log_std_layer = nn.Linear(network_features[-1], self.action_dim)
        
        # Initialize
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.constant_(self.log_std_layer.weight, 0)
        nn.init.constant_(self.log_std_layer.bias, initial_log_std)
        
        # Action bounds for tanh squashing
        self.register_buffer("action_scale", torch.tensor(
            (action_space.high - action_space.low) / 2.0,
            dtype=torch.float32, device=device
        ))
        self.register_buffer("action_bias", torch.tensor(
            (action_space.high + action_space.low) / 2.0,
            dtype=torch.float32, device=device
        ))
        
        self.to(device)
    
    def _build_network(self, input_dim, features, activation):
        """Build feature extraction network."""
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for feature in features:
            layers.append(nn.Linear(prev_dim, feature))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = feature
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def compute(self, inputs: Dict, role: str = "") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute policy outputs with tanh squashing."""
        states = inputs["states"]
        
        features = self.network(states)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        if self.clip_log_std:
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mean, log_std, {}


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy for applications requiring deterministic action selection.
    Can be used for evaluation or in DDPG-style algorithms.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        network_features=[256, 256],
        activation="relu",
        **kwargs
    ):
        super().__init__()
        
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.input_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # Build network
        self.network = self._build_network(self.input_dim, network_features, activation)
        self.output_layer = nn.Linear(network_features[-1], self.action_dim)
        
        # Action bounds
        self.register_buffer("action_scale", torch.tensor(
            (action_space.high - action_space.low) / 2.0,
            dtype=torch.float32, device=device
        ))
        self.register_buffer("action_bias", torch.tensor(
            (action_space.high + action_space.low) / 2.0,
            dtype=torch.float32, device=device
        ))
        
        self.to(device)
    
    def _build_network(self, input_dim, features, activation):
        """Build feature extraction network."""
        layers = []
        prev_dim = input_dim
        
        for feature in features:
            layers.append(nn.Linear(prev_dim, feature))
            layers.append(self._get_activation(activation))
            prev_dim = feature
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get deterministic actions.
        
        Args:
            states: Input states
            
        Returns:
            Actions (tanh-squashed)
        """
        features = self.network(states)
        actions = torch.tanh(self.output_layer(features))
        
        # Scale to action space
        actions = actions * self.action_scale + self.action_bias
        
        return actions
