# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Q-Network (Critic) for SAC
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from skrl.models.torch import DeterministicMixin


class QNetwork(DeterministicMixin, nn.Module):
    """
    Q-Network (Critic) for SAC.
    
    Estimates Q(s, a) - the expected return of taking action a in state s.
    Uses twin Q-networks to reduce overestimation bias.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        clip_actions=False,
        network_features=[256, 256],
        activation="relu",
        use_layer_norm=False,
        **kwargs
    ):
        """
        Initialize Q-Network.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            device: Computation device
            clip_actions: Whether to clip actions (unused for critic)
            network_features: Hidden layer sizes
            activation: Activation function name
            use_layer_norm: Whether to use layer normalization
        """
        nn.Module.__init__(self)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.input_dim = observation_space.shape[0] + action_space.shape[0]
        self.use_layer_norm = use_layer_norm
        
        # Build network
        self.network = self._build_network(
            self.input_dim,
            network_features,
            activation,
            use_layer_norm
        )
        
        # Output layer
        self.output_layer = nn.Linear(network_features[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)
    
    def _build_network(self, input_dim, features, activation, use_layer_norm):
        """Build the Q-network."""
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for i, feature in enumerate(features):
            layers.append(nn.Linear(prev_dim, feature))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(feature))
            
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
    
    def _init_weights(self):
        """Initialize network weights."""
        # Use small initialization for final layer for stable initial Q-values
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
    
    def compute(self, inputs: Dict, role: str = "") -> Tuple[torch.Tensor, None, Dict]:
        """
        Compute Q-value.
        
        Args:
            inputs: Dictionary containing 'states' and 'taken_actions'
            role: Role identifier
            
        Returns:
            Tuple of (Q-value, None, outputs_dict)
        """
        states = inputs["states"]
        actions = inputs["taken_actions"]
        
        # Concatenate state and action
        x = torch.cat([states, actions], dim=-1)
        
        # Forward pass
        features = self.network(x)
        q_value = self.output_layer(features)
        
        return q_value, None, {}


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for better uncertainty estimation.
    Can be used for enhanced exploration or uncertainty-based methods.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        num_ensemble=2,
        network_features=[256, 256],
        activation="relu",
        **kwargs
    ):
        super().__init__()
        
        self.device = device
        self.num_ensemble = num_ensemble
        
        # Create ensemble of Q-networks
        self.q_networks = nn.ModuleList([
            QNetwork(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                network_features=network_features,
                activation=activation,
                **kwargs
            ) for _ in range(num_ensemble)
        ])
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.
        
        Args:
            states: Input states
            actions: Input actions
            
        Returns:
            Tuple of (q_values [num_ensemble, batch_size, 1], mean_q_value)
        """
        q_values = []
        for q_net in self.q_networks:
            q, _, _ = q_net.compute({"states": states, "taken_actions": actions})
            q_values.append(q)
        
        q_values = torch.stack(q_values, dim=0)
        mean_q = q_values.mean(dim=0)
        
        return q_values, mean_q
    
    def get_min_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value across ensemble (for conservative estimation)."""
        q_values, _ = self.forward(states, actions)
        return q_values.min(dim=0)[0]
    
    def get_std(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get standard deviation of Q-values (uncertainty estimate)."""
        q_values, _ = self.forward(states, actions)
        return q_values.std(dim=0)


class DoubleQNetwork(nn.Module):
    """
    Double Q-Network wrapper for SAC twin critics.
    Provides convenient interface for twin Q-networks.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        network_features=[256, 256],
        activation="relu",
        use_layer_norm=False,
        **kwargs
    ):
        super().__init__()
        
        self.q1 = QNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            network_features=network_features,
            activation=activation,
            use_layer_norm=use_layer_norm,
            **kwargs
        )
        
        self.q2 = QNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            network_features=network_features,
            activation=activation,
            use_layer_norm=use_layer_norm,
            **kwargs
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both networks."""
        q1, _, _ = self.q1.compute({"states": states, "taken_actions": actions})
        q2, _, _ = self.q2.compute({"states": states, "taken_actions": actions})
        return q1, q2
    
    def get_min_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value (used for target computation in SAC)."""
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)
