# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Replay Buffer for off-policy RL
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import copy

debug = False
def debug_allocated(i: int):
    if debug:
        import torch
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        print(f"Step {i} | Allocated: {allocated / 1e6:.2f} MB | Reserved: {reserved / 1e6:.2f} MB")
class ReplayBuffer:
    """
    Standard replay buffer for off-policy RL algorithms.
    Stores and samples transitions (s, a, r, s', done).
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            obs: Shape of observations
            action_shape: Shape of actions
            device: Device to store tensors on
            dtype: Data type for tensors
        """
        self.buffer_size = buffer_size
        self.obs = obs
        self.action_shape = action_shape
        self.device = device
        self.dtype = dtype
        
        # Allocate memory
        self.observations = torch.zeros((buffer_size, *obs), dtype=dtype, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=dtype, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=dtype, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs), dtype=dtype, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=dtype, device=device)
        
        self.position = 0
        self.size = 0
    
    def add(
        self,
        observation: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        reward: Union[torch.Tensor, np.ndarray],
        next_observation: Union[torch.Tensor, np.ndarray],
        done: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """

        observation = observation.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_observation = next_observation.to(self.device)
        done = done.to(self.device)
        # debug_allocated(12)
        # Store
        idx = self.position
        self.observations[idx] = observation.detach()
        self.actions[idx] = action.detach()
        self.rewards[idx] = reward.detach()
        self.next_observations[idx] = next_observation.detach()
        self.dones[idx] = done.detach()
        
        # Update position
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary containing sampled transitions
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
    
    def sample_indices(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Sample indices and return corresponding data.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (data dict, indices)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "truncated": self.truncated[indices],
        }
        
        return data, indices
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all transitions in the buffer."""
        return {
            "observations": self.observations[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "next_observations": self.next_observations[:self.size],
            "dones": self.dones[:self.size],
            "truncated": self.truncated[:self.size],
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.buffer_size
    
    def save(self, filepath: str) -> None:
        """Save buffer to file."""
        torch.save({
            "observations": self.observations[:self.size].cpu(),
            "actions": self.actions[:self.size].cpu(),
            "rewards": self.rewards[:self.size].cpu(),
            "next_observations": self.next_observations[:self.size].cpu(),
            "dones": self.dones[:self.size].cpu(),
            "truncated": self.truncated[:self.size].cpu(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load buffer from file."""
        data = torch.load(filepath, map_location=self.device)
        
        size = data["observations"].shape[0]
        self.size = size
        self.position = size % self.buffer_size
        
        self.observations[:size] = data["observations"].to(self.device)
        self.actions[:size] = data["actions"].to(self.device)
        self.rewards[:size] = data["rewards"].to(self.device)
        self.next_observations[:size] = data["next_observations"].to(self.device)
        self.dones[:size] = data["dones"].to(self.device)
        self.truncated[:size] = data["truncated"].to(self.device)


