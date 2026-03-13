# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Storage utilities
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms.
    Can be used for GAE computation.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.dtype = dtype
        
        # Storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.position = 0
    
    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a rollout step."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        self.position += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value of last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        return returns, advantages
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all rollout data."""
        return {
            "observations": torch.stack(self.observations),
            "actions": torch.stack(self.actions),
            "rewards": torch.stack(self.rewards),
            "values": torch.stack(self.values),
            "log_probs": torch.stack(self.log_probs),
            "dones": torch.stack(self.dones),
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.position = 0
    
    def __len__(self) -> int:
        return self.position


class FrameStack:
    """
    Frame stacking for observation preprocessing.
    """
    
    def __init__(self, num_frames: int, frame_shape: Tuple[int, ...]):
        """
        Initialize frame stack.
        
        Args:
            num_frames: Number of frames to stack
            frame_shape: Shape of a single frame
        """
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset frame stack."""
        self.frames.clear()
        
        if frame is not None:
            for _ in range(self.num_frames):
                self.frames.append(frame)
        else:
            for _ in range(self.num_frames):
                self.frames.append(np.zeros(self.frame_shape))
        
        return self.get_stack()
    
    def add(self, frame: np.ndarray) -> np.ndarray:
        """Add frame and return stacked frames."""
        self.frames.append(frame)
        return self.get_stack()
    
    def get_stack(self) -> np.ndarray:
        """Get current stack."""
        return np.concatenate(list(self.frames), axis=-1)


class RewardNormalizer:
    """
    Running reward normalizer.
    """
    
    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize reward normalizer.
        
        Args:
            alpha: Update rate
            epsilon: Small constant
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = 0.0
        self.var = 1.0
    
    def update(self, reward: float) -> None:
        """Update running statistics."""
        self.mean = (1 - self.alpha) * self.mean + self.alpha * reward
        self.var = (1 - self.alpha) * self.var + self.alpha * (reward - self.mean) ** 2
    
    def normalize(self, reward: float) -> float:
        """Normalize reward."""
        return (reward - self.mean) / (np.sqrt(self.var) + self.epsilon)


class StateNormalizer:
    """
    Running state normalizer.
    """
    
    def __init__(self, state_dim: int, epsilon: float = 1e-8):
        """
        Initialize state normalizer.
        
        Args:
            state_dim: State dimension
            epsilon: Small constant
        """
        self.epsilon = epsilon
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = epsilon
    
    def update(self, state: np.ndarray) -> None:
        """Update running statistics."""
        self.count += 1
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.var += delta * delta2
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state."""
        return (state - self.mean) / (np.sqrt(self.var / self.count) + self.epsilon)


def merge_batches(batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Merge multiple batches into one.
    
    Args:
        batches: List of batch dictionaries
        
    Returns:
        Merged batch
    """
    merged = {}
    keys = batches[0].keys()
    
    for key in keys:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    
    return merged


def split_batch(batch: Dict[str, torch.Tensor], num_splits: int) -> List[Dict[str, torch.Tensor]]:
    """
    Split a batch into multiple smaller batches.
    
    Args:
        batch: Batch dictionary
        num_splits: Number of splits
        
    Returns:
        List of split batches
    """
    batch_size = next(iter(batch.values())).shape[0]
    split_size = batch_size // num_splits
    
    splits = []
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else batch_size
        
        split = {key: value[start:end] for key, value in batch.items()}
        splits.append(split)
    
    return splits


def compute_discount_rewards(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        Discounted rewards
    """
    discounted = []
    running = 0
    
    for r in reversed(rewards):
        running = r + gamma * running
        discounted.insert(0, running)
    
    return discounted
