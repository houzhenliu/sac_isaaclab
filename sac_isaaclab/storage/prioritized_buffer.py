# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Prioritized Experience Replay Buffer
Based on "Prioritized Experience Replay" (Schaul et al., 2016)
"""

import torch
import numpy as np
import gc
import os
import time
from typing import Dict, Tuple, Optional, Union
import sys

class SumTree:
    """
    Sum Tree data structure for efficient priority sampling.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree.
        
        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index from priority."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority."""
        return self.tree[0]
    
    def add(self, priority: float, data) -> None:
        """Add data with priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get data for priority s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Samples transitions with probability proportional to their TD error.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        gpu_mem_log_path: Optional[str] = None,
        max_priority: float = 1.0,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size: Maximum number of transitions
            obs: Shape of observations
            action_shape: Shape of actions
            device: Device for tensors
            dtype: Data type
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta annealing rate
            epsilon: Small constant to ensure non-zero priorities
        """
        self.buffer_size = buffer_size
        self.obs = obs
        self.action_shape = action_shape
        self.device = device
        self.dtype = dtype
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_beta = 1.0
        self._gpu_mem_log_path = gpu_mem_log_path
        self._last_mem_allocated = None
        self._last_mem_reserved = None
        self._gpu_log_enabled = (
            gpu_mem_log_path is not None
            and torch.cuda.is_available()
            and torch.device(device).type == "cuda"
        )

        if self._gpu_log_enabled:
            self._init_gpu_log_file()
        
        # Sum tree for priorities
        self.tree = SumTree(buffer_size)
        
        # Storage
        self.observations = torch.zeros((buffer_size, *obs), dtype=dtype,device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=dtype,device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=dtype,device=device)
        self.next_observations = torch.zeros((buffer_size, *obs), dtype=dtype,device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=dtype,device=device)
        
        self.position = 0
        self._log_gpu_memory("init", f"buffer_size={buffer_size}")

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        return f"{num_bytes / (1024 ** 2):.2f}MB"

    def _get_cuda_index(self) -> Optional[int]:
        if not self._gpu_log_enabled:
            return None
        cuda_device = torch.device(self.device)
        if cuda_device.index is not None:
            return cuda_device.index
        return torch.cuda.current_device()

    def _init_gpu_log_file(self) -> None:
        if self._gpu_mem_log_path is None:
            return
        log_dir = os.path.dirname(self._gpu_mem_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self._gpu_mem_log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== PrioritizedReplayBuffer GPU Memory Trace Start ===\n")

    def _log_gpu_memory(self, event: str, note: str = "") -> None:
        if not self._gpu_log_enabled or self._gpu_mem_log_path is None:
            return

        cuda_idx = self._get_cuda_index()
        if cuda_idx is None:
            return

        allocated = torch.cuda.memory_allocated(cuda_idx)
        reserved = torch.cuda.memory_reserved(cuda_idx)
        max_allocated = torch.cuda.max_memory_allocated(cuda_idx)
        max_reserved = torch.cuda.max_memory_reserved(cuda_idx)

        delta_allocated = 0 if self._last_mem_allocated is None else allocated - self._last_mem_allocated
        delta_reserved = 0 if self._last_mem_reserved is None else reserved - self._last_mem_reserved

        release_detected = ""
        if delta_allocated < 0 or delta_reserved < 0:
            release_detected = " | release_detected=yes"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = (
            f"{timestamp} | event={event} | device=cuda:{cuda_idx}"
            f" | allocated={self._format_bytes(allocated)} (delta={self._format_bytes(delta_allocated)})"
            f" | reserved={self._format_bytes(reserved)} (delta={self._format_bytes(delta_reserved)})"
            f" | max_allocated={self._format_bytes(max_allocated)}"
            f" | max_reserved={self._format_bytes(max_reserved)}"
            f"{release_detected}"
        )
        if note:
            message += f" | note={note}"

        with open(self._gpu_mem_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

        self._last_mem_allocated = allocated
        self._last_mem_reserved = reserved

    def log_gpu_memory_snapshot(self, tag: str = "manual") -> None:
        self._log_gpu_memory(f"snapshot:{tag}")

    def flush_gpu_cache(self, tag: str = "manual") -> None:
        self._log_gpu_memory(f"empty_cache:before:{tag}")
        if self._gpu_log_enabled:
            gc.collect()
            torch.cuda.empty_cache()
        self._log_gpu_memory(f"empty_cache:after:{tag}")
    
    def _get_priority(self, error: float) -> float:
        """Compute priority from TD error."""
        error = np.nan_to_num(error, nan=0.0, posinf=1e6, neginf=1e-6)
        p = (np.abs(error) + self.epsilon) ** self.alpha
        p = np.clip(p, 1e-6, 1e3)
        return p
    
    def add(
        self,
        observation: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        reward: Union[torch.Tensor, np.ndarray],
        next_observation: Union[torch.Tensor, np.ndarray],
        done: Union[torch.Tensor, np.ndarray],
        error: Optional[float] = None,
    ) -> None:
        """
        Add transition to buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Done flag
            error: TD error (if None, use max priority)
        """
        # Convert to tensors
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(self.dtype)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.dtype)
        if isinstance(reward, (np.ndarray, float, int)):
            reward = torch.tensor([reward], dtype=self.dtype)
        if isinstance(next_observation, np.ndarray):
            next_observation = torch.from_numpy(next_observation).to(self.dtype)
        if isinstance(done, (np.ndarray, bool)):
            done = torch.tensor([float(done)], dtype=self.dtype)
        
        # Store data
        idx = self.tree.write_idx
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_observation
        self.dones[idx] = done
        
        # Add to tree with max priority (new transitions get max priority)
        if error is None:
            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority == 0:
                max_priority = 1.0
            priority = max_priority
        else:
            priority = self._get_priority(error)
        
        self.tree.add(priority, idx)
        
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (data dict, indices, importance weights)
        """

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Sample from tree
        total = self.tree.total()
        if total == 0:
            total = 1e-6

        segment = total / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            # print(f"Sampling segment {i}: low={low:.4f}, high={high:.4f}")
            try:
                s = np.random.uniform(low, high)
            except OverflowError:
                print(f"[ERROR] np.random.uniform failed!")
                print(f"low = {low}, high = {high}")
                sys.exit(1)
            idx, priority, data_idx = self.tree.get(s)
            
            indices[i] = data_idx
            priorities[i] = priority
        
        # Compute importance sampling weights
        # P(i) = p_i / sum(p)
        # w_i = (N * P(i))^(-beta)
        sampling_probs = priorities / total
        sampling_probs = np.clip(sampling_probs, 1e-6, 1.0)

        weights = (self.tree.size * sampling_probs) ** (-self.beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights /= (weights.max()+1e-6)  
        # Anneal beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        # Get data
        data = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
        # print(f"data type: f{type(data)}")
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device).unsqueeze(1)
        self._log_gpu_memory("sample:end", f"batch_size={batch_size}")
        
        return data, indices, weights
    
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            errors: TD errors
        """
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

        self._log_gpu_memory("update_priorities", f"num_indices={len(indices)}")
        
    def save(self, filepath: str) -> None:
        """Save buffer to file."""
        self._log_gpu_memory("save:start", f"path={filepath}")
        torch.save({
            "observations": self.observations[:self.tree.size].cpu(),
            "actions": self.actions[:self.tree.size].cpu(),
            "rewards": self.rewards[:self.tree.size].cpu(),
            "next_observations": self.next_observations[:self.tree.size].cpu(),
            "dones": self.dones[:self.tree.size].cpu(),
        }, filepath)
        self._log_gpu_memory("save:end", f"path={filepath}")
    
    def load(self, filepath: str) -> None:
        """Load buffer from file."""
        self._log_gpu_memory("load:start", f"path={filepath}")
        data = torch.load(filepath, map_location=self.device)
        
        size = data["observations"].shape[0]
        self.tree.size = size
        self.position = size % self.buffer_size
        
        self.observations[:size] = data["observations"].to(self.device)
        self.actions[:size] = data["actions"].to(self.device)
        self.rewards[:size] = data["rewards"].to(self.device)
        self.next_observations[:size] = data["next_observations"].to(self.device)
        self.dones[:size] = data["dones"].to(self.device)
        self._log_gpu_memory("load:end", f"path={filepath}, size={size}")

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.tree.size
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.tree.size >= self.buffer_size

    def __del__(self):
        self._log_gpu_memory("buffer:destroy")


class RankBasedPrioritizedBuffer(PrioritizedReplayBuffer):
    """
    Rank-based prioritized replay buffer.
    Uses rank-based prioritization instead of proportional.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors = np.zeros(self.buffer_size)
        self.error_idx = 0
    
    def _get_priority(self, error: float) -> float:
        """Compute priority based on rank."""
        # Store error
        self.errors[self.error_idx % self.buffer_size] = abs(error)
        self.error_idx += 1
        
        # Compute rank
        valid_errors = self.errors[:min(self.error_idx, self.buffer_size)]
        rank = np.sum(valid_errors < abs(error)) + 1
        
        # Priority = 1 / rank^alpha
        return (1.0 / rank) ** self.alpha
    
    
