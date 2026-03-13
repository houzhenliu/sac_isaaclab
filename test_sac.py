#!/usr/bin/env python3
"""
Quick test script to verify SAC implementation works correctly.
Run this to check if all components can be imported and basic functionality works.
"""

import sys
import torch
import numpy as np
import gymnasium as gym

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from modules import SAC, SACPolicy, QNetwork, set_seed
        print("✓ modules imported successfully")
    except Exception as e:
        print(f"✗ modules import failed: {e}")
        return False
    
    try:
        from network import GaussianActor, QCritic, TwinQCritic
        print("✓ network imported successfully")
    except Exception as e:
        print(f"✗ network import failed: {e}")
        return False
    
    try:
        from storage import ReplayBuffer, PrioritizedReplayBuffer
        print("✓ storage imported successfully")
    except Exception as e:
        print(f"✗ storage import failed: {e}")
        return False
    
    try:
        from runners import OfflineRunner
        print("✓ runners imported successfully")
    except Exception as e:
        print(f"✗ runners import failed: {e}")
        return False
    
    return True


def test_policy():
    """Test policy network."""
    print("\nTesting policy network...")
    
    try:
        from modules import SACPolicy
        
        # Create dummy spaces
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Create policy
        policy = SACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network_features=[64, 64],
        )
        
        # Test forward pass
        obs = torch.randn(4, 8)
        mean, log_std, _ = policy.compute({"states": obs})
        
        assert mean.shape == (4, 2), f"Expected shape (4, 2), got {mean.shape}"
        assert log_std.shape == (4, 2), f"Expected shape (4, 2), got {log_std.shape}"
        
        print("✓ Policy network test passed")
        return True
    except Exception as e:
        print(f"✗ Policy network test failed: {e}")
        return False


def test_critic():
    """Test critic network."""
    print("\nTesting critic network...")
    
    try:
        from modules import QNetwork
        
        # Create dummy spaces
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Create critic
        critic = QNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network_features=[64, 64],
        )
        
        # Test forward pass
        obs = torch.randn(4, 8)
        action = torch.randn(4, 2)
        
        q_value, _, _ = critic.compute({"states": obs, "taken_actions": action})
        
        assert q_value.shape == (4, 1), f"Expected shape (4, 1), got {q_value.shape}"
        
        print("✓ Critic network test passed")
        return True
    except Exception as e:
        print(f"✗ Critic network test failed: {e}")
        return False


def test_replay_buffer():
    """Test replay buffer."""
    print("\nTesting replay buffer...")
    
    try:
        from storage import ReplayBuffer
        
        buffer = ReplayBuffer(
            buffer_size=1000,
            observation_shape=(8,),
            action_shape=(2,),
            device="cpu",
        )
        
        # Add some transitions
        for i in range(10):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(8).astype(np.float32)
            done = False
            
            buffer.add(obs, action, reward, next_obs, done)
        
        assert len(buffer) == 10, f"Expected buffer size 10, got {len(buffer)}"
        
        # Sample batch
        batch = buffer.sample(4)
        
        assert batch["observations"].shape == (4, 8)
        assert batch["actions"].shape == (4, 2)
        
        print("✓ Replay buffer test passed")
        return True
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False


def test_sac_agent():
    """Test SAC agent creation."""
    print("\nTesting SAC agent...")
    
    try:
        from modules import SAC, SACPolicy, QNetwork
        
        # Create dummy spaces
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Create networks
        policy = SACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network_features=[64, 64],
        )
        
        critic_1 = QNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network_features=[64, 64],
        )
        
        critic_2 = QNetwork(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network_features=[64, 64],
        )
        
        models = {
            "policy": policy,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": None,
            "target_critic_2": None,
        }
        
        agent_cfg = {
            "batch_size": 32,
            "learning_starts": 100,
        }
        
        agent = SAC(
            models=models,
            memory=None,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            cfg=agent_cfg,
        )
        
        # Test action selection
        obs = torch.randn(4, 8)
        actions = agent.act(obs, timestep=1000, eval_mode=False)
        
        assert actions.shape == (4, 2), f"Expected shape (4, 2), got {actions.shape}"
        
        print("✓ SAC agent test passed")
        return True
    except Exception as e:
        print(f"✗ SAC agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("SAC Implementation Test Suite")
    print("=" * 50)
    
    # Print environment info
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Gymnasium: {gym.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Policy", test_policy()))
    results.append(("Critic", test_critic()))
    results.append(("Replay Buffer", test_replay_buffer()))
    results.append(("SAC Agent", test_sac_agent()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
