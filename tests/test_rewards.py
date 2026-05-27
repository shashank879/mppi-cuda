"""Tests for the trajectory-tracking reward function."""

import numpy as np
import pytest
import torch

from mppi_cuda.rewards import tracking_reward, recompute_reward_torch


def test_reward_zero_when_perfect():
    """No tracking error, no action, no jerk → reward is exactly 0."""
    ee = np.array([0.5, 0.0, 0.5])
    target = np.array([0.5, 0.0, 0.5])
    u = np.zeros(7)
    r = tracking_reward(ee, u, target, action_prev=np.zeros(7))
    assert float(r) == 0.0


def test_reward_strictly_nonpositive():
    """Reward is the negation of a sum of squared quantities."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        ee = rng.standard_normal(3)
        target = rng.standard_normal(3)
        u = rng.standard_normal(7)
        u_prev = rng.standard_normal(7)
        r = tracking_reward(ee, u, target, action_prev=u_prev)
        assert float(r) <= 0.0


def test_reward_components_decomposable():
    """Tracking, effort, jerk should each contribute independently."""
    ee = np.array([0.5, 0.0, 0.5])
    target = np.array([0.6, 0.0, 0.5])      # tracking error of 0.1m
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])   # effort = 1
    u_prev = np.zeros(7)                    # jerk = 1

    r = tracking_reward(ee, u, target, u_prev,
                        alpha_u=0.005, alpha_du=0.01)
    # expected = -(0.01 + 0.005*1 + 0.01*1) = -0.025
    np.testing.assert_allclose(float(r), -0.025, atol=1e-6)


def test_reward_no_prev_action_no_jerk():
    """action_prev=None should give zero jerk (no penalty for first action)."""
    ee = np.array([0.5, 0.0, 0.5])
    target = np.array([0.5, 0.0, 0.5])
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])
    r_with_zero_prev = tracking_reward(ee, u, target,
                                        action_prev=np.zeros(7),
                                        alpha_u=0.005, alpha_du=0.01)
    r_with_none = tracking_reward(ee, u, target,
                                   action_prev=None,
                                   alpha_u=0.005, alpha_du=0.01)
    # When action_prev=zeros, jerk_sq = ||u||² = 1, contributes -0.01.
    # When action_prev=None, jerk_sq = 0.
    # Difference should be exactly the jerk term (-0.01).
    np.testing.assert_allclose(float(r_with_none) - float(r_with_zero_prev),
                                0.01, atol=1e-6)


def test_reward_batched_shape():
    """Reward should broadcast across leading batch dims."""
    B = 32
    ee = np.random.randn(B, 3)
    target = np.random.randn(B, 3)
    u = np.random.randn(B, 7)
    u_prev = np.random.randn(B, 7)
    r = tracking_reward(ee, u, target, u_prev)
    assert r.shape == (B,)


def test_numpy_torch_parity():
    """The numpy and torch versions of the reward must agree to fp32 tolerance."""
    rng = np.random.default_rng(42)
    ee_np = rng.standard_normal((64, 3)).astype(np.float64)
    target_np = rng.standard_normal((64, 3)).astype(np.float64)
    u_np = rng.standard_normal((64, 7)).astype(np.float64)
    u_prev_np = rng.standard_normal((64, 7)).astype(np.float64)

    r_np = tracking_reward(ee_np, u_np, target_np, u_prev_np,
                            alpha_u=0.005, alpha_du=0.01)

    ee = torch.from_numpy(ee_np)
    target = torch.from_numpy(target_np)
    u = torch.from_numpy(u_np)
    u_prev = torch.from_numpy(u_prev_np)
    r_torch = recompute_reward_torch(ee, u, target, u_prev,
                                      alpha_u=0.005, alpha_du=0.01)

    np.testing.assert_allclose(r_torch.numpy(), r_np, rtol=1e-6, atol=1e-9)


def test_reward_scales_with_alphas():
    """Increasing alphas must make the reward more negative for fixed inputs."""
    ee = np.array([0.5, 0.0, 0.5])
    target = np.array([0.5, 0.0, 0.5])
    u = np.array([1.0, 0, 0, 0, 0, 0, 0])
    u_prev = np.zeros(7)

    r_low  = tracking_reward(ee, u, target, u_prev, alpha_u=0.001, alpha_du=0.001)
    r_high = tracking_reward(ee, u, target, u_prev, alpha_u=0.100, alpha_du=0.100)
    assert float(r_high) < float(r_low) <= 0.0
