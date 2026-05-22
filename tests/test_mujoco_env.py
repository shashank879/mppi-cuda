"""Basic tests for the MuJoCo Franka env."""

import numpy as np
import pytest

from mppi_cuda import MujocoFrankaEnv, FRANKA_HOME_Q, forward_kinematics
import torch


@pytest.fixture(scope="module")
def env():
    e = MujocoFrankaEnv(control_dt=0.02)
    yield e
    e.close()


def test_state_action_shape(env):
    s = env.reset()
    assert s.shape == (14,)
    assert s.dtype == np.float32
    assert env.state_dim == 14
    assert env.action_dim == 7


def test_reset_home_is_at_zero_velocity(env):
    s = env.reset()
    np.testing.assert_allclose(s[:7], FRANKA_HOME_Q, atol=1e-5)
    np.testing.assert_allclose(s[7:], 0.0, atol=1e-5)


def test_reset_with_state(env):
    custom = np.concatenate([np.array(FRANKA_HOME_Q) + 0.1, np.zeros(7)])
    s = env.reset(initial_state=custom)
    np.testing.assert_allclose(s[:7], custom[:7], atol=1e-5)


def test_step_advances_under_gravity(env):
    """With ctrl=home, the arm should stay near home. With ctrl ≠ home, it moves."""
    s0 = env.reset()
    # Holding the current pose
    s1 = env.step(s0[:7])
    pos_change_hold = np.linalg.norm(s1[:7] - s0[:7])
    assert pos_change_hold < 0.05, f"arm drifted holding pose: {pos_change_hold} rad"

    # Commanding a different pose
    s0 = env.reset()
    target = s0[:7] + 0.2
    s1 = env.step(target)
    pos_change_move = np.linalg.norm(s1[:7] - s0[:7])
    assert pos_change_move > pos_change_hold, "commanded motion smaller than hold drift"


def test_ee_position_matches_our_fk(env):
    """The env's reported EE matches our DH-based FK."""
    s = env.reset()
    env_ee = env.ee_position
    our_ee, _ = forward_kinematics(torch.from_numpy(s[:7]).double())
    np.testing.assert_allclose(env_ee, our_ee.numpy(), atol=1e-4)
