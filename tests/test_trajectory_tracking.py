"""CPU-only tests for trajectory-tracking refactor.

Covers:
- ReachingCost accepts (T, 3) targets and returns step-indexed targets
- MPPIController auto-detects step kwarg support and forwards it
- trajectories module: parametric primitives + buffer builder
"""

import numpy as np
import pytest
import torch

from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)
from mppi_cuda.trajectories import (
    Circle, FigureEight, Waypoints, build_target_traj, sample_random_trajectory,
)


# -------- cost class --------

def test_cost_static_target_back_compat():
    """Static (3,) target keeps target_traj=None and uses target_pos."""
    cost = ReachingCost(target_pos=[0.5, 0.0, 0.5])
    assert cost.target_traj is None
    assert cost.target_pos.shape == (3,)
    # _target_at always returns the same thing regardless of step
    t0 = cost._target_at(0)
    t5 = cost._target_at(5)
    t_none = cost._target_at(None)
    assert torch.allclose(t0, t5) and torch.allclose(t0, t_none)


def test_cost_trajectory_target_indexes_per_step():
    """(T, 3) target makes _target_at vary with step."""
    traj = torch.linspace(0, 1, 30).unsqueeze(-1).expand(-1, 3).contiguous()
    cost = ReachingCost(target_pos=traj)
    assert cost.target_traj is not None
    assert cost.target_traj.shape == (30, 3)
    assert torch.allclose(cost._target_at(0), traj[0])
    assert torch.allclose(cost._target_at(10), traj[10])
    # Clamps past the end:
    assert torch.allclose(cost._target_at(999), traj[-1])


def test_cost_set_target_traj_runtime_update():
    """set_target_traj swaps the buffer at runtime without rebuilding cost."""
    cost = ReachingCost(target_pos=[0.5, 0.0, 0.5])
    assert cost.target_traj is None
    new_traj = torch.zeros(20, 3)
    new_traj[:, 0] = torch.linspace(0.4, 0.6, 20)
    cost.set_target_traj(new_traj)
    assert cost.target_traj is not None
    assert cost.target_traj.shape == (20, 3)
    assert torch.allclose(cost._target_at(0), new_traj[0])
    assert torch.allclose(cost._target_at(19), new_traj[19])


def test_cost_running_cost_varies_with_step_when_trajectory_set():
    """running_cost(x, u, step=t) should produce different values across t
    when target varies — otherwise the trajectory plumbing isn't actually wired."""
    traj = torch.tensor([
        [0.5, 0.0, 0.5],
        [0.6, 0.1, 0.5],
        [0.7, 0.2, 0.5],
    ], dtype=torch.float32)
    cost = ReachingCost(target_pos=traj, w_pos=100.0)
    q = torch.tensor(FRANKA_HOME_Q, dtype=torch.float32)
    qdot = torch.zeros(7)
    x = torch.cat([q, qdot])
    u = torch.zeros(7)
    c0 = cost.running_cost(x, u, step=0).item()
    c1 = cost.running_cost(x, u, step=1).item()
    c2 = cost.running_cost(x, u, step=2).item()
    assert c0 != c1 != c2


# -------- controller plumbing --------

def test_mppi_controller_detects_step_kwarg_on_cost():
    """The bound method `ReachingCost.running_cost` advertises a `step` kwarg
    and the controller should auto-detect that and pass it."""
    cost = ReachingCost(target_pos=[0.5, 0.0, 0.5])
    dyn = DoubleIntegratorArm(dt=0.02)
    ctrl = MPPIController(
        dynamics=dyn.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7, horizon=20, num_samples=64,
        sigma=2.0, temperature=1.0,
        u_min=-10.0, u_max=10.0,
        device="cpu", seed=0,
    )
    assert ctrl._rc_takes_step is True
    assert ctrl._tc_takes_step is True


def test_mppi_controller_back_compat_without_step_kwarg():
    """A user-provided lambda without step kwarg must still work — controller
    falls back to calling it positionally."""
    target = torch.tensor([0.5, 0.0, 0.5])
    def my_running(x, u):                 # no step kwarg
        q = x[..., :7]
        # Trivial cost so we don't depend on FK
        return (q ** 2).sum(-1) + (u ** 2).sum(-1)

    dyn = DoubleIntegratorArm(dt=0.02)
    ctrl = MPPIController(
        dynamics=dyn.step,
        running_cost=my_running,
        terminal_cost=None,
        action_dim=7, horizon=10, num_samples=32,
        sigma=2.0, temperature=1.0,
        u_min=-10.0, u_max=10.0,
        device="cpu", seed=0,
    )
    assert ctrl._rc_takes_step is False
    # Should not raise:
    x = torch.zeros(14)
    u = ctrl.step(x)
    assert u.shape == (7,)


def test_mppi_controller_step_runs_to_completion_with_trajectory_cost():
    """End-to-end: cost with a (T, 3) trajectory, controller produces finite actions."""
    H = 20
    # A small linear ramp in EE space.
    traj = torch.zeros(H + 1, 3)
    traj[:, 0] = torch.linspace(0.4, 0.5, H + 1)
    traj[:, 2] = 0.5
    cost = ReachingCost(target_pos=traj, w_pos=200.0)
    dyn = DoubleIntegratorArm(dt=0.02)
    ctrl = MPPIController(
        dynamics=dyn.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7, horizon=H, num_samples=128,
        sigma=2.0, temperature=1.0,
        u_min=-20.0, u_max=20.0,
        device="cpu", seed=0,
    )
    x = torch.cat([torch.tensor(FRANKA_HOME_Q, dtype=torch.float32), torch.zeros(7)])
    u = ctrl.step(x)
    assert torch.isfinite(u).all()
    assert u.shape == (7,)


# -------- trajectories module --------

def test_circle_repeats_at_period():
    c = Circle(center=(0.5, 0.0, 0.5), radius=0.1, period=4.0)
    p0 = c(0.0)
    p_period = c(4.0)
    p_half   = c(2.0)
    np.testing.assert_allclose(p0, p_period, atol=1e-6)
    # Diametrically opposite at half period
    np.testing.assert_allclose(p_half[:2], 2 * np.array(c.center[:2]) - p0[:2], atol=1e-6)


def test_figure_eight_passes_through_center_at_zero():
    f = FigureEight(center=(0.5, 0.0, 0.5), radius=0.1, period=6.0)
    p0 = f(0.0)
    np.testing.assert_allclose(p0, [0.5, 0.0, 0.5], atol=1e-6)


def test_waypoints_lerp_endpoints():
    pts = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5)]
    wp = Waypoints(points=pts, segment_time=2.0)
    np.testing.assert_allclose(wp(0.0), pts[0], atol=1e-6)
    np.testing.assert_allclose(wp(2.0), pts[1], atol=1e-6)
    np.testing.assert_allclose(wp(1.0), [0.5, 0.0, 0.5], atol=1e-6)


def test_build_target_traj_shape_and_first_sample():
    fn = Circle(center=(0.5, 0.0, 0.5), radius=0.1, period=4.0)
    buf = build_target_traj(fn, t_now=0.0, dt=0.02, horizon=40)
    assert buf.shape == (41, 3)
    np.testing.assert_allclose(buf[0], fn(0.0), atol=1e-6)


def test_sample_random_trajectory_returns_callable():
    """Smoke: should return a Callable that produces finite (3,)."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        fn = sample_random_trajectory(rng)
        p = fn(1.234)
        assert p.shape == (3,)
        assert np.isfinite(p).all()
