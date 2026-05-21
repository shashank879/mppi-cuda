"""Behavioral tests for the MPPI controller."""

import math
import torch

from mppi_cuda import (
    MPPIController,
    DoubleIntegratorArm,
    ReachingCost,
    forward_kinematics,
    FRANKA_HOME_Q,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
)


def _make_controller(device="cpu"):
    plant = DoubleIntegratorArm(dt=0.02, device=device)
    cost = ReachingCost(
        target_pos=[0.5, 0.3, 0.5],
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        device=device,
    )
    return plant, cost, MPPIController(
        dynamics=plant.step,
        running_cost=cost.running_cost,
        terminal_cost=cost.terminal_cost,
        action_dim=7,
        horizon=20,
        num_samples=256,
        sigma=2.0,
        temperature=1.0,
        u_min=-10.0,
        u_max=10.0,
        device=device,
        seed=42,
    )


def test_step_output_shape_and_range():
    plant, cost, ctrl = _make_controller()
    x = torch.cat([torch.tensor(FRANKA_HOME_Q, dtype=torch.float32), torch.zeros(7)])
    u = ctrl.step(x)
    assert u.shape == (7,)
    assert (u >= -10.0).all() and (u <= 10.0).all()


def test_warm_start_persists_across_calls():
    """After a step the nominal U should be shifted and non-zero in early entries."""
    plant, cost, ctrl = _make_controller()
    x = torch.cat([torch.tensor(FRANKA_HOME_Q, dtype=torch.float32), torch.zeros(7)])
    _ = ctrl.step(x)
    # Last entry should be zeroed (the shift policy)
    assert torch.allclose(ctrl.U[-1], torch.zeros(7))
    # Some earlier entry should be non-trivially non-zero
    assert ctrl.U[:5].abs().max() > 1e-3


def test_reaches_target_within_tolerance():
    """The headline behavioural test: closed-loop reach should converge."""
    plant, cost, ctrl = _make_controller()
    target = cost.target_pos

    q = torch.tensor(FRANKA_HOME_Q, dtype=torch.float32)
    qdot = torch.zeros(7)
    x = torch.cat([q, qdot])

    for _ in range(150):
        u = ctrl.step(x)
        x = plant.step(x, u)

    ee, _ = forward_kinematics(x[:7].unsqueeze(0))
    err = (ee.squeeze(0) - target).norm().item()
    assert err < 0.02, f"final EE error too high: {err*1000:.2f} mm"


def test_seed_determinism():
    """Two controllers with same seed should produce same first action."""
    plant, cost, ctrl1 = _make_controller()
    plant2, cost2, ctrl2 = _make_controller()
    x = torch.cat([torch.tensor(FRANKA_HOME_Q, dtype=torch.float32), torch.zeros(7)])
    u1 = ctrl1.step(x.clone())
    u2 = ctrl2.step(x.clone())
    assert torch.allclose(u1, u2), "same seed -> different output"
