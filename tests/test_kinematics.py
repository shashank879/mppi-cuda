"""Forward kinematics sanity checks."""

import math
import torch
import pytest

from mppi_cuda.kinematics import (
    forward_kinematics,
    FRANKA_DH,
    FRANKA_FLANGE_D,
    FRANKA_HOME_Q,
)


def test_fk_shapes_unbatched():
    q = torch.zeros(7)
    pos, rot = forward_kinematics(q)
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)


def test_fk_shapes_batched():
    q = torch.randn(11, 5, 7)
    pos, rot = forward_kinematics(q)
    assert pos.shape == (11, 5, 3)
    assert rot.shape == (11, 5, 3, 3)


def test_fk_rotation_is_orthonormal():
    q = torch.randn(8, 7)
    _, R = forward_kinematics(q)
    I = torch.eye(3).expand(8, 3, 3)
    err = (R @ R.transpose(-1, -2) - I).abs().max()
    assert err < 1e-5, f"R R^T not identity, max err {err}"


def test_fk_home_position_is_reasonable():
    q = torch.tensor(FRANKA_HOME_Q)
    pos, _ = forward_kinematics(q)
    # Franka home is the standard "ready" pose; EE should be in front and above the base.
    assert pos[2] > 0.3, f"EE z too low: {pos}"
    assert pos[2] < 1.0, f"EE z too high: {pos}"
    # Reach in front of the base — depends on home pose; the standard one is ~0.3-0.5 m forward.
    assert pos[:2].norm() > 0.1, f"EE too close to base axis: {pos}"


def test_fk_zero_q_position():
    """At q=0 with Franka DH, EE should be on the z-axis (no joints rotated)."""
    q = torch.zeros(7)
    pos, _ = forward_kinematics(q)
    # All a_i are small, mostly zero — at q=0 the EE should be near the z-axis.
    # Allow some xy offset from the small a values that don't cancel.
    assert pos[:2].abs().max() < 0.2, f"unexpected xy at q=0: {pos}"
