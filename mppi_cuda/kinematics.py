"""Forward kinematics for serial manipulators using modified DH parameters.

Implementation note: we use Craig's modified DH convention, where
each row of the DH table is (a_{i-1}, d_i, alpha_{i-1}) and the
joint angle theta_i is supplied separately at evaluation time.

The transform for joint i is:

    T_i = | cos(t)         -sin(t)         0          a       |
          | sin(t)*cos(a)  cos(t)*cos(a)  -sin(a)    -d*sin(a) |
          | sin(t)*sin(a)  cos(t)*sin(a)   cos(a)     d*cos(a) |
          | 0              0               0          1        |

where (a, alpha, d) are the row's parameters and t is the joint angle.
"""

from __future__ import annotations
import math
import torch


# Franka Panda modified DH parameters
# Each row: (a_{i-1}, d_i, alpha_{i-1})
FRANKA_DH = (
    (0.0,      0.333,  0.0),
    (0.0,      0.0,    -math.pi / 2),
    (0.0,      0.316,  math.pi / 2),
    (0.0825,   0.0,    math.pi / 2),
    (-0.0825,  0.384,  -math.pi / 2),
    (0.0,      0.0,    math.pi / 2),
    (0.088,    0.0,    math.pi / 2),
)

# Flange offset (additional pure translation along the final z-axis to the TCP).
FRANKA_FLANGE_D = 0.107

# A common Franka "ready" pose - elbow bent, EE forward and up.
FRANKA_HOME_Q = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)


def _dh_matrix(a: float, d: float, alpha: float, theta: torch.Tensor) -> torch.Tensor:
    """Build a (..., 4, 4) modified-DH transform.

    `a`, `d`, `alpha` are scalars (compile-time constants for a given joint).
    `theta` is a tensor of any shape — it broadcasts.
    """
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    ct = torch.cos(theta)
    st = torch.sin(theta)

    # Assemble row-by-row. Most entries either depend only on theta or are constants.
    T = torch.zeros(*theta.shape, 4, 4, device=theta.device, dtype=theta.dtype)
    T[..., 0, 0] = ct
    T[..., 0, 1] = -st
    T[..., 0, 2] = 0.0
    T[..., 0, 3] = a
    T[..., 1, 0] = st * ca
    T[..., 1, 1] = ct * ca
    T[..., 1, 2] = -sa
    T[..., 1, 3] = -d * sa
    T[..., 2, 0] = st * sa
    T[..., 2, 1] = ct * sa
    T[..., 2, 2] = ca
    T[..., 2, 3] = d * ca
    T[..., 3, 3] = 1.0
    return T


def forward_kinematics(
    q: torch.Tensor,
    dh: tuple = FRANKA_DH,
    flange_d: float = FRANKA_FLANGE_D,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute end-effector pose from joint angles.

    Args:
        q: (..., n_joints) joint angles in radians.
        dh: sequence of (a, d, alpha) tuples, one per joint.
        flange_d: extra translation along the final z-axis (e.g. tool flange).

    Returns:
        ee_pos: (..., 3) end-effector position in base frame.
        ee_rot: (..., 3, 3) end-effector rotation matrix.
    """
    n_joints = len(dh)
    assert q.shape[-1] == n_joints, f"q must have last dim {n_joints}, got {q.shape[-1]}"

    batch_shape = q.shape[:-1]
    T = (
        torch.eye(4, device=q.device, dtype=q.dtype)
        .expand(*batch_shape, 4, 4)
        .clone()
    )

    for i, (a, d, alpha) in enumerate(dh):
        Ti = _dh_matrix(a, d, alpha, q[..., i])
        T = T @ Ti

    if flange_d != 0.0:
        # Pure translation along z by flange_d.
        flange = (
            torch.eye(4, device=q.device, dtype=q.dtype)
            .expand(*batch_shape, 4, 4)
            .clone()
        )
        flange[..., 2, 3] = flange_d
        T = T @ flange

    ee_pos = T[..., :3, 3]
    ee_rot = T[..., :3, :3]
    return ee_pos, ee_rot
