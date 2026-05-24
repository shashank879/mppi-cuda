"""Dynamics models for MPPI rollouts.

For Day 1 we use a simple double-integrator in joint space:
the control input is interpreted as joint acceleration, velocity
is integrated semi-implicitly, then position. Velocity is hard-clamped
to the manipulator's velocity limits.

This is the "predictive model" inside MPPI. It does not need to match
real physics exactly; it needs to be fast and roughly directionally
correct. Real-arm experiments later will swap in a learned MLP or
MuJoCo-derived dynamics behind the same interface.
"""

from __future__ import annotations
import math
import torch


# Franka Panda joint limits (radians).
FRANKA_Q_MIN = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
FRANKA_Q_MAX = ( 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973)
# Velocity limits (rad/s).
FRANKA_QDOT_MAX = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)


class DoubleIntegratorArm:
    """7-DoF arm modeled as a double integrator in joint space.

    State:   x = [q (7,), qdot (7,)]   (14,)
    Control: u = q_ddot (7,)           (7,)

    Optional `gravity_bias` is a (7,) tensor of joint accelerations that
    are added to the commanded `u` inside step(). Used to inject gravity
    compensation: the controller plans against a model where the arm
    "knows" gravity is pulling it down, so it commands the right counter
    automatically. The bias is updated externally (once per MPPI tick)
    rather than computed analytically inside dynamics — see
    `mppi_cuda.gravity.compute_gravity_acceleration`.
    """

    state_dim = 14
    control_dim = 7

    def __init__(
        self,
        dt: float = 0.02,
        q_min=FRANKA_Q_MIN,
        q_max=FRANKA_Q_MAX,
        qdot_max=FRANKA_QDOT_MAX,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        gravity_bias=None,
    ):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.q_min = torch.as_tensor(q_min, device=device, dtype=dtype)
        self.q_max = torch.as_tensor(q_max, device=device, dtype=dtype)
        self.qdot_max = torch.as_tensor(qdot_max, device=device, dtype=dtype)
        self.gravity_bias = (
            torch.as_tensor(gravity_bias, device=device, dtype=dtype)
            if gravity_bias is not None else None
        )

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.step(x, u)

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Advance state one timestep. Supports arbitrary leading batch dims."""
        q = x[..., :7]
        qdot = x[..., 7:]

        # Effective acceleration = controller's command + gravity-induced bias.
        u_eff = u if self.gravity_bias is None else u + self.gravity_bias

        # Semi-implicit Euler: integrate velocity first, then position.
        qdot_new = qdot + self.dt * u_eff
        qdot_new = torch.clamp(qdot_new, -self.qdot_max, self.qdot_max)
        q_new = q + self.dt * qdot_new

        return torch.cat([q_new, qdot_new], dim=-1)
