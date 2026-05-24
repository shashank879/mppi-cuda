"""Analytic dynamics utilities derived from MuJoCo.

These compute quantities like the gravity-induced joint acceleration
`q̈_g(q) = -M(q)⁻¹ g(q)` that we want to inject into the controller's
otherwise-simple predictive model. They use MuJoCo internally but are
called sparingly (once per MPPI tick, not once per rollout step), so
the cost is negligible.

The CUDA kernel will reuse the *output* of these utilities — the
controller computes a per-tick gravity bias on the CPU, then ships it
to the GPU as a small constant vector that every rollout uses.
"""

from __future__ import annotations
import numpy as np


def compute_gravity_acceleration(model, data, q: np.ndarray) -> np.ndarray:
    """Joint accelerations induced by gravity at configuration q.

    The free-dynamics equation with no applied torque is::

        M(q) q̈ + g(q) = 0   =>   q̈ = -M(q)⁻¹ g(q)

    To extract g(q) we use mj_inverse with qvel=0, qacc=0: that
    leaves qfrc_inverse = g(q) (Coriolis vanishes when qvel=0).
    Then mj_solveM solves M·x = qfrc_inverse for x.

    Args:
        model: mujoco.MjModel
        data:  mujoco.MjData (scratch; mutated)
        q:     (7,) joint angles

    Returns:
        q_acc_gravity: (7,) negative of the acceleration controller
                       should counter-command. Sign convention is such
                       that adding this to the controller's `u` causes
                       its predictive model to match free-fall.
    """
    import mujoco

    data.qpos[:7] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    # qfrc_inverse is the torque needed to maintain (q, 0, 0). With
    # qvel=0 and qacc=0 that's exactly g(q).
    g_q = data.qfrc_inverse[:7].copy()

    # Solve M(q) x = g(q) for x = M⁻¹ g.
    x = np.zeros(model.nv)
    x_in = np.zeros_like(x)
    x_in[:7] = g_q
    mujoco.mj_solveM(model, data, x.reshape(1, -1), x_in.reshape(1, -1))

    # q̈ induced by gravity is -M⁻¹ g. Clip to guard against numerical
    # blow-up at near-singular configurations (e.g. q=0 for Franka, where
    # joint axes align and M becomes ill-conditioned). 50 rad/s² is well
    # above any physically meaningful gravity acceleration on a Franka.
    return np.clip(-x[:7], -50.0, 50.0).astype(np.float32)
