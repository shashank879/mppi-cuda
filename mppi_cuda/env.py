"""Robot environments for the MPPI controller.

The `RobotEnv` ABC defines the minimal interface the controller needs
from any environment: reset, step, get_state, render. New environments
(other arms, humanoid bases, Isaac variants) just implement this ABC
and the controller code stays unchanged.

`MujocoFrankaEnv` is the concrete MuJoCo-backed Franka Panda
implementation used for Day 2 and beyond.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class RobotEnv(ABC):
    """Minimal interface every environment must provide.

    Subclasses must set the three class attrs (state_dim, action_dim,
    control_dt) and implement reset/step/get_state. render() is optional.
    """

    state_dim: int
    action_dim: int
    control_dt: float

    @abstractmethod
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset to initial_state (or a default), return the new state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply action, advance dynamics by control_dt, return next state."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return current state without advancing dynamics."""

    def render(self) -> Optional[np.ndarray]:
        """Return RGB frame of current state, or None if not supported."""
        return None

    def close(self) -> None:
        """Release any resources (renderers, viewers)."""


def _default_franka_xml() -> Path:
    """Locate the Franka panda_nohand.xml across common install layouts."""
    # 1. Vendored assets inside an editable repo install.
    repo = Path(__file__).resolve().parent.parent / "assets" / "franka_panda" / "panda_nohand.xml"
    if repo.exists():
        return repo
    # 2. MUJOCO_MENAGERIE_PATH (set by the dev Dockerfile, or by users who clone it).
    env_path = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if env_path:
        cand = Path(env_path) / "franka_emika_panda" / "panda_nohand.xml"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not locate panda_nohand.xml. Either vendor it under "
        "assets/franka_panda/, set MUJOCO_MENAGERIE_PATH, or pass "
        "model_path explicitly to MujocoFrankaEnv()."
    )


class MujocoFrankaEnv(RobotEnv):
    """7-DoF Franka Panda in MuJoCo, with position-controlled actuators.

    State:  x = [q (7,), qdot (7,)]   (14,)
    Action: q_target (7,)             — position target sent to the
                                        PD-configured general actuators.

    The actuator does internal PD tracking; the bridge from MPPI
    acceleration commands to q_target lives in the demo script, not here.
    """

    state_dim = 14
    action_dim = 7

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        control_dt: float = 0.02,
        render_size: tuple[int, int] = (480, 640),
    ):
        # Imported here so the package is importable without MuJoCo installed
        # (e.g. for tests of the pure-Python controller).
        import mujoco

        self._mujoco = mujoco

        path = Path(model_path) if model_path else _default_franka_xml()
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self.control_dt = control_dt
        self.sim_dt = self.model.opt.timestep
        # Number of physics sub-steps per control tick.
        n = control_dt / self.sim_dt
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"control_dt ({control_dt}) must be an integer multiple of "
                f"sim_dt ({self.sim_dt})."
            )
        self.n_sim_steps = int(round(n))

        # The attachment_site is the canonical Franka tool0 frame.
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        self._renderer = None
        self._render_size = render_size

        # Sensible default home pose.
        from .kinematics import FRANKA_HOME_Q
        self._default_q = np.asarray(FRANKA_HOME_Q, dtype=np.float64)

    # ---------- RobotEnv interface ----------

    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        self._mujoco.mj_resetData(self.model, self.data)
        if initial_state is None:
            self.data.qpos[:7] = self._default_q
            self.data.qvel[:7] = 0.0
        else:
            initial_state = np.asarray(initial_state, dtype=np.float64)
            self.data.qpos[:7] = initial_state[:7]
            self.data.qvel[:7] = initial_state[7:14]

        # Initialize ctrl to current qpos so the PD actuators start at rest
        # (otherwise ctrl=0 would yank the arm toward a different pose at t=0).
        self.data.ctrl[:7] = self.data.qpos[:7]
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}")
        self.data.ctrl[:7] = action
        for _ in range(self.n_sim_steps):
            self._mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.concatenate(
            [self.data.qpos[:7], self.data.qvel[:7]]
        ).astype(np.float32)

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model,
                height=self._render_size[0],
                width=self._render_size[1],
            )
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------- Useful extras ----------

    @property
    def ee_position(self) -> np.ndarray:
        """Current end-effector position in the base frame."""
        return np.array(self.data.site_xpos[self._ee_site_id])
