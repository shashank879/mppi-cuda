"""FK consistency: our DH-based FK vs MuJoCo's kinematics on the same arm.

If our forward_kinematics() and MuJoCo's mj_forward agree on the
end-effector position for several joint configurations, we can trust
both our cost function (which uses our FK) and the MuJoCo plant
(which we'll add in Day 2b).

Discrepancies usually mean one of:
  - DH parameter mismatch
  - Reference frame mismatch (base offset, tool offset)
  - Joint sign convention mismatch
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import mujoco

from mppi_cuda.kinematics import forward_kinematics, FRANKA_HOME_Q


MODEL_PATH = (
    Path(__file__).parent.parent / "assets" / "franka_panda" / "panda_nohand.xml"
).resolve()


@pytest.fixture(scope="module")
def model_and_data():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def _mujoco_ee_position(model, data, q):
    """Read MuJoCo's attachment_site world position for a given q."""
    data.qpos[:7] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return np.array(data.site_xpos[site_id])


def _ours_ee_position(q):
    q_t = torch.as_tensor(q, dtype=torch.float64)
    pos, _ = forward_kinematics(q_t)
    return pos.numpy()


@pytest.mark.parametrize(
    "q_label, q",
    [
        ("zero", np.zeros(7)),
        ("home", np.array(FRANKA_HOME_Q)),
        ("random_1", np.array([0.5, -0.3, 0.2, -2.0, 0.4, 2.0, 0.6])),
        ("random_2", np.array([-1.0, -1.2, 1.5, -2.5, -0.5, 1.0, -0.5])),
        ("random_3", np.array([1.5, 0.5, -0.8, -1.5, 1.2, 0.8, 1.0])),
    ],
)


def test_fk_matches_mujoco(model_and_data, q_label, q):
    model, data = model_and_data
    mj_pos = _mujoco_ee_position(model, data, q)
    our_pos = _ours_ee_position(q)
    err = np.linalg.norm(mj_pos - our_pos)
    print(f"\n[{q_label}] MuJoCo: {mj_pos}  Ours: {our_pos}  err: {err*1000:.3f} mm")
    assert err < 1e-3, (
        f"FK mismatch at {q_label}: MuJoCo {mj_pos} vs ours {our_pos}, err {err*1000:.2f} mm"
    )
