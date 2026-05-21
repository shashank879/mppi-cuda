"""mppi-cuda — fused CUDA MPPI kernels for robot manipulation.

PyTorch baseline first; CUDA kernels will be added in csrc/.
"""

from .controller import MPPIController
from .dynamics import (
    DoubleIntegratorArm,
    FRANKA_Q_MIN,
    FRANKA_Q_MAX,
    FRANKA_QDOT_MAX,
)
from .costs import ReachingCost
from .kinematics import (
    forward_kinematics,
    FRANKA_DH,
    FRANKA_FLANGE_D,
    FRANKA_HOME_Q,
)

__version__ = "0.0.1"
__all__ = [
    "MPPIController",
    "DoubleIntegratorArm",
    "ReachingCost",
    "forward_kinematics",
    "FRANKA_DH",
    "FRANKA_FLANGE_D",
    "FRANKA_HOME_Q",
    "FRANKA_Q_MIN",
    "FRANKA_Q_MAX",
    "FRANKA_QDOT_MAX",
]
