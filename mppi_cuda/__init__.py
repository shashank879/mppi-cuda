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
from .env import RobotEnv, MujocoFrankaEnv

# CudaMPPIController is gated on the compiled extension. Don't fail
# import if it's missing (e.g. CPU-only install) — users on CPU should
# still be able to use the rest of the package.
try:
    from .cuda_controller import CudaMPPIController
    _CUDA_AVAILABLE = True
except ImportError:
    CudaMPPIController = None
    _CUDA_AVAILABLE = False

__version__ = "0.0.1"
__all__ = [
    "MPPIController",
    "CudaMPPIController",
    "DoubleIntegratorArm",
    "ReachingCost",
    "forward_kinematics",
    "RobotEnv",
    "MujocoFrankaEnv",
    "FRANKA_DH",
    "FRANKA_FLANGE_D",
    "FRANKA_HOME_Q",
    "FRANKA_Q_MIN",
    "FRANKA_Q_MAX",
    "FRANKA_QDOT_MAX",
]
