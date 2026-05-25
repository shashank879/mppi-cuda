"""Build script for the CUDA extension.

Project metadata lives in pyproject.toml; this file only handles the
CUDA extension that pyproject.toml can't express. The two work together:
running `pip install -e .` picks up the metadata from pyproject and the
extension from here.

Requires a CUDA toolkit installed; on the dev Dockerfile that's
nvidia/cuda:12.4.1-devel-ubuntu22.04 plus the PyTorch cu124 wheels.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Architectures we compile for. The dev Dockerfile pins
# TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" by default; override with
#   TORCH_CUDA_ARCH_LIST=8.9 pip install -e .
# to halve build time when you know the target GPU.
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")

extra_compile_args = {
    "cxx":  ["-O3", "-Wall"],
    "nvcc": [
        "-O3",
        "--use_fast_math",            # cosf/sinf intrinsics; matches PyTorch
        "--expt-relaxed-constexpr",
        # Surface register pressure during compilation so we know if v0
        # spills. Anything > 128/thread on Ampere/Hopper warrants a fix.
        "--ptxas-options=-v",
    ],
}

setup(
    ext_modules=[
        CUDAExtension(
            name="mppi_cuda._kernels",
            sources=["csrc/mppi_rollout.cu"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
