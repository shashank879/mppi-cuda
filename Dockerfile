# ---------- mppi-cuda dev image ----------
# Driver 555.42.06 supports CUDA <= 12.5.
# We pin to 12.4 toolkit because PyTorch's cu124 wheels are built against it;
# matching them avoids version-skew warnings when compiling CUDAExtensions.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Architectures to compile kernels for. Override with --build-arg if you know
# the exact GPU (single arch builds much faster). 7.5=T4/RTX20, 8.0=A100,
# 8.6=A10/A40/RTX30, 8.9=L4/L40/RTX40, 9.0=H100.
ARG TORCH_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG USERNAME=ss3966
ARG USER_UID=1459679
ARG USER_GID=${USER_UID}

RUN echo 'alias c=clear' >> /etc/bash.bashrc

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_ARCH_LIST} \
    MUJOCO_GL=egl \
    MUJOCO_MENAGERIE_PATH=/opt/mujoco_menagerie

# OS packages: Python toolchain, build tools, GL/EGL for headless MuJoCo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        git wget curl ca-certificates \
        build-essential cmake ninja-build pkg-config \
        libgl1 libegl1 libgles2 libgomp1 libosmesa6 \
        vim less tmux htop \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — sidesteps PEP 668 and keeps system Python clean.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.4 wheels — matches base image).
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Scientific + dev tooling.
RUN pip install \
        numpy \
        matplotlib \
        pytest \
        pybind11 \
        mujoco \
        tqdm \
        ipython

# Robot model zoo (Franka Panda, Unitree H1/G1, etc. — used by Day-2+ demos).
RUN git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git \
        ${MUJOCO_MENAGERIE_PATH}

# Build-time sanity check. Fails the image build if anything is broken.
RUN python -c "import torch; \
               print('torch', torch.__version__, '/ built for cuda', torch.version.cuda); \
               import mujoco; print('mujoco', mujoco.__version__)"

# Create user, then give it ownership of the venv so `pip install -e .`
# inside the container can write to it without sudo.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -ms /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
 && chown -R ${USER_UID}:${USER_GID} /opt/venv

USER ${USERNAME}
WORKDIR /home/${USERNAME}/mppi-cuda

# CMD ["/bin/bash"]
CMD pip install -e ".[dev]" --no-deps --force-reinstall && exec /bin/bash
