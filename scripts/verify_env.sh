#!/usr/bin/env bash
# Verify the container environment is healthy.
# Run inside the container:  bash scripts/verify_env.sh
set -u

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
yel()   { printf "\033[33m%s\033[0m\n" "$1"; }

check() {
    local name="$1"; shift
    if "$@" > /tmp/check.out 2>&1; then
        green "OK    $name"
        sed 's/^/      /' /tmp/check.out | head -3
    else
        red "FAIL  $name"
        sed 's/^/      /' /tmp/check.out | head -10
    fi
    echo
}

echo "===================="
echo "Environment check"
echo "===================="
echo

check "python"      python --version
check "pip"         pip --version
check "nvidia-smi"  nvidia-smi -L
check "nvcc"        nvcc --version

check "torch import"        python -c "import torch; print('version:', torch.__version__)"
check "torch.cuda.is_available" \
    python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0))"

check "mujoco import"       python -c "import mujoco; print('version:', mujoco.__version__)"

if [ -d "${MUJOCO_MENAGERIE_PATH:-}" ]; then
    green "OK    MuJoCo Menagerie present"
    echo "      ${MUJOCO_MENAGERIE_PATH}"
    ls "${MUJOCO_MENAGERIE_PATH}" 2>/dev/null | head -5 | sed 's/^/      /'
else
    yel  "WARN  MUJOCO_MENAGERIE_PATH not set or missing"
fi
echo

# Quick sanity test of the package.
if [ -f "./mppi_cuda/__init__.py" ]; then
    check "mppi_cuda import" \
        bash -c "cd . && PYTHONPATH=. python -c 'import mppi_cuda; print(mppi_cuda.__version__)'"
else
    yel  "WARN  . doesn't contain mppi_cuda — mount your repo with -v"
fi
