#!/bin/bash
# =============================================
# Ascend 910C NPU Environment Setup Script
# =============================================
# Usage: source setup_npu_env.sh
#
# Prerequisites:
#   - CANN toolkit installed (e.g. /usr/local/Ascend/ascend-toolkit/latest)
#   - conda available

set -e

echo "=== Setting up Ascend NPU environment ==="

# ---- HuggingFace Mirror (for China network) ----
# Uncomment / modify if direct huggingface.co access is blocked.
# Option 1: hf-mirror.com (most popular China mirror)
export HF_ENDPOINT="https://hf-mirror.com"
# Option 2: custom mirror
# export HF_ENDPOINT="https://your-company-mirror.example.com"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[OK] HF_ENDPOINT=${HF_ENDPOINT}"

# ---- Source CANN environment ----
CANN_TOOLKIT=${CANN_TOOLKIT:-/usr/local/Ascend/ascend-toolkit/latest}
if [ -f "${CANN_TOOLKIT}/bin/setenv.bash" ]; then
    source "${CANN_TOOLKIT}/bin/setenv.bash"
    echo "[OK] CANN toolkit sourced from ${CANN_TOOLKIT}"
elif [ -f "${CANN_TOOLKIT}/set_env.sh" ]; then
    source "${CANN_TOOLKIT}/set_env.sh"
    echo "[OK] CANN toolkit sourced from ${CANN_TOOLKIT}"
else
    echo "[WARN] CANN toolkit not found at ${CANN_TOOLKIT}, skipping."
    echo "       Set CANN_TOOLKIT env var to your CANN installation path."
fi

# ---- Create isolated conda environment ----
ENV_NAME="parameter_golf_npu"
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists, activating..."
else
    echo "[INFO] Creating conda env '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi
conda activate ${ENV_NAME}

# ---- Install dependencies ----
echo "[INFO] Installing Python dependencies..."
pip install -r requirements_npu.txt

echo "=== Environment setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Download data:  python3 data/cached_challenge_fineweb.py --variant sp1024"
echo "  2. Single-card run: bash run_npu.sh 1"
echo "  3. Multi-card run:  bash run_npu.sh 8"
echo ""
echo "If HuggingFace download fails, check HF_ENDPOINT setting."
echo "Current HF_ENDPOINT=${HF_ENDPOINT}"
