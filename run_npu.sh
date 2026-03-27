#!/bin/bash
# =============================================
# Launch training on Ascend 910C NPU
# =============================================
# Usage:
#   Single card:  bash run_npu.sh 1
#   Multi card:   bash run_npu.sh 8

set -e

NPROC=${1:-1}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---- HuggingFace Mirror ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- CANN environment ----
CANN_TOOLKIT=${CANN_TOOLKIT:-/usr/local/Ascend/ascend-toolkit/latest}
if [ -f "${CANN_TOOLKIT}/bin/setenv.bash" ]; then
    source "${CANN_TOOLKIT}/bin/setenv.bash"
elif [ -f "${CANN_TOOLKIT}/set_env.sh" ]; then
    source "${CANN_TOOLKIT}/set_env.sh"
fi

# ---- ACL Precision Mode (required for 910B/C) ----
export ACL_PRECISION_MODE="${ACL_PRECISION_MODE:-allow_mix_precision}"

# ---- Check data exists ----
DATA_DIR="${SCRIPT_DIR}/data/datasets/fineweb10B_sp1024"
if [ ! -d "${DATA_DIR}" ]; then
    echo "[ERROR] Training data not found at ${DATA_DIR}"
    echo "        Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi

# ---- Check NPU availability ----
python3 -c "import torch; import torch_npu; assert torch.npu.is_available(), 'NPU not available'; print(f'NPU count: {torch.npu.device_count()}')"

echo "=== Launching training on ${NPROC} NPU(s) ==="

if [ "${NPROC}" -eq 1 ]; then
    python3 "${SCRIPT_DIR}/train_gpt_npu.py"
else
    torchrun \
        --nproc_per_node=${NPROC} \
        --standalone \
        "${SCRIPT_DIR}/train_gpt_npu.py"
fi
