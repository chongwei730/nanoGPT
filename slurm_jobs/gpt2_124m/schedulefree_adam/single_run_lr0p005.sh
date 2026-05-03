#!/bin/bash
#SBATCH --job-name=gpt124_sfa_lr5e3
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ssafo
#SBATCH --gres=gpu:4
#SBATCH -p saffo-a100
#SBATCH --chdir=/scratch.global/zhan9381/nanoGPT
#SBATCH --output=/scratch.global/zhan9381/nanoGPT/exp_log/gpt2_124m_schedulefree_adam_single_run_lr0p005_%A_%a.out
#SBATCH --error=/scratch.global/zhan9381/nanoGPT/exp_log/gpt2_124m_schedulefree_adam_single_run_lr0p005_%A_%a.err

set -euo pipefail

REPO_ROOT="/scratch.global/zhan9381/nanoGPT"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"

mkdir -p "${REPO_ROOT}/exp_log/slurm"
cd "${REPO_ROOT}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Missing venv activate script at ${VENV_ACTIVATE}" >&2
  exit 127
fi
source "${VENV_ACTIVATE}"
echo "Python: $(command -v python)"
python --version

OUT_DIR="${REPO_ROOT}/out_schedulefree_gpt2_lr0p005"
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"

torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py \
  --optimizer_type=AdamWScheduleFree \
  --learning_rate=0.005 \
  --decay_lr=False \
  --compile=False \
  --out_dir="$OUT_DIR"
