#!/bin/bash
#SBATCH --job-name=sens_ls_adam_llama60m
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ssafo
#SBATCH --gres=gpu:4
#SBATCH -p saffo-a100
#SBATCH --chdir=/scratch.global/zhan9381/nanoGPT
#SBATCH --array=0-9
#SBATCH --output=/scratch.global/zhan9381/nanoGPT/exp_log/llama_60m_sensitivity_linesearch_adam_c1_%A_%a.out
#SBATCH --error=/scratch.global/zhan9381/nanoGPT/exp_log/llama_60m_sensitivity_linesearch_adam_c1_%A_%a.err

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
echo "Array task ${TASK_ID} -> sweep=linesearch_adam_c1 trial_index=${TASK_ID}"

python run_sensitivity.py \
  config/experiments/sensitivity_analysis.yaml \
  --only linesearch_adam_c1 \
  --trial-index "${TASK_ID}" \
  --resume
