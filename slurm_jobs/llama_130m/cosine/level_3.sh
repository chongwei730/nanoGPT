#!/bin/bash
#SBATCH --job-name=l3_llama130m
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ssafo
#SBATCH --gres=gpu:4
#SBATCH -p saffo-a100
#SBATCH --chdir=/scratch.global/zhan9381/nanoGPT
#SBATCH --output=/scratch.global/zhan9381/nanoGPT/exp_log/llama_130m_cosine_level_3_%A_%a.out
#SBATCH --error=/scratch.global/zhan9381/nanoGPT/exp_log/llama_130m_cosine_level_3_%A_%a.err

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

MAX_ITERS="${MAX_ITERS:-10000}"
RUN_ROOT="/scratch.global/chen8596/experiment_runs_last/llama130m_lr_search_tuning_batch_level_3_maxiters_${MAX_ITERS}"
echo "Launching or resuming tuning-batch level 3 run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_train_llama130m.yaml \
  --num-trials 12 \
  --num-iterations-per-trial "$MAX_ITERS" \
  --run-root "$RUN_ROOT" \
  --batch-index 3
