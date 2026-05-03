#!/bin/bash
#SBATCH --job-name=n12_gpt124_sfa
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ssafo
#SBATCH --gres=gpu:4
#SBATCH -p saffo-a100
#SBATCH --chdir=/scratch.global/zhan9381/nanoGPT
#SBATCH --output=/scratch.global/zhan9381/nanoGPT/exp_log/gpt2_124m_schedulefree_adam_num_trials_12_%A_%a.out
#SBATCH --error=/scratch.global/zhan9381/nanoGPT/exp_log/gpt2_124m_schedulefree_adam_num_trials_12_%A_%a.err

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

MAX_ITERS="${MAX_ITERS:-5200}"
RUN_ROOT="/scratch.global/chen8596/experiment_runs_modified/gpt124m_schedulefree_adam_lr_search_serial_halving_num_trials_12_maxiters_${MAX_ITERS}"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_schedulefree_adam_gpt124m.yaml \
  --num-trials 12 \
  --num-iterations-per-trial "$MAX_ITERS" \
  --run-root "$RUN_ROOT"
