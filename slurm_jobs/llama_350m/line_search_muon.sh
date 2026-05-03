#!/bin/bash
#SBATCH --job-name=lsmu_llama350m
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ssafo
#SBATCH --gres=gpu:4
#SBATCH -p saffo-a100
#SBATCH --chdir=/scratch.global/zhan9381/nanoGPT
#SBATCH --output=/scratch.global/zhan9381/nanoGPT/exp_log/llama_350m_line_search_muon_%A_%a.out
#SBATCH --error=/scratch.global/zhan9381/nanoGPT/exp_log/llama_350m_line_search_muon_%A_%a.err

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

MAX_ITERS="${MAX_ITERS:-59600}"
RUN_ROOT="/scratch.global/chen8596/experiment_runs_modified/llama350m_line_search_muon_stage2_maxiters_${MAX_ITERS}"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch_llama_muon.py" \
  --config-path "config/train_llama_350m.py" \
  --nproc-per-node 4 \
  --experiment-name "llama350m_line_search_muon" \
  --trial-id "stage2_final" \
  -- config/train_llama_350m.py --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS"
