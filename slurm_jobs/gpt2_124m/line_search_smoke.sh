#!/bin/bash
#SBATCH --job-name=ls_gpt124m_smk
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/gpt124m_linesearch_smoke_%A_%a.out
#SBATCH --error=exp_log/gpt124m_linesearch_smoke_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../..

SMOKE_EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-2}"
RUN_ROOT="/work/nvme/bgop/cchen47/experiment_runs/gpt124m_line_search_stage2_smoke"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch.py" \
  --config-path "config/train_gpt2.py" \
  --nproc-per-node 4 \
  --experiment-name "gpt124m_line_search_smoke" \
  --trial-id "stage2_final_smoke" \
  --max-running-time-hours 0.05 \
  --max-study-time-hours 0.05 \
  -- config/train_gpt2.py \
  --compile=False \
  --max_iters=20 \
  --warmup_iters=2 \
  --lr_decay_iters=20 \
  --eval_interval="${SMOKE_EVAL_INTERVAL}" \
  --eval_iters=4 \
  --log_interval=1
