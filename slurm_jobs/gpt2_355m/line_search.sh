#!/bin/bash
#SBATCH --job-name=ls_gpt355m
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/gpt355m_linesearch_%A_%a.out
#SBATCH --error=exp_log/gpt355m_linesearch_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../..

RUN_ROOT="experiment_runs/gpt355m_line_search_stage2"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch.py" \
  --config-path "config/train_gpt2_355m_7b.py" \
  --nproc-per-node 4 \
  --experiment-name "gpt355m_line_search" \
  --trial-id "stage2_final" \
  --learning-rate 1e-6 \
  -- config/train_gpt2_355m_7b.py --learning_rate=1e-6
