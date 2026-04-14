#!/bin/bash
#SBATCH --job-name=ls_gpt770m
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4 
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/gpt770m_linesearch_%A_%a.out
#SBATCH --error=exp_log/gpt770m_linesearch_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../..

RUN_ROOT="/work/nvme/bgop/cchen47/gpt770m_line_search_stage2"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch.py" \
  --config-path "config/train_gpt2_770m_15b.py" \
  --nproc-per-node 4 \
  --experiment-name "gpt770m_line_search" \
  --trial-id "stage2_final" \
  -- config/train_gpt2_770m_15b.py
