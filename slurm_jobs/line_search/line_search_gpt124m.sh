#!/bin/bash
#SBATCH --job-name=ls_gpt124m
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4 
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/line_search_%A_%a.out
#SBATCH --error=exp_log/line_search_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm

# Adjust these if your environment needs module/conda activation
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nanogpt
cd ..
cd ..

RUN_ROOT="/work/nvme/bgop/cchen47/gpt124m_line_search_stage2"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --nproc-per-node 4 \
  --experiment-name "gpt124m_line_search" \
  --trial-id "stage2_final"
