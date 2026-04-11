#!/bin/bash
#SBATCH --job-name=ls_gpt124m
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/line_search_%A_%a.out
#SBATCH --error=exp_log/line_search_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm

# Adjust these if your environment needs module/conda activation
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nanogpt
cd /users/9/chen8596/nanoGPT

RUN_ROOT="experiment_runs/gpt124m_line_search_stage2"
FINAL_DIR="$RUN_ROOT/final"
mkdir -p "$FINAL_DIR"

torchrun --standalone --nproc_per_node=4 train_linesearch.py 