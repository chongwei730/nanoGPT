#!/bin/bash
#SBATCH --job-name=6h_s1_gpt
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4 
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/small_%A_%a.out
#SBATCH --error=exp_log/small_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm

# Adjust these if your environment needs module/conda activation
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nanogpt
cd /users/9/chen8596/nanoGPT

python run_stage1_optuna.py config/experiments/optuna_train_gpt124m.yaml \
  --levels 6 \
  --run-root experiment_runs/gpt124m_lr_search_staged_6h
