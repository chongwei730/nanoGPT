#!/bin/bash
#SBATCH --job-name=48h_s2_gpt355_cos
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/gpt355m_cosine_s2_%A_%a.out
#SBATCH --error=exp_log/gpt355m_cosine_s2_%A_%a.err

set -euo pipefail
mkdir -p exp_log/slurm
cd ../..

python run_stage2_final.py \
  experiment_runs/gpt355m_lr_search_staged_48h/level_48h/stage1/stage1_result.json
