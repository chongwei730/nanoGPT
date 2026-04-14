#!/bin/bash
#SBATCH --job-name=n12_gpt124_cos
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4 
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/gpt124m_cosine_n12_%A_%a.out
#SBATCH --error=exp_log/gpt124m_cosine_n12_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../../..

RUN_ROOT="/work/nvme/bgop/cchen47/gpt124m_lr_search_serial_halving_num_trials_12"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_train_gpt124m.yaml \
  --num-trials 12 \
  --run-root "$RUN_ROOT"
