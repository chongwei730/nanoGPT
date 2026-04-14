#!/bin/bash
#SBATCH --job-name=n32_gpt770_sfa
#SBATCH --time=196:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4 
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/gpt770m_sfa_n32_%A_%a.out
#SBATCH --error=exp_log/gpt770m_sfa_n32_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../..

RUN_ROOT="/work/nvme/bgop/cchen47/gpt770m_schedulefree_adam_lr_search_serial_halving_num_trials_32"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_schedulefree_adam_gpt770m.yaml \
  --num-trials 32 \
  --run-root "$RUN_ROOT"
