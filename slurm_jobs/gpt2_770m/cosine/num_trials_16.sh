#!/bin/bash
#SBATCH --job-name=n16_gpt770_cos
#SBATCH --time=196:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/gpt770m_cosine_n16_%A_%a.out
#SBATCH --error=exp_log/gpt770m_cosine_n16_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../..

RUN_ROOT="experiment_runs/gpt770m_lr_search_serial_halving_num_trials_16"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_train_gpt770m.yaml \
  --num-trials 16 \
  --run-root "$RUN_ROOT"
