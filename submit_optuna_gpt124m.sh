#!/bin/bash
#SBATCH --job-name=nanogpt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4 
#SBATCH -p a100-4,a100-8,apollo_agate
#SBATCH --output=exp_log/small_%A_%a.out
#SBATCH --error=exp_log/small_%A_%a.err

set -euo pipefail

cd /users/9/chen8596/nanoGPT
mkdir -p exp_log

python run_optuna_experiment.py config/experiments/optuna_train_gpt124m.yaml
