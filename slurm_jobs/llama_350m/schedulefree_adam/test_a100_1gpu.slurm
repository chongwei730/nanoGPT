#!/bin/bash
#SBATCH --job-name=test_a100_1gpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=ssafo
#SBATCH --partition=saffo-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./test_a100_1gpu_%j.out
#SBATCH --error=./test_a100_1gpu_%j.err

hostname
nvidia-smi
