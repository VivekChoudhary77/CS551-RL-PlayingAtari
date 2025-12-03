#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem 32G
#SBATCH --job-name="A2C_Pong"
#SBATCH -o logs/slurm_a2c_pong_%j.out
#SBATCH -e logs/slurm_a2c_pong_%j.err

# Load CUDA modules
module load cuda

# Activate virtual environment
source ~/miniconda3/envs/myenv2/bin/activate

# Navigate to project directory
cd ~/CS551-RL-PlayingAtari

# Run training (5M steps consistent across all algorithms)
python scripts/train.py --algo a2c --game pong --steps 5000000 --seed 0

echo "A2C Pong training completed!"

