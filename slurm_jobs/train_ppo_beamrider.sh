#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 48:00:00
#SBATCH --mem 32G
#SBATCH --job-name="PPO_BeamRider"
#SBATCH -o logs/slurm_ppo_beamrider_%j.out
#SBATCH -e logs/slurm_ppo_beamrider_%j.err

# Load CUDA modules
module load cuda

# Activate virtual environment
source ~/miniconda3/envs/myenv2/bin/activate

# Navigate to project directory
cd ~/CS551-RL-PlayingAtari

# Run training
python scripts/train.py --algo ppo --game beamrider --steps 5000000 --seed 0

echo "PPO BeamRider training completed!"

