#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 48:00:00
#SBATCH --mem 32G
#SBATCH --job-name="QRDQN_BeamRider"
#SBATCH -o logs/slurm_qrdqn_beamrider_%j.out
#SBATCH -e logs/slurm_qrdqn_beamrider_%j.err

# Load CUDA modules
module load cuda

# Activate virtual environment
source ~/miniconda3/envs/myenv2/bin/activate

# Navigate to project directory
cd ~/CS551-RL-PlayingAtari

# Run training (5M steps - Quantile Regression DQN)
python scripts/train.py --algo qrdqn --game beamrider --steps 5000000 --seed 0

echo "QRDQN BeamRider training completed!"

