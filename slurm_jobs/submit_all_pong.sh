#!/bin/bash
# Submit all Pong training jobs (DQN, A2C, PPO)

echo "Submitting 3 Pong training jobs (DQN, A2C, PPO)..."

sbatch slurm_jobs/train_dqn_pong.sh
sbatch slurm_jobs/train_a2c_pong.sh
sbatch slurm_jobs/train_ppo_pong.sh

echo "All 3 Pong jobs submitted! Check status with: squeue -u \$USER"

