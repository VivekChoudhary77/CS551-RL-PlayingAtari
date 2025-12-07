#!/bin/bash
# Submit all Pong training jobs (DQN, QRDQN, A2C, PPO, RecurrentPPO)

echo "Submitting 5 Pong training jobs (DQN, QRDQN, A2C, PPO, RecurrentPPO)..."

sbatch slurm_jobs/train_dqn_pong.sh
sbatch slurm_jobs/train_qrdqn_pong.sh
sbatch slurm_jobs/train_a2c_pong.sh
sbatch slurm_jobs/train_ppo_pong.sh
sbatch slurm_jobs/train_recurrentppo_pong.sh

echo "All 5 Pong jobs submitted! Check status with: squeue -u \$USER"

