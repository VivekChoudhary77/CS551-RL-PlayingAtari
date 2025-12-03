#!/usr/bin/env bash
# Submit all BeamRider training jobs

echo "Submitting BeamRider training jobs..."

# Submit DQN
job_dqn=$(sbatch slurm_jobs/train_dqn_beamrider.sh | awk '{print $4}')
echo "Submitted DQN BeamRider: Job ID $job_dqn"

# Submit A2C
job_a2c=$(sbatch slurm_jobs/train_a2c_beamrider.sh | awk '{print $4}')
echo "Submitted A2C BeamRider: Job ID $job_a2c"

# Submit PPO
job_ppo=$(sbatch slurm_jobs/train_ppo_beamrider.sh | awk '{print $4}')
echo "Submitted PPO BeamRider: Job ID $job_ppo"

echo "All BeamRider jobs submitted!"
squeue -u $USER

