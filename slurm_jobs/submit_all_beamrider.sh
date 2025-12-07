#!/usr/bin/env bash
# Submit all BeamRider training jobs (DQN, QRDQN, A2C, PPO, RecurrentPPO)

echo "Submitting 5 BeamRider training jobs..."

# Submit DQN
job_dqn=$(sbatch slurm_jobs/train_dqn_beamrider.sh | awk '{print $4}')
echo "Submitted DQN BeamRider: Job ID $job_dqn"

# Submit QRDQN
job_qrdqn=$(sbatch slurm_jobs/train_qrdqn_beamrider.sh | awk '{print $4}')
echo "Submitted QRDQN BeamRider: Job ID $job_qrdqn"

# Submit A2C
job_a2c=$(sbatch slurm_jobs/train_a2c_beamrider.sh | awk '{print $4}')
echo "Submitted A2C BeamRider: Job ID $job_a2c"

# Submit PPO
job_ppo=$(sbatch slurm_jobs/train_ppo_beamrider.sh | awk '{print $4}')
echo "Submitted PPO BeamRider: Job ID $job_ppo"

# Submit RecurrentPPO
job_rppo=$(sbatch slurm_jobs/train_recurrentppo_beamrider.sh | awk '{print $4}')
echo "Submitted RecurrentPPO BeamRider: Job ID $job_rppo"

echo "All 5 BeamRider jobs submitted!"
squeue -u $USER

