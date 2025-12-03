# Deep Reinforcement Learning for Atari Games: Pong and Beam Rider

This project implements and compares multiple Deep Reinforcement Learning algorithms on Atari games using Stable-Baselines3 and Gymnasium.

## Algorithms Implemented

**Value-Based:**
- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN

**Policy-Based:**
- A2C (Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)

## Games

- **Pong** (reflex-based, simpler dynamics)
- **BeamRider** (strategic shooting, complex dynamics)

## Project Structure

```
CS551-RL-PlayingAtari/
├── scripts/
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Model evaluation and video recording
│   ├── plot_results.py   # Visualization utilities
│   └── test_setup.py     # Setup verification
├── slurm_jobs/
│   ├── train_*.sh        # SLURM job scripts for Turing
│   └── submit_all_pong.sh
├── logs/                 # TensorBoard logs
├── models/               # Saved models
├── videos/               # Gameplay recordings
├── results/              # Plots and analysis
├── materials/            # Project documentation
│   └── SETUP_INSTRUCTIONS.md
└── requirements.txt
```

## Quick Start

### Local Testing

```bash
# Activate virtual environment
source /home/vickinez_077/WPI\ FALL\ 2025/RL/venvrl/bin/activate

# Test setup
python scripts/test_setup.py

# Quick training test (local)
python scripts/train.py --algo dqn --game pong --steps 10000 --seed 0
```

### Training on Turing

See `materials/SETUP_INSTRUCTIONS.md` for detailed setup steps.

Quick commands:
```bash
# Submit all Pong training jobs
bash slurm_jobs/submit_all_pong.sh

# Monitor jobs
squeue -u $USER

# Check logs
tail -f logs/slurm_dqn_pong_*.out
```

## Usage Examples

### Train a model
```bash
python scripts/train.py --algo ppo --game pong --steps 2000000 --seed 0
```

### Evaluate a trained model
```bash
python scripts/evaluate.py \
    --model models/ppo_pong_seed0_final.zip \
    --game pong \
    --episodes 10 \
    --record
```

### Generate comparison plots
```bash
python scripts/plot_results.py \
    --log-dirs logs/dqn_pong_seed0 logs/ppo_pong_seed0 \
    --labels DQN PPO \
    --game pong
```

## References

1. Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
2. Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature 518
3. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
4. Van Hasselt, H. et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI

## Links

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
- [SB3 RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
