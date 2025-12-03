"""
Main training script for RL Atari Project
Supports: DQN, A2C, PPO
Games: Pong, BeamRider
"""

import os
import argparse
import torch
import ale_py  # Required for ALE namespace
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym


def make_atari_env(env_id, seed=0):
    """Create Atari environment with standard wrappers."""
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        env = AtariWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def train_agent(algo, game, total_timesteps, seed=0, log_dir=None, model_dir=None):
    """
    Train an RL agent on an Atari game.
    
    Args:
        algo: Algorithm name ('dqn', 'a2c', 'ppo')
        game: Game name ('pong', 'beamrider')
        total_timesteps: Total training steps
        seed: Random seed for reproducibility
        log_dir: Directory for TensorBoard logs
        model_dir: Directory to save models
    """
    
    # Map game names to standard NoFrameskip versions (Standard Benchmark)
    game_mapping = {
        'pong': 'PongNoFrameskip-v4',
        'beamrider': 'BeamRiderNoFrameskip-v4'
    }
    
    env_id = game_mapping[game.lower()]
    
    # Create directories
    if log_dir is None:
        log_dir = f"logs/{algo}_{game}_seed{seed}"
    if model_dir is None:
        model_dir = "models"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Configure logger
    logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
    
    # Set device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment (different for DQN vs A2C/PPO)
    # DQN: Single env is fine (off-policy with replay buffer)
    # A2C: Needs 16 parallel envs (RL Zoo standard for Atari)
    # PPO: Needs 8 parallel envs (RL Zoo standard for Atari)
    if algo.lower() == 'dqn':
        n_envs = 1
        print(f"Using {n_envs} environment (DQN is off-policy, single env OK)")
    elif algo.lower() == 'a2c':
        n_envs = 16  # A2C needs MORE envs than PPO for variance reduction
        print(f"Using {n_envs} parallel environments (A2C needs many envs for stability)")
    else:  # PPO
        n_envs = 8
        print(f"Using {n_envs} parallel environments (PPO with clipping reduces variance)")
    
    env = DummyVecEnv([make_atari_env(env_id, seed + i) for i in range(n_envs)])
    # Add frame stacking (critical for temporal information!)
    env = VecFrameStack(env, n_stack=4)
    
    # Algorithm-specific configurations
    if algo.lower() == 'dqn':
        # DQN hyperparameters (FIXED for better Pong performance)
        model = DQN(
            'CnnPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=500_000,  # Reduced from 1M to fit in 32GB RAM
            learning_starts=10_000,  # FIXED: Start learning much earlier
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10_000,
            exploration_fraction=0.5,  # FIXED: Explore longer - 2.5M steps (50% of 5M)
            exploration_final_eps=0.01,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            device=device
        )
        
    elif algo.lower() == 'a2c':
        # A2C hyperparameters (RL Zoo standard for Atari with n_envs=16)
        # With 16 parallel envs: 16 envs × 5 steps = 80 samples per update
        model = A2C(
            'CnnPolicy',
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # FIXED: Restored to 0.01 (Zoo standard)
            vf_coef=0.25,
            max_grad_norm=0.5,
            normalize_advantage=False,  # FIXED: False is standard for A2C Atari
            use_rms_prop=True,
            rms_prop_eps=1e-5,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            device=device
        )
        
    elif algo.lower() == 'ppo':
        # PPO hyperparameters (RL Zoo standard for Atari with n_envs=8)
        # With 8 parallel envs: 8 envs × 128 steps = 1024 samples per rollout
        model = PPO(
            'CnnPolicy',
            env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,  # FIXED: Restored to 0.01 (Zoo standard)
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Set logger
    model.set_logger(logger)
    
    # Checkpoint callback - save every 250k steps (more frequent for better recovery)
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path=model_dir,
        name_prefix=f"{algo}_{game}_seed{seed}"
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training {algo.upper()} on {game.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Logs: {log_dir}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"{algo}_{game}_seed{seed}_final.zip")
    model.save(final_model_path)
    print(f"\n✓ Training complete! Model saved to: {final_model_path}")
    
    env.close()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agents on Atari games')
    parser.add_argument('--algo', type=str, required=True,
                       choices=['dqn', 'a2c', 'ppo'],
                       help='RL algorithm to use')
    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'beamrider'],
                       help='Atari game to play')
    parser.add_argument('--steps', type=int, default=5_000_000,
                       help='Total training timesteps (default: 5M)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Custom log directory')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    train_agent(
        algo=args.algo,
        game=args.game,
        total_timesteps=args.steps,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )

