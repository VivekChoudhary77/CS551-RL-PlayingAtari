"""
Evaluation script for trained RL agents
Records gameplay videos and computes performance metrics
"""

import os
import argparse
import numpy as np
import ale_py  # Required for ALE namespace
from stable_baselines3 import DQN, A2C, PPO
from sb3_contrib import QRDQN, RecurrentPPO  # Additional algorithms from SB3-Contrib
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


def make_atari_env(env_id, seed=0):
    """Create Atari environment with standard wrappers."""
    def _init():
        env = gym.make(env_id, render_mode='rgb_array')
        env = Monitor(env)
        env = AtariWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate_agent(model_path, game, n_eval_episodes=10, record_video=True, video_folder='videos'):
    """
    Evaluate a trained agent and optionally record videos.
    
    Args:
        model_path: Path to saved model (.zip file)
        game: Game name ('pong', 'beamrider')
        n_eval_episodes: Number of episodes to evaluate
        record_video: Whether to record gameplay video
        video_folder: Folder to save videos
    
    Returns:
        mean_reward, std_reward
    """
    
    # Map game names to Gymnasium IDs
    game_mapping = {
        'pong': 'PongNoFrameskip-v4',
        'beamrider': 'BeamRiderNoFrameskip-v4'
    }
    
    env_id = game_mapping[game.lower()]
    
    # Determine algorithm from model path
    # Check for qrdqn BEFORE dqn (since 'dqn' is substring of 'qrdqn')
    # Check for recurrentppo/rppo BEFORE ppo (since 'ppo' is substring)
    if 'qrdqn' in model_path.lower():
        Model = QRDQN
    elif 'recurrentppo' in model_path.lower() or 'rppo' in model_path.lower():
        Model = RecurrentPPO
    elif 'dqn' in model_path.lower():
        Model = DQN
    elif 'a2c' in model_path.lower():
        Model = A2C
    elif 'ppo' in model_path.lower():
        Model = PPO
    else:
        raise ValueError("Cannot determine algorithm from model path")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = Model.load(model_path)
    
    # Create environment
    env = DummyVecEnv([make_atari_env(env_id)])
    env = VecFrameStack(env, n_stack=4)
    
    # Optionally wrap with video recorder
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        video_length = 1000  # Record 1000 steps per video
        env = VecVideoRecorder(
            env, 
            video_folder,
            record_video_trigger=lambda x: x % video_length == 0,
            video_length=video_length,
            name_prefix=f"eval_{os.path.basename(model_path).replace('.zip', '')}"
        )
    
    # Evaluate
    print(f"\nEvaluating on {game.upper()} for {n_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"\n{'='*60}")
    print(f"Results for {os.path.basename(model_path)}")
    print(f"Game: {game.upper()}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"{'='*60}\n")
    
    env.close()
    
    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'beamrider'],
                       help='Atari game')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--record', action='store_true',
                       help='Record gameplay videos')
    parser.add_argument('--video-folder', type=str, default='videos',
                       help='Folder to save videos')
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model,
        game=args.game,
        n_eval_episodes=args.episodes,
        record_video=args.record,
        video_folder=args.video_folder
    )

