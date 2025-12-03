"""
Visualization script to plot learning curves from TensorBoard logs
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir, tag='rollout/ep_rew_mean'):
    """
    Load data from the LATEST TensorBoard event file in the directory.
    We assume the latest file corresponds to the final, successful training run.
    """
    # Find all event files recursively
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'tfevents' in file:
                full_path = os.path.join(root, file)
                event_files.append(full_path)
    
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return [], []

    # Sort by modification time (newest last) to pick the latest run
    event_files.sort(key=os.path.getmtime)
    latest_file = event_files[-1]
    print(f"Loading data from latest file: {latest_file}")

    try:
        ea = event_accumulator.EventAccumulator(latest_file)
        ea.Reload()
        
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            return steps, values
        else:
            print(f"Warning: Tag '{tag}' not found in {latest_file}")
            return [], []
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")
        return [], []


def plot_learning_curves(log_dirs, labels, game, output_file='results/learning_curves.png'):
    """
    Plot learning curves from multiple runs.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Use the specific color palette requested (Blue, Orange, Green)
    # Corresponds to default matplotlib/seaborn cycle: 0=Blue, 1=Orange, 2=Green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        steps, rewards = load_tensorboard_data(log_dir)
        if steps:
            # Use the fixed colors to match the previous graph
            color = colors[i % len(colors)]
            plt.plot(steps, rewards, label=label, linewidth=2, alpha=0.9, color=color)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title(f'Learning Curves - {game.upper()}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def create_comparison_plot(results_dict, output_file='results/algorithm_comparison.png'):
    """
    Create bar plot comparing final performance of different algorithms.
    
    Args:
        results_dict: Dict with {algorithm: (mean_reward, std_reward)}
        output_file: Path to save plot
    """
    algorithms = list(results_dict.keys())
    means = [results_dict[algo][0] for algo in algorithms]
    stds = [results_dict[algo][1] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    x = np.arange(len(algorithms))
    bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=sns.color_palette("husl", len(algorithms)))
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Mean Episode Reward', fontsize=12)
    plt.title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, algorithms, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves from TensorBoard logs')
    parser.add_argument('--log-dirs', nargs='+', required=True,
                       help='List of TensorBoard log directories')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='Labels for each run')
    parser.add_argument('--game', type=str, required=True,
                       help='Game name for plot title')
    parser.add_argument('--output', type=str, default='results/learning_curves.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    if len(args.log_dirs) != len(args.labels):
        raise ValueError("Number of log directories must match number of labels")
    
    plot_learning_curves(args.log_dirs, args.labels, args.game, args.output)

