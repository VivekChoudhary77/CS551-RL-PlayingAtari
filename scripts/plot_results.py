"""
Visualization script to plot learning curves from TensorBoard logs
"""

import os
import re
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
    
    # Color palette for 5 algorithms: DQN, QRDQN, A2C, PPO, RecurrentPPO
    # Blue, Red, Orange, Green, Purple
    color_map = {
        'DQN': '#1f77b4',      # Blue
        'QRDQN': '#d62728',     # Red
        'A2C': '#ff7f0e',       # Orange
        'PPO': '#2ca02c',       # Green
        'RecurrentPPO': '#9467bd'  # Purple
    } 
    
    for log_dir, label in zip(log_dirs, labels):
        steps, rewards = load_tensorboard_data(log_dir)
        if steps:
            # Get color from map, default to cycle if not found
            color = color_map.get(label, None)
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


def parse_evaluation_results(results_dir='results'):
    """
    Parse evaluation results from .txt files in results directory.
    Returns dict: {algorithm: {game: (mean, std)}}
    """
    results = {}
    games = ['pong', 'beamrider']
    algorithms = ['dqn', 'qrdqn', 'a2c', 'ppo', 'recurrentppo']
    
    for algo in algorithms:
        results[algo.upper()] = {}
        for game in games:
            eval_file = os.path.join(results_dir, f'{algo}_{game}_eval.txt')
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'r') as f:
                        content = f.read()
                        # Extract mean and std from "Mean reward: X.XX +/- Y.YY"
                        match = re.search(r'Mean reward:\s*([\d.]+)\s*\+/-\s*([\d.]+)', content)
                        if match:
                            mean = float(match.group(1))
                            std = float(match.group(2))
                            results[algo.upper()][game] = (mean, std)
                except Exception as e:
                    print(f"Error parsing {eval_file}: {e}")
    
    return results


def plot_final_performance_bar(results_dict, game, output_file):
    """
    Create bar plot comparing final performance for a specific game.
    """
    algorithms = []
    means = []
    stds = []
    
    color_map = {
        'DQN': '#1f77b4',
        'QRDQN': '#d62728',
        'A2C': '#ff7f0e',
        'PPO': '#2ca02c',
        'RECURRENTPPO': '#9467bd'
    }
    
    for algo in ['DQN', 'QRDQN', 'A2C', 'PPO', 'RECURRENTPPO']:
        if algo in results_dict and game in results_dict[algo]:
            algorithms.append(algo)
            mean, std = results_dict[algo][game]
            means.append(mean)
            stds.append(std)
    
    if not algorithms:
        print(f"No data found for {game}")
        return
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    x = np.arange(len(algorithms))
    colors = [color_map.get(algo, '#808080') for algo in algorithms]
    bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + height*0.01,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    plt.title(f'Final Performance Comparison - {game.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(x, algorithms, rotation=0, ha='center', fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_file}")
    plt.close()


def plot_performance_heatmap(results_dict, output_file='results/performance_heatmap.png'):
    """
    Create heatmap showing performance across algorithms and games.
    """
    algorithms = ['DQN', 'QRDQN', 'A2C', 'PPO', 'RECURRENTPPO']
    games = ['Pong', 'BeamRider']
    
    # Create matrix
    data_matrix = []
    for algo in algorithms:
        row = []
        for game in games:
            game_lower = game.lower()
            if algo in results_dict and game_lower in results_dict[algo]:
                mean, _ = results_dict[algo][game_lower]
                row.append(mean)
            else:
                row.append(0)
        data_matrix.append(row)
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Normalize for better visualization (since scales are very different)
    data_array = np.array(data_matrix)
    # Use log scale for better visualization of large differences
    data_normalized = np.log1p(data_array)  # log(1+x) to handle zeros
    
    sns.heatmap(data_normalized, annot=data_array, fmt='.1f', cmap='YlOrRd', 
                xticklabels=games, yticklabels=algorithms,
                cbar_kws={'label': 'Mean Reward (log scale)'}, 
                linewidths=0.5, linecolor='gray', square=False)
    
    plt.title('Algorithm Performance Heatmap Across Games', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Game', fontsize=12, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    plt.close()


def plot_algorithm_type_comparison(results_dict, output_file='results/algorithm_type_comparison.png'):
    """
    Compare Value-based vs Policy-based algorithms.
    """
    value_based = {'DQN': [], 'QRDQN': []}
    policy_based = {'A2C': [], 'PPO': [], 'RECURRENTPPO': []}
    games = ['Pong', 'BeamRider']
    
    for game in games:
        game_lower = game.lower()
        for algo in value_based.keys():
            if algo in results_dict and game_lower in results_dict[algo]:
                mean, _ = results_dict[algo][game_lower]
                value_based[algo].append(mean)
            else:
                value_based[algo].append(0)
        
        for algo in policy_based.keys():
            if algo in results_dict and game_lower in results_dict[algo]:
                mean, _ = results_dict[algo][game_lower]
                policy_based[algo].append(mean)
            else:
                policy_based[algo].append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")
    
    x = np.arange(len(games))
    width = 0.15
    
    # Value-based algorithms
    for i, (algo, values) in enumerate(value_based.items()):
        offset = (i - len(value_based)/2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=algo, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Game', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Value-Based Algorithms', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(games)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Policy-based algorithms
    for i, (algo, values) in enumerate(policy_based.items()):
        offset = (i - len(policy_based)/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=algo, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Game', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Policy-Based Algorithms', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(games)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Algorithm Type Comparison: Value-Based vs Policy-Based', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Algorithm type comparison saved to: {output_file}")
    plt.close()


def plot_sample_efficiency(log_dirs, labels, game, threshold_ratios, output_file):
    """
    Plot sample efficiency: steps needed to reach X% of final performance.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    color_map = {
        'DQN': '#1f77b4',
        'QRDQN': '#d62728',
        'A2C': '#ff7f0e',
        'PPO': '#2ca02c',
        'RecurrentPPO': '#9467bd'
    }
    
    efficiency_data = {label: [] for label in labels}
    
    for log_dir, label in zip(log_dirs, labels):
        steps, rewards = load_tensorboard_data(log_dir)
        if steps and rewards:
            final_reward = max(rewards[-100:])  # Use max of last 100 points as final
            for threshold_ratio in threshold_ratios:
                target = final_reward * threshold_ratio
                # Find first step where reward >= target
                reached_step = None
                for i, (s, r) in enumerate(zip(steps, rewards)):
                    if r >= target:
                        reached_step = s
                        break
                efficiency_data[label].append(reached_step if reached_step else steps[-1])
    
    x = np.arange(len(threshold_ratios))
    width = 0.15
    
    for i, label in enumerate(labels):
        if efficiency_data[label]:
            offset = (i - len(labels)/2 + 0.5) * width
            color = color_map.get(label, None)
            plt.bar(x + offset, efficiency_data[label], width, label=label, 
                   alpha=0.8, color=color, edgecolor='black', linewidth=1)
    
    plt.xlabel('Performance Threshold (% of Final)', fontsize=12, fontweight='bold')
    plt.ylabel('Training Steps', fontsize=12, fontweight='bold')
    plt.title(f'Sample Efficiency - {game.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{int(t*100)}%' for t in threshold_ratios])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sample efficiency plot saved to: {output_file}")
    plt.close()


def generate_all_plots(results_dir='results', logs_base='logs'):
    """
    Generate all comparison plots for the project.
    """
    print("="*60)
    print("Generating All Comparison Plots")
    print("="*60)
    
    # Parse evaluation results
    print("\n1. Parsing evaluation results...")
    results_dict = parse_evaluation_results(results_dir)
    
    # 1. Final Performance Bar Charts (per game)
    print("\n2. Generating final performance bar charts...")
    plot_final_performance_bar(results_dict, 'pong', 
                              os.path.join(results_dir, 'final_performance_pong.png'))
    plot_final_performance_bar(results_dict, 'beamrider', 
                              os.path.join(results_dir, 'final_performance_beamrider.png'))
    
    # 2. Performance Heatmap
    print("\n3. Generating performance heatmap...")
    plot_performance_heatmap(results_dict, 
                            os.path.join(results_dir, 'performance_heatmap.png'))
    
    # 3. Algorithm Type Comparison
    print("\n4. Generating algorithm type comparison...")
    plot_algorithm_type_comparison(results_dict, 
                                  os.path.join(results_dir, 'algorithm_type_comparison.png'))
    
    # 4. Sample Efficiency Plots
    print("\n5. Generating sample efficiency plots...")
    log_dirs_pong = [
        os.path.join(logs_base, 'dqn_pong_seed0'),
        os.path.join(logs_base, 'qrdqn_pong_seed0'),
        os.path.join(logs_base, 'a2c_pong_seed0'),
        os.path.join(logs_base, 'ppo_pong_seed0'),
        os.path.join(logs_base, 'recurrentppo_pong_seed0')
    ]
    labels = ['DQN', 'QRDQN', 'A2C', 'PPO', 'RecurrentPPO']
    plot_sample_efficiency(log_dirs_pong, labels, 'Pong', [0.5, 0.75, 0.9, 0.95],
                          os.path.join(results_dir, 'sample_efficiency_pong.png'))
    
    log_dirs_beamrider = [
        os.path.join(logs_base, 'dqn_beamrider_seed0'),
        os.path.join(logs_base, 'qrdqn_beamrider_seed0'),
        os.path.join(logs_base, 'a2c_beamrider_seed0'),
        os.path.join(logs_base, 'ppo_beamrider_seed0'),
        os.path.join(logs_base, 'recurrentppo_beamrider_seed0')
    ]
    plot_sample_efficiency(log_dirs_beamrider, labels, 'BeamRider', [0.5, 0.75, 0.9, 0.95],
                          os.path.join(results_dir, 'sample_efficiency_beamrider.png'))
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves and comparison plots')
    parser.add_argument('--mode', type=str, default='learning_curves',
                       choices=['learning_curves', 'all'],
                       help='Plot mode: learning_curves or all')
    parser.add_argument('--log-dirs', nargs='+', default=None,
                       help='List of TensorBoard log directories (for learning_curves mode)')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Labels for each run (for learning_curves mode)')
    parser.add_argument('--game', type=str, default=None,
                       help='Game name for plot title (for learning_curves mode)')
    parser.add_argument('--output', type=str, default='results/learning_curves.png',
                       help='Output file path (for learning_curves mode)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory (for all mode)')
    parser.add_argument('--logs-base', type=str, default='logs',
                       help='Base directory for logs (for all mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        generate_all_plots(args.results_dir, args.logs_base)
    else:
        if args.log_dirs is None or args.labels is None or args.game is None:
            raise ValueError("--log-dirs, --labels, and --game are required for learning_curves mode")
        if len(args.log_dirs) != len(args.labels):
            raise ValueError("Number of log directories must match number of labels")
        plot_learning_curves(args.log_dirs, args.labels, args.game, args.output)

