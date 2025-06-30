#!/usr/bin/env python3
"""Compare results from baseline and PDA experiments"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(log_dir):
    """Load metrics from experiment directory."""
    metrics_file = os.path.join(log_dir, 'metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Try to load from checkpoint
    checkpoint_files = list(Path(log_dir).glob('checkpoint_*.pt'))
    if checkpoint_files:
        import torch
        # Load the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        return {
            'final_accuracy': checkpoint.get('best_val_acc', 0),
            'epoch': checkpoint.get('epoch', 0)
        }
    
    return None


def create_comparison_plot(results, output_path='comparison_results.png'):
    """Create a comparison plot of results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Baseline vs PDA Comparison', fontsize=16)
    
    # Clean accuracy comparison
    ax1 = axes[0, 0]
    methods = list(results.keys())
    accuracies = [results[m].get('final_accuracy', 0) for m in methods]
    bars1 = ax1.bar(methods, accuracies, color=['blue', 'green'])
    ax1.set_ylabel('Clean Accuracy (%)')
    ax1.set_title('Clean Test Accuracy')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Training time comparison (if available)
    ax2 = axes[0, 1]
    if all('training_time' in results[m] for m in methods):
        times = [results[m]['training_time'] for m in methods]
        bars2 = ax2.bar(methods, times, color=['blue', 'green'])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Training Time')
        
        for bar, time in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.0f}s', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'Training time data not available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # Robustness comparison (if available)
    ax3 = axes[1, 0]
    if all('mce' in results[m] for m in methods):
        mces = [results[m]['mce'] for m in methods]
        bars3 = ax3.bar(methods, mces, color=['blue', 'green'])
        ax3.set_ylabel('Mean Corruption Error')
        ax3.set_title('Corruption Robustness (lower is better)')
        
        for bar, mce in zip(bars3, mces):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mce:.1f}', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'Robustness data not available\nRun evaluation script for full results',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    
    for method in methods:
        summary_text += f"{method}:\n"
        summary_text += f"  • Clean Accuracy: {results[method].get('final_accuracy', 'N/A'):.1f}%\n"
        
        if 'mce' in results[method]:
            summary_text += f"  • mCE: {results[method]['mce']:.1f}\n"
        
        if 'training_time' in results[method]:
            summary_text += f"  • Training Time: {results[method]['training_time']:.0f}s\n"
        
        summary_text += "\n"
    
    # Calculate improvements
    if len(methods) == 2 and all('final_accuracy' in results[m] for m in methods):
        baseline_acc = results[methods[0]]['final_accuracy']
        pda_acc = results[methods[1]]['final_accuracy']
        improvement = pda_acc - baseline_acc
        summary_text += f"Accuracy Improvement: {improvement:+.1f}%"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--exp1', type=str, required=True, help='Path to first experiment logs')
    parser.add_argument('--exp2', type=str, required=True, help='Path to second experiment logs')
    parser.add_argument('--output', type=str, default='comparison_results.png', 
                       help='Output path for comparison plot')
    args = parser.parse_args()
    
    # Load results
    results = {}
    
    for exp_path, exp_name in [(args.exp1, 'Baseline'), (args.exp2, 'PDA')]:
        print(f"Loading results from {exp_path}...")
        metrics = load_metrics(exp_path)
        
        if metrics:
            results[exp_name] = metrics
            print(f"  Found metrics for {exp_name}")
        else:
            print(f"  Warning: No metrics found for {exp_name}")
            results[exp_name] = {'final_accuracy': 0}
    
    # Create comparison plot
    create_comparison_plot(results, args.output)
    
    # Print detailed comparison
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60)
    
    for method, metrics in results.items():
        print(f"\n{method}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()