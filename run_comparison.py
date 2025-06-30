#!/usr/bin/env python3
"""Script to run baseline vs PDA comparison."""

import subprocess
import argparse
import os
import yaml
import json
from datetime import datetime


def run_experiment(config_override, experiment_name):
    """Run a single experiment with given config overrides."""
    cmd = [
        "python", "train.py",
        "--config", "config.yml"
    ]
    
    for key, value in config_override.items():
        cmd.extend(["--override", f"{key}={value}"])
    
    cmd.extend(["--override", f"experiment.name={experiment_name}"])
    
    print(f"\nRunning experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd)
    return process


def main():
    parser = argparse.ArgumentParser(description='Run baseline vs PDA comparison')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet32'])
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'wide_resnet28_10', 'vit_small'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--quick', action='store_true', help='Run quick test with 10 epochs')
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        'data.dataset': args.dataset,
        'model.architecture': args.model,
        'training.epochs': args.epochs if not args.quick else 10
    }
    
    # Create timestamp for experiment group
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Baseline experiment
    baseline_config = base_config.copy()
    baseline_config['pda.enable'] = 'false'
    baseline_name = f"baseline_{args.dataset}_{args.model}_{timestamp}"
    
    # PDA experiment
    pda_config = base_config.copy()
    pda_config['pda.enable'] = 'true'
    pda_name = f"pda_{args.dataset}_{args.model}_{timestamp}"
    
    # Run experiments
    print("="*60)
    print("Running Baseline vs PDA Comparison")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {baseline_config['training.epochs']}")
    print("="*60)
    
    # Start both experiments
    baseline_proc = run_experiment(baseline_config, baseline_name)
    pda_proc = run_experiment(pda_config, pda_name)
    
    # Wait for both to complete
    print("\nWaiting for experiments to complete...")
    baseline_proc.wait()
    pda_proc.wait()
    
    print("\n" + "="*60)
    print("Experiments completed!")
    print("="*60)
    print(f"Baseline results: logs/{baseline_name}/")
    print(f"PDA results: logs/{pda_name}/")
    
    # Compare results if available
    try:
        baseline_log = f"logs/{baseline_name}/checkpoint.pth"
        pda_log = f"logs/{pda_name}/checkpoint.pth"
        
        if os.path.exists(baseline_log) and os.path.exists(pda_log):
            import torch
            baseline_ckpt = torch.load(baseline_log, map_location='cpu')
            pda_ckpt = torch.load(pda_log, map_location='cpu')
            
            print("\nResults Summary:")
            print(f"Baseline - Best Accuracy: {baseline_ckpt.get('best_acc', 'N/A'):.2f}%")
            print(f"PDA - Best Accuracy: {pda_ckpt.get('best_acc', 'N/A'):.2f}%")
            
            if 'best_acc' in baseline_ckpt and 'best_acc' in pda_ckpt:
                improvement = pda_ckpt['best_acc'] - baseline_ckpt['best_acc']
                print(f"Improvement: {improvement:+.2f}%")
    except Exception as e:
        print(f"Could not compare results: {e}")


if __name__ == '__main__':
    main()