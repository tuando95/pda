#!/usr/bin/env python3
"""Script to run baseline vs PDA comparison experiments"""

import subprocess
import os
import sys
import yaml
from omegaconf import OmegaConf

def run_experiment(experiment_name, enable_pda, epochs=20):
    """Run a single experiment with specified configuration."""
    
    # Create a temporary config file with the desired settings
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['experiment']['name'] = experiment_name
    config['pda']['enable'] = enable_pda
    config['training']['epochs'] = epochs
    
    # Save to temporary config
    temp_config_path = f'config_{experiment_name}.yml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    cmd = [
        sys.executable, 'train.py',
        '--config', temp_config_path
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"PDA enabled: {enable_pda}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {experiment_name}: {e}")
        return False
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    return True


def main():
    """Run baseline and PDA experiments for comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline vs PDA')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--baseline-only', action='store_true', help='Run only baseline')
    parser.add_argument('--pda-only', action='store_true', help='Run only PDA')
    args = parser.parse_args()
    
    experiments = []
    
    if not args.pda_only:
        experiments.append(('baseline_no_pda', False))
    
    if not args.baseline_only:
        experiments.append(('pda_augmented', True))
    
    results = []
    
    for exp_name, enable_pda in experiments:
        success = run_experiment(exp_name, enable_pda, args.epochs)
        results.append((exp_name, success))
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for exp_name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{exp_name}: {status}")
    
    # Create comparison script
    if all(success for _, success in results) and len(results) == 2:
        print("\nTo compare results, run:")
        print("python compare_results.py --exp1 logs/baseline_no_pda --exp2 logs/pda_augmented")


if __name__ == '__main__':
    main()