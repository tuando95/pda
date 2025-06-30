#!/usr/bin/env python3
"""Comprehensive evaluation script for trained models."""

import argparse
import os
import yaml
import torch
import json
from tqdm import tqdm

from src.data.datasets import get_data_loaders
from src.models.architectures import get_model
from src.evaluation.robustness import CorruptionBenchmark, AdversarialRobustness
from src.utils.efficiency import EfficiencyTracker, compare_efficiency
from src.utils.visualization import (
    plot_corruption_results,
    plot_efficiency_comparison,
    create_experiment_summary_plot
)
from src.augmentation.diffusion_utils import NoiseScheduler, DiffusionModelWrapper
from src.augmentation.pda import PDABatchTransform


def evaluate_model(args):
    """Run comprehensive evaluation on a trained model."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data...")
    _, test_loader, num_classes = get_data_loaders(config)
    
    print("Loading model...")
    config['model']['num_classes'] = num_classes
    model = get_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'config': config
    }
    
    print("\nEvaluating clean accuracy...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    clean_accuracy = 100. * correct / total
    results['clean_accuracy'] = clean_accuracy
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")
    
    if args.eval_robustness:
        print("\nEvaluating corruption robustness...")
        corruption_benchmark = CorruptionBenchmark()
        
        corruption_types = config['evaluation'].get('corruption_types', [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'motion_blur'
        ])
        
        corruption_results = corruption_benchmark.evaluate(
            model, test_loader, corruption_types, device,
            dataset_name=config['data'].get('dataset', 'cifar10')
        )
        
        mce = corruption_benchmark.calculate_mce(corruption_results, clean_accuracy)
        
        results['corruption_results'] = corruption_results
        results['mce'] = mce
        
        print(f"Mean Corruption Error (mCE): {mce:.2f}")
        
        plot_corruption_results(
            corruption_results,
            save_path=os.path.join(args.output_dir, 'corruption_results.png')
        )
    
    if args.eval_adversarial:
        print("\nEvaluating adversarial robustness...")
        adv_robustness = AdversarialRobustness(model, device)
        
        epsilons = [0.01, 0.03, 0.05]
        adv_results = {}
        
        for eps in epsilons:
            fgsm_acc = adv_robustness.evaluate_adversarial(
                test_loader, 'fgsm', epsilon=eps
            )
            pgd_acc = adv_robustness.evaluate_adversarial(
                test_loader, 'pgd', epsilon=eps, alpha=eps/3, num_iter=10
            )
            
            adv_results[f'fgsm_eps_{eps}'] = fgsm_acc
            adv_results[f'pgd_eps_{eps}'] = pgd_acc
            
            print(f"FGSM (ε={eps}): {fgsm_acc:.2f}%")
            print(f"PGD (ε={eps}): {pgd_acc:.2f}%")
        
        results['adversarial_results'] = adv_results
    
    if args.eval_efficiency:
        print("\nEvaluating computational efficiency...")
        
        train_loader, _, _ = get_data_loaders(config)
        
        pda_transform = None
        if config['pda']['enable']:
            diffusion_model = DiffusionModelWrapper(
                model_path=None,
                model_type=config['diffusion']['model_type']
            )
            
            noise_scheduler = NoiseScheduler(
                num_timesteps=config['diffusion']['num_timesteps'],
                beta_start=config['diffusion']['beta_start'],
                beta_end=config['diffusion']['beta_end'],
                schedule_type=config['diffusion']['noise_schedule']
            )
            
            pda_transform = PDABatchTransform(
                diffusion_model=diffusion_model,
                noise_scheduler=noise_scheduler,
                t_min=config['pda']['t_min'],
                t_max=config['pda']['t_max'],
                reverse_steps=config['pda']['reverse_steps'],
                lambda1=config['pda']['lambda1'],
                lambda2=config['pda']['lambda2'],
                device=device
            )
        
        efficiency_results = compare_efficiency(
            model, train_loader, config, pda_transform, num_batches=50
        )
        
        results['efficiency_results'] = efficiency_results
        
        plot_efficiency_comparison(
            efficiency_results,
            save_path=os.path.join(args.output_dir, 'efficiency_comparison.png')
        )
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if args.create_summary:
        summary_data = {
            'final_metrics': {
                'Clean Acc': clean_accuracy,
                'mCE': results.get('mce', 0),
                'FGSM': results.get('adversarial_results', {}).get('fgsm_eps_0.03', 0),
                'PGD': results.get('adversarial_results', {}).get('pgd_eps_0.03', 0)
            }
        }
        
        if 'corruption_results' in results:
            summary_data['corruption_results'] = results['corruption_results']
        
        if 'efficiency_results' in results:
            summary_data['efficiency_metrics'] = results['efficiency_results']['iteration_time']
        
        create_experiment_summary_plot(
            summary_data,
            save_path=os.path.join(args.output_dir, 'experiment_summary.png')
        )
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--eval_robustness', action='store_true', help='Evaluate corruption robustness')
    parser.add_argument('--eval_adversarial', action='store_true', help='Evaluate adversarial robustness')
    parser.add_argument('--eval_efficiency', action='store_true', help='Evaluate computational efficiency')
    parser.add_argument('--create_summary', action='store_true', help='Create summary visualization')
    parser.add_argument('--all', action='store_true', help='Run all evaluations')
    
    args = parser.parse_args()
    
    if args.all:
        args.eval_robustness = True
        args.eval_adversarial = True
        args.eval_efficiency = True
        args.create_summary = True
    
    evaluate_model(args)