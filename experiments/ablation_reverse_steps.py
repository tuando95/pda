#!/usr/bin/env python3
"""Ablation study for number of reverse denoising steps (K)."""

import argparse
import os
import yaml
import torch
from omegaconf import OmegaConf

from src.data.datasets import get_data_loaders
from src.models.architectures import get_model
from src.augmentation.diffusion_utils import NoiseScheduler, DiffusionModelWrapper
from src.augmentation.pda import PDABatchTransform
from src.training.trainer import Trainer
from src.utils.visualization import plot_ablation_results
from src.utils.efficiency import EfficiencyTracker


def run_reverse_steps_ablation(base_config_path: str, output_dir: str):
    """Run ablation study on number of reverse steps."""
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    reverse_steps = [1, 2, 3, 5, 10, 20]
    
    results = {
        'accuracy': {},
        'time_overhead': {}
    }
    
    for K in reverse_steps:
        print(f"\nRunning experiment with K={K} reverse steps")
        
        config = OmegaConf.create(base_config)
        config.pda.reverse_steps = K
        config.experiment.name = f"ablation_reverse_steps_{K}"
        config.training.epochs = 100
        
        config = OmegaConf.to_container(config, resolve=True)
        
        torch.manual_seed(config['experiment']['seed'])
        device = torch.device(config['experiment']['device'])
        
        train_loader, val_loader, num_classes = get_data_loaders(config)
        config['model']['num_classes'] = num_classes
        
        model = get_model(config)
        
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
            reverse_steps=K,
            lambda1=config['pda']['lambda1'],
            lambda2=config['pda']['lambda2'],
            device=device
        )
        
        efficiency_tracker = EfficiencyTracker()
        augmentation_times = []
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 50:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            with efficiency_tracker.track_time('augmentation'):
                _, _, _ = pda_transform.augment_batch(images, labels)
            
            augmentation_times.append(efficiency_tracker.time_history[-1][1])
        
        avg_time = sum(augmentation_times) / len(augmentation_times)
        results['time_overhead'][K] = avg_time
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            pda_transform=pda_transform,
            device=device
        )
        
        trainer.train()
        
        final_metrics = trainer.validate(config['training']['epochs'] - 1)
        results['accuracy'][K] = final_metrics['acc1']
        
        print(f"K={K}: Accuracy={final_metrics['acc1']:.2f}%, Avg Time={avg_time:.4f}s")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'reverse_steps_ablation_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    steps = list(results['accuracy'].keys())
    accuracies = list(results['accuracy'].values())
    times = list(results['time_overhead'].values())
    
    ax1.plot(steps, accuracies, 'bo-', markersize=8)
    ax1.set_xlabel('Number of Reverse Steps (K)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs. Reverse Steps')
    ax1.grid(True)
    
    ax2.plot(steps, times, 'ro-', markersize=8)
    ax2.set_xlabel('Number of Reverse Steps (K)')
    ax2.set_ylabel('Augmentation Time (s)')
    ax2.set_title('Computational Cost vs. Reverse Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reverse_steps_ablation.png'), dpi=300)
    plt.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--output_dir', type=str, default='experiments/results/reverse_steps_ablation')
    args = parser.parse_args()
    
    results = run_reverse_steps_ablation(args.config, args.output_dir)
    
    print("\nReverse Steps Ablation Results:")
    print("K\tAccuracy\tTime")
    for K in sorted(results['accuracy'].keys()):
        print(f"{K}\t{results['accuracy'][K]:.2f}%\t{results['time_overhead'][K]:.4f}s")