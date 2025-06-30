#!/usr/bin/env python3
"""Ablation study for timestep range (t_min, t_max)."""

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


def run_timestep_ablation(base_config_path: str, output_dir: str):
    """Run ablation study on timestep ranges."""
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    timestep_ranges = [
        (25, 200),
        (50, 400),
        (100, 600),
        (200, 800)
    ]
    
    results = {}
    
    for t_min, t_max in timestep_ranges:
        print(f"\nRunning experiment with t_min={t_min}, t_max={t_max}")
        
        config = OmegaConf.create(base_config)
        config.pda.t_min = t_min
        config.pda.t_max = t_max
        config.experiment.name = f"ablation_timestep_{t_min}_{t_max}"
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
            t_min=t_min,
            t_max=t_max,
            reverse_steps=config['pda']['reverse_steps'],
            lambda1=config['pda']['lambda1'],
            lambda2=config['pda']['lambda2'],
            device=device
        )
        
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
        results[f"[{t_min}, {t_max}]"] = final_metrics['acc1']
        
        print(f"Final accuracy for t_min={t_min}, t_max={t_max}: {final_metrics['acc1']:.2f}%")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'timestep_ablation_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    x_labels = list(results.keys())
    accuracies = list(results.values())
    
    plot_ablation_results(
        {'Timestep Range': accuracies},
        x_label='Timestep Range [t_min, t_max]',
        title='Ablation Study: Timestep Range',
        save_path=os.path.join(output_dir, 'timestep_ablation.png')
    )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--output_dir', type=str, default='experiments/results/timestep_ablation')
    args = parser.parse_args()
    
    results = run_timestep_ablation(args.config, args.output_dir)
    
    print("\nTimestep Ablation Results:")
    for range_str, acc in results.items():
        print(f"{range_str}: {acc:.2f}%")