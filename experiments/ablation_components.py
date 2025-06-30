#!/usr/bin/env python3
"""Ablation study for PDA components (original, noised, denoised contributions)."""

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


class ComponentAblationTransform:
    """Modified PDA transform for component ablation."""
    
    def __init__(self, pda_transform, component_mode='all'):
        self.pda_transform = pda_transform
        self.component_mode = component_mode
    
    def augment_batch(self, x, y):
        """Apply selective component augmentation."""
        if self.component_mode == 'original_only':
            return x, y, torch.ones(x.size(0), device=x.device)
        
        elif self.component_mode == 'noised_only':
            batch_size = x.shape[0]
            device = x.device
            
            t = torch.randint(
                self.pda_transform.t_min,
                self.pda_transform.t_max + 1,
                (batch_size,),
                device=device
            )
            
            x_noised, _ = self.pda_transform.forward_diffusion.add_noise(x, t)
            
            x_combined = torch.cat([x, x_noised], dim=0)
            y_combined = torch.cat([y, y], dim=0)
            weights = torch.cat([
                torch.ones(batch_size, device=device),
                torch.full((batch_size,), self.pda_transform.lambda1, device=device)
            ])
            
            return x_combined, y_combined, weights
        
        elif self.component_mode == 'denoised_only':
            x_all, y_all, weights_all = self.pda_transform.augment_batch(x, y)
            
            batch_size = x.size(0)
            x_combined = torch.cat([x_all[:batch_size], x_all[2*batch_size:]], dim=0)
            y_combined = torch.cat([y_all[:batch_size], y_all[2*batch_size:]], dim=0)
            weights = torch.cat([weights_all[:batch_size], weights_all[2*batch_size:]])
            
            return x_combined, y_combined, weights
        
        elif self.component_mode == 'noised_denoised':
            x_all, y_all, weights_all = self.pda_transform.augment_batch(x, y)
            
            batch_size = x.size(0)
            x_combined = x_all[batch_size:]
            y_combined = y_all[batch_size:]
            weights = weights_all[batch_size:]
            
            return x_combined, y_combined, weights
        
        else:
            return self.pda_transform.augment_batch(x, y)


def run_component_ablation(base_config_path: str, output_dir: str):
    """Run ablation study on PDA components."""
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    component_modes = [
        'original_only',
        'noised_only',
        'denoised_only',
        'noised_denoised',
        'all'
    ]
    
    results = {}
    
    for mode in component_modes:
        print(f"\nRunning experiment with component mode: {mode}")
        
        config = OmegaConf.create(base_config)
        config.experiment.name = f"ablation_component_{mode}"
        config.training.epochs = 100
        
        config = OmegaConf.to_container(config, resolve=True)
        
        torch.manual_seed(config['experiment']['seed'])
        device = torch.device(config['experiment']['device'])
        
        train_loader, val_loader, num_classes = get_data_loaders(config)
        config['model']['num_classes'] = num_classes
        
        model = get_model(config)
        
        if mode != 'original_only':
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
            
            base_pda_transform = PDABatchTransform(
                diffusion_model=diffusion_model,
                noise_scheduler=noise_scheduler,
                t_min=config['pda']['t_min'],
                t_max=config['pda']['t_max'],
                reverse_steps=config['pda']['reverse_steps'],
                lambda1=config['pda']['lambda1'],
                lambda2=config['pda']['lambda2'],
                device=device
            )
            
            pda_transform = ComponentAblationTransform(base_pda_transform, mode)
        else:
            pda_transform = None
        
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
        results[mode] = final_metrics['acc1']
        
        print(f"Final accuracy for {mode}: {final_metrics['acc1']:.2f}%")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'component_ablation_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['gray', 'orange', 'green', 'blue', 'red']
    bars = ax.bar(modes, accuracies, color=colors)
    
    ax.set_xlabel('Component Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('PDA Component Ablation Study')
    ax.grid(True, axis='y')
    
    for i, (mode, acc) in enumerate(zip(modes, accuracies)):
        ax.text(i, acc + 0.5, f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_ablation.png'), dpi=300)
    plt.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--output_dir', type=str, default='experiments/results/component_ablation')
    args = parser.parse_args()
    
    results = run_component_ablation(args.config, args.output_dir)
    
    print("\nComponent Ablation Results:")
    for mode, acc in results.items():
        print(f"{mode}: {acc:.2f}%")