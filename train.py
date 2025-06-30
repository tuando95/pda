#!/usr/bin/env python3
import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from src.data.datasets import get_data_loaders
from src.models.architectures import get_model
from src.augmentation.diffusion_utils import NoiseScheduler, DiffusionModelWrapper
from src.augmentation.pda import PDABatchTransform
from src.training.trainer import Trainer
from src.utils.config import convert_config_types


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train model with PDA')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--override', nargs='+', default=[], help='Override config parameters')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = OmegaConf.create(config)
    
    for override in args.override:
        key, value = override.split('=')
        print(f"Override: {key} = {value}")
        OmegaConf.update(config, key, value)
    
    config = OmegaConf.to_container(config, resolve=True)
    print(f"Before type conversion - pda.enable: {config['pda']['enable']} (type: {type(config['pda']['enable'])})")
    
    # Convert config types to ensure proper data types
    config = convert_config_types(config)
    
    set_seed(config['experiment']['seed'])
    
    device = torch.device(config['experiment']['device'])
    
    print("Loading data...")
    train_loader, val_loader, num_classes = get_data_loaders(config)
    
    config['model']['num_classes'] = num_classes
    
    print("Creating model...")
    model = get_model(config)
    
    pda_transform = None
    print(f"PDA enable flag: {config['pda']['enable']} (type: {type(config['pda']['enable'])})")
    if config['pda']['enable']:
        print("Setting up PDA...")
        
        # Don't pass model_path if it's null/None
        model_path = config['diffusion'].get('model_path')
        if model_path and model_path.lower() != 'null':
            diffusion_model = DiffusionModelWrapper(
                model_path=model_path,
                model_type=config['diffusion']['model_type']
            )
        else:
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
    else:
        print("PDA is disabled - training without augmentation")
    
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        pda_transform=pda_transform,
        device=device
    )
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.resume(args.resume)
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    main()