"""Wrapper for OpenAI's Guided Diffusion models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import os
import sys


class GuidedDiffusionWrapper(nn.Module):
    """Wrapper for OpenAI's Guided Diffusion models."""
    
    def __init__(self, model_path: str, image_size: int = 64, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        
        # Import guided diffusion
        try:
            # Add guided diffusion to path if cloned locally
            guided_diffusion_path = './guided-diffusion'
            if os.path.exists(guided_diffusion_path):
                sys.path.append(guided_diffusion_path)
            
            from guided_diffusion.script_util import (
                model_and_diffusion_defaults,
                create_model_and_diffusion
            )
        except ImportError:
            print("Please install guided-diffusion:")
            print("git clone https://github.com/openai/guided-diffusion.git")
            print("cd guided-diffusion && pip install -e .")
            raise
        
        # Model configuration based on image size
        if image_size == 64:
            model_config = {
                'image_size': 64,
                'num_channels': 192,
                'num_res_blocks': 3,
                'num_heads': 4,
                'num_heads_upsample': -1,
                'attention_resolutions': '32,16,8',
                'channel_mult': '',
                'dropout': 0.1,
                'class_cond': False,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False
            }
        elif image_size == 256:
            model_config = {
                'image_size': 256,
                'num_channels': 256,
                'num_res_blocks': 2,
                'num_heads': 4,
                'num_heads_upsample': -1,
                'attention_resolutions': '32,16,8',
                'channel_mult': '',
                'dropout': 0.0,
                'class_cond': False,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False
            }
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
        
        # Add diffusion configuration
        model_config.update({
            'diffusion_steps': 1000,
            'learn_sigma': True,
            'sigma_small': False,
            'noise_schedule': 'linear',
            'use_kl': False,
            'predict_xstart': False,
            'rescale_timesteps': False,
            'rescale_learned_sigmas': False,
            'timestep_respacing': ''
        })
        
        # Create model and diffusion
        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded Guided Diffusion model for {image_size}x{image_size} images")
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy input."""
        with torch.no_grad():
            # Guided diffusion expects different input format
            model_output = self.model(x_t, t)
            
            # Extract predicted noise (first half of output if learn_sigma=True)
            if model_output.shape[1] == 6:  # learn_sigma=True
                eps_pred = model_output[:, :3]
            else:
                eps_pred = model_output
            
            return eps_pred
    
    def apply_pda(
        self,
        x_0: torch.Tensor,
        t_min: int = 50,
        t_max: int = 400,
        num_reverse_steps: int = 3
    ) -> torch.Tensor:
        """Apply PDA using guided diffusion."""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(t_min, t_max + 1, (batch_size,), device=device)
        
        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise=noise)
        
        # Reverse diffusion for K steps
        x_current = x_t
        for k in range(num_reverse_steps):
            t_current = t - k
            
            # Skip if we've reached t=0
            mask = (t_current > 0)
            if not mask.any():
                break
            
            # Use guided diffusion's reverse step
            with torch.no_grad():
                out = self.diffusion.p_mean_variance(
                    self.model,
                    x_current[mask],
                    t_current[mask],
                    clip_denoised=True
                )
                
                # Sample from the distribution
                noise = torch.randn_like(x_current[mask])
                nonzero_mask = (t_current[mask] != 0).float().view(-1, 1, 1, 1)
                x_denoised = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                
                x_current[mask] = x_denoised
        
        return x_current


def download_guided_diffusion_model(image_size: int = 64):
    """Download pre-trained guided diffusion model."""
    import urllib.request
    
    urls = {
        64: "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt",
        256: "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    }
    
    if image_size not in urls:
        raise ValueError(f"No pre-trained model available for size {image_size}")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/guided_diffusion_{image_size}x{image_size}.pt"
    
    if not os.path.exists(model_path):
        print(f"Downloading Guided Diffusion model for {image_size}x{image_size} images...")
        urllib.request.urlretrieve(urls[image_size], model_path)
        print(f"Model downloaded to {model_path}")
    
    return model_path