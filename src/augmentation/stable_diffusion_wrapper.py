"""Wrapper for Stable Diffusion models (more memory efficient)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
try:
    from diffusers import DDPMPipeline, DDIMPipeline, AutoencoderKL
    from diffusers.models import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class HuggingFaceDiffusionWrapper(nn.Module):
    """Wrapper for Hugging Face diffusion models."""
    
    def __init__(self, model_name: str = "google/ddpm-cifar10-32", device: str = 'cuda'):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Please install diffusers: pip install diffusers")
        
        self.device = device
        
        # Load pre-trained pipeline
        print(f"Loading {model_name}...")
        if "ddpm" in model_name:
            self.pipeline = DDPMPipeline.from_pretrained(model_name)
        elif "ddim" in model_name:
            self.pipeline = DDIMPipeline.from_pretrained(model_name)
        else:
            # Try generic loading
            self.pipeline = DDPMPipeline.from_pretrained(model_name)
        
        # Extract the UNet model
        self.model = self.pipeline.unet
        self.model.to(device)
        self.model.eval()
        
        # Get the noise scheduler
        self.scheduler = self.pipeline.scheduler
        
        print(f"Loaded {model_name} successfully")
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy input."""
        with torch.no_grad():
            # Hugging Face models expect timestep as input
            return self.model(x_t, t).sample
    
    def apply_pda(
        self,
        x_0: torch.Tensor,
        t_min: int = 50,
        t_max: int = 400,
        num_reverse_steps: int = 3
    ) -> torch.Tensor:
        """Apply PDA using Hugging Face diffusion models."""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(t_min, t_max + 1, (batch_size,), device=device)
        
        # Forward diffusion using the scheduler
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, noise, t)
        
        # Reverse diffusion for K steps
        x_current = x_t
        for k in range(num_reverse_steps):
            t_current = t - k
            
            # Skip if we've reached t=0
            mask = (t_current > 0)
            if not mask.any():
                break
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(x_current[mask], t_current[mask]).sample
            
            # Use scheduler's step function
            for i, idx in enumerate(torch.where(mask)[0]):
                # Set the timestep for the scheduler
                self.scheduler.set_timesteps(1000)
                
                # Perform one denoising step
                x_current[idx] = self.scheduler.step(
                    noise_pred[i],
                    t_current[idx],
                    x_current[idx]
                ).prev_sample
        
        return x_current


class CompactDiffusionWrapper(nn.Module):
    """Wrapper for smaller, CIFAR-trained diffusion models."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # Available pre-trained models for CIFAR-10
        self.available_models = {
            'cifar10': 'google/ddpm-cifar10-32',
            'celeba': 'google/ddpm-celebahq-256',
            'church': 'google/ddpm-church-256',
            'bedroom': 'google/ddpm-bedroom-256'
        }
        
        # Use CIFAR-10 model by default (smallest and most relevant)
        self.wrapper = HuggingFaceDiffusionWrapper(
            model_name=self.available_models['cifar10'],
            device=device
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.wrapper(x_t, t)
    
    def apply_pda(self, x_0: torch.Tensor, t_min: int = 50, t_max: int = 400, 
                  num_reverse_steps: int = 3) -> torch.Tensor:
        return self.wrapper.apply_pda(x_0, t_min, t_max, num_reverse_steps)


def setup_diffusion_model(model_type: str = 'auto', device: str = 'cuda') -> nn.Module:
    """Setup the best available diffusion model."""
    
    if model_type == 'auto':
        # Try to use the best available model
        if DIFFUSERS_AVAILABLE:
            print("Using Hugging Face diffusion models...")
            return CompactDiffusionWrapper(device)
        else:
            print("Diffusers not installed. Please run: pip install diffusers")
            print("Falling back to untrained model...")
            from .lightweight_ddpm import create_lightweight_ddpm
            return create_lightweight_ddpm()
    
    elif model_type == 'guided':
        # Use OpenAI's guided diffusion
        from .guided_diffusion_wrapper import GuidedDiffusionWrapper, download_guided_diffusion_model
        model_path = download_guided_diffusion_model(64)
        return GuidedDiffusionWrapper(model_path, image_size=64, device=device)
    
    elif model_type == 'huggingface':
        # Use Hugging Face models
        return CompactDiffusionWrapper(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Easy-to-use function for PDA
def create_pda_with_pretrained(device: str = 'cuda') -> 'PDAWithPretrained':
    """Create PDA augmentation with the best available pre-trained model."""
    
    class PDAWithPretrained:
        def __init__(self):
            self.model = setup_diffusion_model('auto', device)
            self.device = device
        
        def augment(self, images: torch.Tensor, t_min: int = 50, t_max: int = 400,
                   num_reverse_steps: int = 3) -> torch.Tensor:
            """Apply PDA augmentation to images."""
            return self.model.apply_pda(images, t_min, t_max, num_reverse_steps)
    
    return PDAWithPretrained()