import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
from .diffusion_utils import NoiseScheduler, DiffusionForward, DiffusionReverse, DiffusionModelWrapper


class PDATransform:
    """Partial Diffusion Augmentation transform for PyTorch datasets."""
    
    def __init__(
        self,
        diffusion_model: DiffusionModelWrapper,
        noise_scheduler: NoiseScheduler,
        t_min: int = 50,
        t_max: int = 400,
        reverse_steps: int = 3,
        prob: float = 0.5,
        device: str = 'cuda'
    ):
        """Initialize PDA transform.
        
        Args:
            diffusion_model: Pre-trained diffusion model wrapper
            noise_scheduler: Noise scheduler for diffusion process
            t_min: Minimum timestep for noise addition
            t_max: Maximum timestep for noise addition
            reverse_steps: Number of reverse denoising steps (K)
            prob: Probability of applying PDA to each sample
            device: Device to run computations on
        """
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.forward_diffusion = DiffusionForward(noise_scheduler)
        self.reverse_diffusion = DiffusionReverse(noise_scheduler)
        
        self.t_min = t_min
        self.t_max = t_max
        self.reverse_steps = reverse_steps
        self.prob = prob
        self.device = device
        
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply PDA to input tensor.
        
        Args:
            x: Input image tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Tuple of (original, noised, denoised) images
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        device = x.device
        
        apply_pda = torch.rand(batch_size) < self.prob
        
        t = torch.randint(self.t_min, self.t_max + 1, (batch_size,), device=device)
        
        x_noised, noise = self.forward_diffusion.add_noise(x, t)
        
        x_denoised = x_noised.clone()
        
        for idx in range(batch_size):
            if apply_pda[idx]:
                x_single = x_noised[idx:idx+1]
                t_single = t[idx:idx+1]
                
                for k in range(self.reverse_steps):
                    current_t = t_single - k
                    if current_t > 0:
                        with torch.no_grad():
                            noise_pred = self.diffusion_model(x_single.to(self.device), current_t.to(self.device))
                        x_single = self.reverse_diffusion.denoise_step(
                            x_single.to(self.device),
                            current_t.to(self.device),
                            noise_pred
                        )
                
                x_denoised[idx] = x_single.squeeze(0).to(device)
        
        if squeeze_output:
            return x.squeeze(0), x_noised.squeeze(0), x_denoised.squeeze(0)
        
        return x, x_noised, x_denoised


class PDABatchTransform:
    """Efficient batch-wise PDA transform."""
    
    def __init__(
        self,
        diffusion_model: DiffusionModelWrapper,
        noise_scheduler: NoiseScheduler,
        t_min: int = 50,
        t_max: int = 400,
        reverse_steps: int = 3,
        lambda1: float = 0.5,
        lambda2: float = 0.5,
        device: str = 'cuda'
    ):
        """Initialize batch PDA transform.
        
        Args:
            diffusion_model: Pre-trained diffusion model wrapper
            noise_scheduler: Noise scheduler for diffusion process
            t_min: Minimum timestep for noise addition
            t_max: Maximum timestep for noise addition
            reverse_steps: Number of reverse denoising steps (K)
            lambda1: Weight for noised image loss
            lambda2: Weight for denoised image loss
            device: Device to run computations on
        """
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.forward_diffusion = DiffusionForward(noise_scheduler)
        self.reverse_diffusion = DiffusionReverse(noise_scheduler)
        
        self.t_min = t_min
        self.t_max = t_max
        self.reverse_steps = reverse_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        
        self.diffusion_model.to(device)
        self.diffusion_model.eval()
    
    def augment_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply PDA to a batch of images.
        
        Args:
            x: Batch of images [B, C, H, W]
            y: Batch of labels [B]
            
        Returns:
            Tuple of (images, labels, weights) for augmented batch
        """
        batch_size = x.shape[0]
        device = x.device
        
        t = torch.randint(self.t_min, self.t_max + 1, (batch_size,), device=device)
        
        x_noised, noise = self.forward_diffusion.add_noise(x, t)
        
        x_denoised = x_noised.clone().to(self.device)
        t_device = t.to(self.device)
        
        for k in range(self.reverse_steps):
            current_t = t_device - k
            mask = current_t > 0
            
            if mask.any():
                with torch.no_grad():
                    noise_pred = self.diffusion_model(x_denoised, current_t)
                
                x_denoised = self.reverse_diffusion.denoise_step(
                    x_denoised,
                    current_t,
                    noise_pred
                )
        
        x_denoised = x_denoised.to(device)
        
        x_combined = torch.cat([x, x_noised, x_denoised], dim=0)
        y_combined = torch.cat([y, y, y], dim=0)
        
        weights = torch.cat([
            torch.ones(batch_size, device=device),
            torch.full((batch_size,), self.lambda1, device=device),
            torch.full((batch_size,), self.lambda2, device=device)
        ])
        
        return x_combined, y_combined, weights


class PDADataset(Dataset):
    """Dataset wrapper that applies PDA on-the-fly."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        pda_transform: PDATransform,
        return_all_views: bool = True
    ):
        """Initialize PDA dataset wrapper.
        
        Args:
            base_dataset: Original dataset
            pda_transform: PDA transform to apply
            return_all_views: If True, return all three views (original, noised, denoised)
        """
        self.base_dataset = base_dataset
        self.pda_transform = pda_transform
        self.return_all_views = return_all_views
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """Get item with PDA applied.
        
        Returns:
            Tuple of ((x_orig, x_noised, x_denoised), y) if return_all_views=True
            Tuple of (x_denoised, y) if return_all_views=False
        """
        x, y = self.base_dataset[idx]
        
        x_orig, x_noised, x_denoised = self.pda_transform(x)
        
        if self.return_all_views:
            return (x_orig, x_noised, x_denoised), y
        else:
            return x_denoised, y


def create_pda_transform(
    config: Dict[str, Any],
    diffusion_model: Optional[DiffusionModelWrapper] = None
) -> PDATransform:
    """Create PDA transform from config.
    
    Args:
        config: Configuration dictionary
        diffusion_model: Optional pre-loaded diffusion model
        
    Returns:
        PDATransform instance
    """
    if diffusion_model is None:
        diffusion_model = DiffusionModelWrapper(
            model_path=config['diffusion']['model_path'],
            model_type=config['diffusion']['model_type']
        )
    
    noise_scheduler = NoiseScheduler(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule_type=config['diffusion']['noise_schedule']
    )
    
    pda_transform = PDATransform(
        diffusion_model=diffusion_model,
        noise_scheduler=noise_scheduler,
        t_min=config['pda']['t_min'],
        t_max=config['pda']['t_max'],
        reverse_steps=config['pda']['reverse_steps'],
        prob=config['pda']['prob'],
        device=config['experiment']['device']
    )
    
    return pda_transform