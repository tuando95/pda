import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import math


class NoiseScheduler:
    """Handles noise scheduling for diffusion models."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine"
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Get variance for timestep t."""
        return self._extract(self.betas, t, t.shape)
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a 1-D tensor for a batch of indices."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionForward:
    """Handles forward diffusion process."""
    
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.scheduler = noise_scheduler
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x_start according to timestep t.
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor [B, C, H, W]
            
        Returns:
            x_t: Noised images
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod = self.scheduler._extract(
            self.scheduler.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod = self.scheduler._extract(
            self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        
        return x_t, noise


class DiffusionReverse:
    """Handles reverse diffusion process."""
    
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.scheduler = noise_scheduler
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        clip_denoised: bool = True,
        return_pred_x0: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform one reverse diffusion step.
        
        Args:
            x_t: Current noisy image [B, C, H, W]
            t: Current timestep [B]
            noise_pred: Predicted noise from model [B, C, H, W]
            clip_denoised: Whether to clip the denoised image to [-1, 1]
            return_pred_x0: Whether to also return the predicted x0
            
        Returns:
            x_t_prev: Denoised image at timestep t-1
            pred_x0: (optional) Predicted clean image
        """
        betas = self.scheduler._extract(self.scheduler.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = self.scheduler._extract(
            self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_alphas_cumprod = self.scheduler._extract(
            self.scheduler.sqrt_alphas_cumprod, t, x_t.shape
        )
        
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod * noise_pred) / sqrt_alphas_cumprod
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        posterior_mean_coef1 = self.scheduler._extract(
            self.scheduler.posterior_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2 = self.scheduler._extract(
            self.scheduler.posterior_mean_coef2, t, x_t.shape
        )
        
        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t
        
        posterior_variance = self.scheduler._extract(
            self.scheduler.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self.scheduler._extract(
            self.scheduler.posterior_log_variance_clipped, t, x_t.shape
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_t_prev = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_clipped) * noise
        
        if return_pred_x0:
            return x_t_prev, pred_x0
        return x_t_prev
    
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Perform one DDIM sampling step.
        
        Args:
            x_t: Current noisy image [B, C, H, W]
            t: Current timestep [B]
            t_prev: Previous timestep [B]
            noise_pred: Predicted noise from model [B, C, H, W]
            eta: DDIM eta parameter (0 for deterministic)
            clip_denoised: Whether to clip the denoised image to [-1, 1]
            
        Returns:
            x_t_prev: Denoised image at timestep t_prev
        """
        alpha_prod_t = self.scheduler._extract(self.scheduler.alphas_cumprod, t, x_t.shape)
        alpha_prod_t_prev = self.scheduler._extract(self.scheduler.alphas_cumprod, t_prev, x_t.shape)
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_x0 = (x_t - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        sigma_t = eta * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev))
        
        pred_dir_xt = torch.sqrt(1 - alpha_prod_t_prev - sigma_t**2) * noise_pred
        
        x_prev = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev


class DiffusionModelWrapper(nn.Module):
    """Wrapper for pre-trained diffusion models."""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "adm"):
        super().__init__()
        self.model_type = model_type
        self.model = None
        
        if model_path:
            self.load_pretrained(model_path)
    
    def load_pretrained(self, model_path: str):
        """Load pre-trained diffusion model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if self.model_type == "adm":
            from .adm_model import create_adm_model
            self.model = create_adm_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.eval()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the diffusion model.
        
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        if self.model is None:
            # Try to use pre-trained models
            try:
                from .stable_diffusion_wrapper import setup_diffusion_model
                self.model = setup_diffusion_model('auto', device=x.device.type)
                print("Loaded pre-trained diffusion model successfully!")
            except Exception as e:
                print(f"Could not load pre-trained model: {e}")
                print("Falling back to untrained model...")
                from .lightweight_ddpm import create_lightweight_ddpm
                self.model = create_lightweight_ddpm(pretrained=False)
                self.model.eval()
                self.model = self.model.to(x.device)
        
        with torch.no_grad():
            # Model directly predicts noise
            return self.model(x, t)