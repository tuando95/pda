import pytest
import torch
import numpy as np

from src.augmentation.diffusion_utils import (
    NoiseScheduler,
    DiffusionForward,
    DiffusionReverse
)


class TestNoiseScheduler:
    """Test noise scheduler functionality."""
    
    def test_linear_schedule(self):
        scheduler = NoiseScheduler(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type="linear"
        )
        
        assert len(scheduler.betas) == 1000
        assert scheduler.betas[0] >= 0.0001
        assert scheduler.betas[-1] <= 0.02
        assert torch.all(scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1])
    
    def test_cosine_schedule(self):
        scheduler = NoiseScheduler(
            num_timesteps=1000,
            schedule_type="cosine"
        )
        
        assert len(scheduler.betas) == 1000
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
        assert torch.all(scheduler.alphas_cumprod >= 0)
        assert torch.all(scheduler.alphas_cumprod <= 1)
    
    def test_variance_extraction(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        t = torch.tensor([0, 100, 500, 999])
        variances = scheduler.get_variance(t)
        
        assert variances.shape == (4,)
        assert torch.all(variances >= 0)


class TestDiffusionForward:
    """Test forward diffusion process."""
    
    def test_add_noise(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        forward_diffusion = DiffusionForward(scheduler)
        
        x_start = torch.randn(4, 3, 32, 32)
        t = torch.tensor([100, 200, 300, 400])
        
        x_t, noise = forward_diffusion.add_noise(x_start, t)
        
        assert x_t.shape == x_start.shape
        assert noise.shape == x_start.shape
        
    def test_noise_level_increases(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        forward_diffusion = DiffusionForward(scheduler)
        
        x_start = torch.ones(1, 3, 32, 32)
        
        t_small = torch.tensor([100])
        t_large = torch.tensor([900])
        
        x_t_small, _ = forward_diffusion.add_noise(x_start, t_small)
        x_t_large, _ = forward_diffusion.add_noise(x_start, t_large)
        
        var_small = torch.var(x_t_small - x_start)
        var_large = torch.var(x_t_large - x_start)
        
        assert var_large > var_small


class TestDiffusionReverse:
    """Test reverse diffusion process."""
    
    def test_denoise_step(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        reverse_diffusion = DiffusionReverse(scheduler)
        
        x_t = torch.randn(4, 3, 32, 32)
        t = torch.tensor([500, 600, 700, 800])
        noise_pred = torch.randn_like(x_t)
        
        x_t_prev = reverse_diffusion.denoise_step(x_t, t, noise_pred)
        
        assert x_t_prev.shape == x_t.shape
    
    def test_denoise_with_pred_x0(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        reverse_diffusion = DiffusionReverse(scheduler)
        
        x_t = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])
        noise_pred = torch.randn_like(x_t)
        
        x_t_prev, pred_x0 = reverse_diffusion.denoise_step(
            x_t, t, noise_pred, return_pred_x0=True
        )
        
        assert x_t_prev.shape == x_t.shape
        assert pred_x0.shape == x_t.shape
        assert torch.all(pred_x0 >= -1) and torch.all(pred_x0 <= 1)
    
    def test_ddim_step(self):
        scheduler = NoiseScheduler(num_timesteps=1000)
        reverse_diffusion = DiffusionReverse(scheduler)
        
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([500, 600])
        t_prev = torch.tensor([400, 500])
        noise_pred = torch.randn_like(x_t)
        
        x_t_prev = reverse_diffusion.ddim_step(x_t, t, t_prev, noise_pred, eta=0.0)
        
        assert x_t_prev.shape == x_t.shape