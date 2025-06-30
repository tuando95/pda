import pytest
import torch
import numpy as np

from src.augmentation.diffusion_utils import NoiseScheduler, DiffusionModelWrapper
from src.augmentation.pda import (
    PDATransform,
    PDABatchTransform,
    PDADataset,
    create_pda_transform
)
from src.augmentation.adm_model import MockADMModel


class TestPDATransform:
    """Test PDA transform functionality."""
    
    @pytest.fixture
    def pda_transform(self):
        model = MockADMModel()
        diffusion_model = DiffusionModelWrapper(model_type="adm")
        diffusion_model.model = model
        
        noise_scheduler = NoiseScheduler(num_timesteps=1000)
        
        transform = PDATransform(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            t_min=50,
            t_max=400,
            reverse_steps=3,
            prob=1.0,
            device='cpu'
        )
        return transform
    
    def test_single_image_transform(self, pda_transform):
        x = torch.randn(3, 32, 32)
        x_orig, x_noised, x_denoised = pda_transform(x)
        
        assert x_orig.shape == x.shape
        assert x_noised.shape == x.shape
        assert x_denoised.shape == x.shape
        assert torch.allclose(x_orig, x)
    
    def test_batch_transform(self, pda_transform):
        x = torch.randn(4, 3, 32, 32)
        x_orig, x_noised, x_denoised = pda_transform(x)
        
        assert x_orig.shape == x.shape
        assert x_noised.shape == x.shape
        assert x_denoised.shape == x.shape
    
    def test_probability_application(self):
        model = MockADMModel()
        diffusion_model = DiffusionModelWrapper(model_type="adm")
        diffusion_model.model = model
        
        noise_scheduler = NoiseScheduler(num_timesteps=1000)
        
        transform = PDATransform(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            t_min=50,
            t_max=400,
            reverse_steps=3,
            prob=0.0,  # Never apply PDA
            device='cpu'
        )
        
        x = torch.randn(1, 3, 32, 32)
        x_orig, x_noised, x_denoised = transform(x)
        
        # When prob=0, denoised should equal noised
        assert torch.allclose(x_noised, x_denoised)


class TestPDABatchTransform:
    """Test batch PDA transform functionality."""
    
    @pytest.fixture
    def batch_transform(self):
        model = MockADMModel()
        diffusion_model = DiffusionModelWrapper(model_type="adm")
        diffusion_model.model = model
        
        noise_scheduler = NoiseScheduler(num_timesteps=1000)
        
        transform = PDABatchTransform(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            t_min=50,
            t_max=400,
            reverse_steps=3,
            lambda1=0.5,
            lambda2=0.5,
            device='cpu'
        )
        return transform
    
    def test_augment_batch(self, batch_transform):
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        
        x_combined, y_combined, weights = batch_transform.augment_batch(x, y)
        
        assert x_combined.shape[0] == 3 * x.shape[0]  # Original + noised + denoised
        assert y_combined.shape[0] == 3 * y.shape[0]
        assert weights.shape[0] == 3 * x.shape[0]
        
        assert torch.all(weights[:8] == 1.0)  # Original weights
        assert torch.all(weights[8:16] == 0.5)  # Noised weights (lambda1)
        assert torch.all(weights[16:24] == 0.5)  # Denoised weights (lambda2)


class TestPDADataset:
    """Test PDA dataset wrapper."""
    
    def test_dataset_wrapper(self):
        # Create a simple dataset
        class SimpleDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx
        
        base_dataset = SimpleDataset()
        
        model = MockADMModel()
        diffusion_model = DiffusionModelWrapper(model_type="adm")
        diffusion_model.model = model
        
        noise_scheduler = NoiseScheduler(num_timesteps=1000)
        pda_transform = PDATransform(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            device='cpu'
        )
        
        pda_dataset = PDADataset(
            base_dataset=base_dataset,
            pda_transform=pda_transform,
            return_all_views=True
        )
        
        assert len(pda_dataset) == len(base_dataset)
        
        (x_orig, x_noised, x_denoised), y = pda_dataset[0]
        assert x_orig.shape == (3, 32, 32)
        assert x_noised.shape == (3, 32, 32)
        assert x_denoised.shape == (3, 32, 32)
        assert y == 0


class TestCreatePDATransform:
    """Test PDA transform creation from config."""
    
    def test_create_from_config(self):
        config = {
            'diffusion': {
                'model_path': None,
                'model_type': 'adm',
                'num_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'noise_schedule': 'linear'
            },
            'pda': {
                't_min': 50,
                't_max': 400,
                'reverse_steps': 3,
                'prob': 0.5
            },
            'experiment': {
                'device': 'cpu'
            }
        }
        
        # Mock the diffusion model
        model = MockADMModel()
        diffusion_model = DiffusionModelWrapper(model_type="adm")
        diffusion_model.model = model
        
        transform = create_pda_transform(config, diffusion_model)
        
        assert transform.t_min == 50
        assert transform.t_max == 400
        assert transform.reverse_steps == 3
        assert transform.prob == 0.5