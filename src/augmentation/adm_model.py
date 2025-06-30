"""Placeholder for ADM model architecture.

This file would contain the actual ADM (Ablated Diffusion Model) architecture
if using a real pre-trained model. For demonstration purposes, we provide
a mock implementation.
"""

import torch
import torch.nn as nn


class MockADMModel(nn.Module):
    """Mock ADM model for testing without actual pre-trained weights."""
    
    def __init__(self, in_channels=3, model_channels=128, out_channels=3):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList([
            ResBlock(model_channels, model_channels),
            ResBlock(model_channels, model_channels),
            ResBlock(model_channels, model_channels),
        ])
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000)
        
        h = self.input_conv(x)
        
        for block in self.blocks:
            h = block(h, t_emb)
        
        return self.output_conv(h)


class ResBlock(nn.Module):
    """Residual block for the mock ADM model."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        self.time_emb_proj = nn.Linear(in_channels, out_channels)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


def create_adm_model():
    """Create ADM model instance.
    
    In a real implementation, this would load the actual pre-trained
    ADM architecture. For now, returns a mock model.
    """
    return MockADMModel()