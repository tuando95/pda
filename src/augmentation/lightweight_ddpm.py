"""Lightweight DDPM implementation that actually works for PDA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        out = self.out(out)
        
        return out + x


class LightweightDDPM(nn.Module):
    """Lightweight DDPM for 32x32 images."""
    
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.1,
        attention_levels: Tuple[int, ...] = (2,)
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_channels, time_dim, dropout)
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if level in attention_levels:
                self.down_blocks.append(AttentionBlock(now_channels))
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1)
                )
                channels.append(now_channels)
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_dim, dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(
                        now_channels + channels.pop(),
                        out_channels,
                        time_dim,
                        dropout
                    )
                )
                now_channels = out_channels
            
            if level in attention_levels:
                self.up_blocks.append(AttentionBlock(now_channels))
            
            if level != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1)
                )
        
        # Final convolution
        self.final_norm = nn.GroupNorm(8, now_channels)
        self.final_conv = nn.Conv2d(now_channels, image_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: predicts noise from noisy input."""
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial conv
        h = self.init_conv(x)
        hs = [h]
        
        # Downsampling
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Conv2d
                h = module(h)
            hs.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # ConvTranspose2d
                h = module(h)
        
        # Final conv
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h


def create_lightweight_ddpm(pretrained: bool = False) -> LightweightDDPM:
    """Create a lightweight DDPM model."""
    model = LightweightDDPM(
        image_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_levels=(2,)
    )
    
    if pretrained:
        # Load pre-trained weights if available
        # In practice, you would load actual pre-trained weights here
        print("Note: Pre-trained weights not available. Using random initialization.")
        print("For best results, train this model on your dataset first.")
    
    return model