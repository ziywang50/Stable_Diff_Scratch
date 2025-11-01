import torch
from torch import nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Initial convolution to get to the right number of channels
            # (3, 256, 256) -> (128, 256, 256)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Downsample blocks: progressively reduce spatial dimensions and increase channels. 
            # Residual blocks will not change size of image
            # (128, 256, 256) -> (128, 128, 128)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (128, 128, 128) -> (256, 64, 64)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            # (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (256, 64, 64) -> (512, 32, 32)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # Three residual blocks at 512 channels before attention
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Add attention at 32x32 spatial dimension
            VAE_AttentionBlock(512),

            # Final blocks
            VAE_ResidualBlock(512, 512),

            # Normalization and activation before final projection
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # Project to latent space (mean and log variance)
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

        # Keep the same ordered list of submodules but register them on an
        # internal Sequential container. This preserves the original
        # behaviour while allowing the class to be an nn.Module.

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel=3, height=256, width=256)
        # Iterate the registered blocks and keep the original behaviour where we
        # pad before strided convolutions to mimic the tutorial implementation.
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Pad right and bottom by 1 (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # (batch_size, 8, height/8, width/8) -> two tensors with size (batch_size, 4, height/8, width/8)
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        # Reparameterization trick
        # std = exp(0.5 * logvar)
        # z = mean + std * noise
        std = torch.exp(0.5 * logvar)
        z = (mean + std * noise) * 0.18215

        return z
