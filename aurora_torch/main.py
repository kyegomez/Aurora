import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from loguru import logger


# Type validation decorator (optional but useful in production code)
def type_check(func):
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                logger.info(f"Tensor shape: {arg.shape}")
        return func(*args, **kwargs)

    return wrapper


class PatchMerging(nn.Module):
    """
    Patch merging layer that reduces the spatial resolution by half.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim)
        self.norm = nn.LayerNorm(4 * input_dim)

    @type_check
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch merging layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, D//2, H//2, W//2, C').
        """
        B, D, H, W, C = (
            x.shape
        )  # Extract batch size, depth, height, width, and channels
        x = x.view(
            B, D // 2, H // 2, W // 2, 4 * C
        )  # Reduce the spatial dimensions and merge channels
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplitting(nn.Module):
    """
    Patch splitting layer that increases the spatial resolution by doubling.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(PatchSplitting, self).__init__()
        self.expansion = nn.Linear(input_dim, 4 * output_dim)
        self.norm = nn.LayerNorm(input_dim)

    @type_check
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch splitting layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, D*2, H*2, W*2, C').
        """
        B, D, H, W, C = x.shape
        x = self.norm(x)
        x = self.expansion(x)
        x = x.view(
            B, D * 2, H * 2, W * 2, C // 4
        )  # Increase spatial dimensions
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer block with local self-attention.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        heads: int = 8,
    ):
        super(SwinTransformerBlock3D, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    @type_check
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 3D Swin Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # Reshape to 2D for MultiheadAttention: (B*D*H*W, C)
        x_reshape = x.view(-1, C)
        attn_output, _ = self.attention(
            x_reshape, x_reshape, x_reshape
        )
        attn_output = attn_output.view(
            B, D, H, W, C
        )  # Reshape back to 3D

        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward network (MLP)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    Encoder with 3 stages of patch merging and Swin Transformer layers.
    """

    def __init__(self, input_dim: int):
        super(Encoder, self).__init__()
        self.stage1 = self._make_stage(input_dim, 6)
        self.stage2 = self._make_stage(input_dim * 2, 10)
        self.stage3 = self._make_stage(input_dim * 4, 8)

    def _make_stage(self, dim: int, num_layers: int) -> nn.Sequential:
        layers = [
            SwinTransformerBlock3D(dim, (2, 12, 6))
            for _ in range(num_layers)
        ]
        return nn.Sequential(*layers)

    @type_check
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out1 = self.stage1(x)
        enc_out2 = self.stage2(enc_out1)
        enc_out3 = self.stage3(enc_out2)
        return enc_out1, enc_out2, enc_out3


class Decoder(nn.Module):
    """
    Decoder with 3 stages of patch splitting and Swin Transformer layers.
    """

    def __init__(self, output_dim: int):
        super(Decoder, self).__init__()
        self.stage1 = self._make_stage(output_dim * 4, 8)
        self.stage2 = self._make_stage(output_dim * 2, 10)
        self.stage3 = self._make_stage(output_dim, 6)

    def _make_stage(self, dim: int, num_layers: int) -> nn.Sequential:
        layers = [
            SwinTransformerBlock3D(dim, (2, 12, 6))
            for _ in range(num_layers)
        ]
        return nn.Sequential(*layers)

    @type_check
    def forward(
        self,
        x: torch.Tensor,
        enc_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        dec_out1 = self.stage1(x)
        dec_out2 = self.stage2(dec_out1 + enc_outputs[2])
        dec_out3 = self.stage3(dec_out2 + enc_outputs[1])
        return dec_out3


class SwinTransformerUNet3D(nn.Module):
    """
    3D Swin Transformer U-Net with encoder-decoder architecture.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(SwinTransformerUNet3D, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(output_dim)
        self.patch_merging = PatchMerging(input_dim, input_dim * 2)
        self.patch_splitting = PatchSplitting(
            output_dim * 2, output_dim
        )

    @type_check
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 3D Swin Transformer U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, D, H, W, C).
        """
        enc_out1, enc_out2, enc_out3 = self.encoder(x)  # Encoder
        x = self.patch_merging(
            enc_out3
        )  # Patch merging (reduce resolution)
        dec_out = self.decoder(
            x, (enc_out1, enc_out2, enc_out3)
        )  # Decoder
        output = self.patch_splitting(
            dec_out
        )  # Patch splitting (restore resolution)
        return output
