import torch
from aurora_torch.main import SwinTransformerUNet3D
from loguru import logger

# Example usage
if __name__ == "__main__":
    # Test with random input tensor of shape (B, D, H, W, C)
    B, D, H, W, C = 2, 16, 64, 64, 32
    model = SwinTransformerUNet3D(input_dim=C, output_dim=C)
    input_tensor = torch.rand(B, D, H, W, C)

    # Forward pass through the model
    output = model(input_tensor)
    logger.info(f"Output shape: {output.shape}")
