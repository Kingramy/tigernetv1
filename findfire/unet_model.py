import torch
import torch.nn as nn


class DoubleConv(nn.Module):
  """A sequence of two 3x3 convolutions, each followed by BatchNorm and ReLU.

  Formula: [Conv2d(3x3) -> BatchNorm2d -> ReLU] x 2
  Padding=1 preserves spatial dimensions (H, W).
  """

  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,  # Bias is redundant when followed immediately by BatchNorm
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.conv_block(x)


class UNet(nn.Module):
  """Standard U-Net architecture adapted for multi-spectral thermal segmentation.

  Inputs:  Tensor of shape [Batch, in_channels, 128, 256]
  Outputs: Tensor of shape [Batch, out_channels, 128, 256] (Raw Logits)
  """

  def __init__(self, in_channels=4, out_channels=1, base_features=64):
    super(UNet, self).__init__()

    f = base_features  # 64 base filters

    # 1. Contracting Path (Encoder)
    self.inc = DoubleConv(in_channels, f)  # 3 -> 64
    self.down1 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(f, f * 2)  # 64 -> 128
    )
    self.down2 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        DoubleConv(f * 2, f * 4),  # 128 -> 256
    )
    self.down3 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        DoubleConv(f * 4, f * 8),  # 256 -> 512
    )

    # 2. Bottleneck
    self.down4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        DoubleConv(f * 8, f * 16),  # 512 -> 1024
    )

    # 3. Expanding Path (Decoder)
    # Level 4 Up: 1024 -> 512
    self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
    self.conv1 = DoubleConv(
        f * 16, f * 8
    )  # Concatenated: 512 (up) + 512 (skip) = 1024 -> 512

    # Level 3 Up: 512 -> 256
    self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
    self.conv2 = DoubleConv(
        f * 8, f * 4
    )  # Concatenated: 256 (up) + 256 (skip) = 512 -> 256

    # Level 2 Up: 256 -> 128
    self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
    self.conv3 = DoubleConv(
        f * 4, f * 2
    )  # Concatenated: 128 (up) + 128 (skip) = 256 -> 128

    # Level 1 Up: 128 -> 64
    self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
    self.conv4 = DoubleConv(
        f * 2, f
    )  # Concatenated: 64 (up) + 64 (skip) = 128 -> 64

    # 4. Final Output Projection (1x1 Convolution)
    # Maps 64 feature maps to 1 channel of raw logits (No Sigmoid here; applied in Loss)
    self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

  def forward(self, x):
    # Encoder with skip connections saved
    x1 = self.inc(x)  # Level 1: [B, 64, 128, 256]
    x2 = self.down1(x1)  # Level 2: [B, 128, 64, 128]
    x3 = self.down2(x2)  # Level 3: [B, 256, 32, 64]
    x4 = self.down3(x3)  # Level 4: [B, 512, 16, 32]

    # Bottleneck
    x5 = self.down4(x4)  # Bottleneck: [B, 1024, 8, 16]

    # Decoder with concatenation along channel dimension (dim=1)
    # Up 1
    d4 = self.up1(x5)  # [B, 512, 16, 32]
    d4 = torch.cat([x4, d4], dim=1)  # [B, 1024, 16, 32]
    d4 = self.conv1(d4)  # [B, 512, 16, 32]

    # Up 2
    d3 = self.up2(d4)  # [B, 256, 32, 64]
    d3 = torch.cat([x3, d3], dim=1)  # [B, 512, 32, 64]
    d3 = self.conv2(d3)  # [B, 256, 32, 64]

    # Up 3
    d2 = self.up3(d3)  # [B, 128, 64, 128]
    d2 = torch.cat([x2, d2], dim=1)  # [B, 256, 64, 128]
    d2 = self.conv3(d2)  # [B, 128, 64, 128]

    # Up 4
    d1 = self.up4(d2)  # [B, 64, 128, 256]
    d1 = torch.cat([x1, d1], dim=1)  # [B, 128, 128, 256]
    d1 = self.conv4(d1)  # [B, 64, 128, 256]

    # Final 1x1 Convolution
    logits = self.out_conv(d1)  # [B, 1, 128, 256]

    return logits


# =====================================================================
# VERIFICATION & HARDWARE SANITY CHECK
# =====================================================================
if __name__ == "__main__":
  # 1. Determine execution device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Executing architecture check on device: {device}")
  if device.type == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(
        f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )

  # 2. Instantiate Model
  model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)

  # 3. Calculate trainable parameter count
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(
      f"Total Trainable Parameters: {total_params:,} (~{total_params * 4 / (1024**2):.2f} MB float32)"
  )

  # 4. Generate dummy mini-batch matching DataLoader format: [Batch=8, Channels=3, Height=128, Width=256]
  dummy_input = torch.randn(8, 3, 128, 256, device=device)
  print(f"\nForward pass with dummy tensor: {dummy_input.shape}")

  # 5. Run Forward Pass
  with torch.no_grad():
    output = model(dummy_input)

  print(f"Output Tensor Shape [B, C, H, W]: {output.shape}")
  assert output.shape == (
      8,
      1,
      128,
      256,
  ), "Output dimensions do not match expected shape!"

  if device.type == "cuda":
    vram_used = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"GPU VRAM Allocated during forward pass: {vram_used:.2f} MB")

  print(
      "\nModel successfully built and verified! Ready for loss function integration."
  )