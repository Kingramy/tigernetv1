from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class WildfireDayDataset(Dataset):

  def __init__(self, index_csv_path, split="train", augment=False):
    self.split = split
    self.augment = augment

    # 1. Load index and isolate DAYTIME scenes only (07:00 to 17:00 UTC)
    df = pd.read_csv(index_csv_path)
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    day_df = df[(df["hour"] >= 7) & (df["hour"] <= 17)].copy()

    # Filter by requested split
    self.records = day_df[day_df["split"] == split].reset_index(drop=True)

    if len(self.records) == 0:
      raise RuntimeError(
          f"No daytime scenes found for split '{split}' in {index_csv_path}"
      )

    # Radiometric statistics for daytime Algerian conditions (in Kelvin)
    # Channel 0: T_3.9  (Mean: 315 K, Std: 15 K)
    # Channel 1: T_10.8 (Mean: 300 K, Std: 12 K)
    # Channel 2: Delta T (Mean: 15 K, Std: 8 K)
    # Channel 3: Local Contrast (Mean: 0 K, Std: 4 K)
    self.means = torch.tensor([315.0, 300.0, 15.0, 0.0]).view(4, 1, 1)
    self.stds = torch.tensor([15.0, 12.0, 8.0, 4.0]).view(4, 1, 1)

  def __len__(self):
    return len(self.records)

  def __getitem__(self, idx):
    row = self.records.iloc[idx]
    data = np.load(row["file_path"])

    # Load raw image [3, 120, 250] and binary mask [120, 250]
    image_np = data["image"].astype(np.float32)
    mask_np = data["mask"].astype(np.float32)

    image = torch.from_numpy(image_np)
    mask = torch.from_numpy(mask_np).unsqueeze(0)

    # -------------------------------------------------------------
    # 2. COMPUTE LOCAL BACKGROUND CONTRAST (Channel 3)
    # -------------------------------------------------------------
    # delta_t is Channel 2 of image
    delta_t = image[2:3, :, :]  # Shape: [1, 120, 250]

    # Add batch dim for pooling: [1, 1, 120, 250]
    delta_t_4d = delta_t.unsqueeze(0)

    # Compute 11x11 moving average using Average Pooling
    # kernel_size=11, stride=1, padding=5 preserves spatial dimensions (120, 250)
    local_bg_mean = F.avg_pool2d(
        delta_t_4d, kernel_size=11, stride=1, padding=5
    ).squeeze(0)

    # Contrast = Delta_T - Local_Background_Mean
    contrast_channel = delta_t - local_bg_mean  # Shape: [1, 120, 250]

    # -------------------------------------------------------------
    # 3. STACK INTO 4-CHANNEL INPUT TENSOR
    # -------------------------------------------------------------
    # Ch 0: T_3.9, Ch 1: T_10.8, Ch 2: Delta_T, Ch 3: Contrast
    image_4ch = torch.cat([image, contrast_channel], dim=0)  # [4, 120, 250]

    # Standardize
    image_4ch = (image_4ch - self.means) / self.stds

    # Symmetric spatial padding to (128, 256) for U-Net divisibility
    # pad: (left=3, right=3, top=4, bottom=4)
    image_padded = F.pad(image_4ch, (3, 3, 4, 4), mode="reflect")
    mask_padded = F.pad(mask, (3, 3, 4, 4), mode="constant", value=0.0)

    # Training Data Augmentation
    if self.augment:
      if torch.rand(1).item() > 0.5:
        image_padded = torch.flip(image_padded, dims=[2])
        mask_padded = torch.flip(mask_padded, dims=[2])
      if torch.rand(1).item() > 0.5:
        image_padded = torch.flip(image_padded, dims=[1])
        mask_padded = torch.flip(mask_padded, dims=[1])

    return image_padded, mask_padded