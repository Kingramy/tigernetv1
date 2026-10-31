from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class WildfireDataset(Dataset):

  def __init__(self, index_csv_path, split="train", augment=False):
    self.split = split
    self.augment = augment

    df = pd.read_csv(index_csv_path)
    self.records = df[df["split"] == split].reset_index(drop=True)

    if len(self.records) == 0:
      raise RuntimeError(
          f"No samples found for split '{split}' in {index_csv_path}"
      )

    # Normalization statistics for the 3 physical channels (in Kelvin)
    self.channel_means = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
    self.channel_stds = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)

  def __len__(self):
    return len(self.records)

  def __getitem__(self, idx):
    row = self.records.iloc[idx]
    file_path = row["file_path"]
    data = np.load(file_path)

    # 1. Load satellite bands and mask
    image_np = data["image"]  # [3, 120, 250]
    mask_np = data["mask"]  # [120, 250]

    image = torch.from_numpy(image_np).float()
    mask = torch.from_numpy(mask_np).float().unsqueeze(0)

    # 2. Standardize physical channels
    image = (image - self.channel_means) / self.channel_stds

    # 3. Compute Solar Cycle Channel S(t) ∈ [0.0, 1.0]
    dt = datetime.fromisoformat(row["timestamp"])
    hour_float = dt.hour + dt.minute / 60.0
    solar_val = (1.0 + np.cos(2.0 * np.pi * (hour_float - 12.0) / 24.0)) / 2.0
    solar_channel = torch.full(
        (1, 120, 250), solar_val, dtype=torch.float32
    )  # [1, 120, 250]

    # 4. Concatenate to form 4-channel tensor: [4, 120, 250]
    image_4ch = torch.cat([image, solar_channel], dim=0)

    # 5. Symmetric Spatial Padding to (128, 256)
    image_4ch = F.pad(image_4ch, (3, 3, 4, 4), mode="reflect")
    mask = F.pad(mask, (3, 3, 4, 4), mode="constant", value=0.0)

    # 6. Data Augmentation (Train only)
    if self.augment:
      if torch.rand(1).item() > 0.5:
        image_4ch = torch.flip(image_4ch, dims=[2])
        mask = torch.flip(mask, dims=[2])
      if torch.rand(1).item() > 0.5:
        image_4ch = torch.flip(image_4ch, dims=[1])
        mask = torch.flip(mask, dims=[1])

    # Also return the scalar solar_val so the loss function can adapt dynamically
    return image_4ch, mask, float(solar_val)