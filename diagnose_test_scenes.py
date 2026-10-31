import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from unet_model import UNet
from wildfire_dataset import WildfireDataset

# 1. Setup & Load Model Checkpoint
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_wildfire_unet.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH, map_location=device, weights_only=False
)
model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Load Test Records directly from index
index_df = pd.read_csv(INDEX_PATH)
test_records = index_df[index_df["split"] == "test"].reset_index(drop=True)
test_dataset = WildfireDataset(INDEX_PATH, split="test", augment=False)

print(f"Loaded {len(test_dataset)} test scenes.")

# 3. Evaluate Scene by Scene
scene_results = []
eps = 1e-6
TAU = 0.50  # Standard detection threshold

print("\nEvaluating each test scene individually...")
with torch.no_grad():
  for idx in range(len(test_dataset)):
    img, mask = test_dataset[idx]
    record = test_records.iloc[idx]
    timestamp = pd.to_datetime(record["timestamp"])
    hour = timestamp.hour

    # Tag scene as Day or Night based on UTC hour
    is_day = 10 <= hour <= 15
    day_night_label = "DAY" if is_day else "NIGHT"

    # Forward pass
    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)

    # Crop padding back to native grid (120 x 250)
    probs = probs[0, 0, 4:124, 3:253].cpu()
    target = mask[0, 4:124, 3:253].cpu()

    # Binarize
    pred = (probs >= TAU).float()

    tp = torch.sum(pred * target).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    scene_results.append({
        "sample_id": record["sample_id"],
        "timestamp": str(timestamp),
        "hour": hour,
        "regime": day_night_label,
        "ground_truth_fires": int(target.sum().item()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "dice": dice,
        "recall": recall,
        "precision": precision,
    })

df_diag = pd.DataFrame(scene_results)

# 4. Aggregate by Day vs. Night Regime
summary = (
    df_diag.groupby("regime")
    .agg(
        scenes=("sample_id", "count"),
        total_gt_fires=("ground_truth_fires", "sum"),
        total_tp=("tp", "sum"),
        total_fp=("fp", "sum"),
        mean_dice=("dice", "mean"),
        mean_recall=("recall", "mean"),
        mean_fp_per_scene=("fp", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("TEST SET BREAKDOWN: DAY (Sunlit) vs. NIGHT (Pure Thermal)")
print("=" * 70)
print(summary.to_string(index=False))
print("=" * 70)

# 5. Save detailed CSV
df_diag.to_csv("test_scenes_diagnosis.csv", index=False)
print("\nDetailed per-scene diagnostics saved to 'test_scenes_diagnosis.csv'.")