from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unet_model import UNet
from wildfire_dataset import WildfireDataset

# =====================================================================
# 1. SETUP & MODEL LOADING
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_diurnal_unet.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CHECKPOINT_PATH):
  raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

print("==================================================")
print(f"Diagnosing 4-Channel Diurnal Model on: {device}")
print("==================================================")

# Load checkpoint (with weights_only=False for compatibility)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

# Initialize 4-channel U-Net
model = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from Epoch {checkpoint['epoch']}")

# Load the test dataset
test_dataset = WildfireDataset(INDEX_PATH, split="test", augment=False)
index_df = pd.read_csv(INDEX_PATH)
test_records = index_df[index_df["split"] == "test"].reset_index(drop=True)

print(f"Total test scenes to evaluate: {len(test_dataset)}\n")

# =====================================================================
# 2. RUN INFERENCE & LOG METRICS PER REGIME
# =====================================================================
TAU = 0.50  # Standard decision threshold
eps = 1e-6
results = []

print("Running evaluation across all test scenes...")

with torch.no_grad():
  for idx in range(len(test_dataset)):
    # The updated dataset returns: image_4ch [4, 128, 256], mask [1, 128, 256], solar_val
    image_4ch, mask, solar_val = test_dataset[idx]
    record = test_records.iloc[idx]
    timestamp = pd.to_datetime(record["timestamp"])
    hour = timestamp.hour

    # Day vs Night classification
    is_day = 10 <= hour <= 15
    regime = "DAY" if is_day else "NIGHT"

    # Forward pass
    image_tensor = image_4ch.unsqueeze(0).to(device)  # [1, 4, 128, 256]
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)

    # Crop padding back to native Northern Algeria grid (120 x 250)
    # Padding: (left=3, right=3, top=4, bottom=4)
    probs_cropped = probs[0, 0, 4:124, 3:253].cpu()
    target_cropped = mask[0, 4:124, 3:253].cpu()

    # Binarize predictions
    preds = (probs_cropped >= TAU).float()

    tp = torch.sum(preds * target_cropped).item()
    fp = torch.sum(preds * (1.0 - target_cropped)).item()
    fn = torch.sum((1.0 - preds) * target_cropped).item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    results.append({
        "sample_id": record["sample_id"],
        "timestamp": str(timestamp),
        "hour": hour,
        "solar_val": solar_val,
        "regime": regime,
        "gt_fires": int(target_cropped.sum().item()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "dice": dice,
        "recall": recall,
        "precision": precision,
    })

# Convert to DataFrame
df_eval = pd.DataFrame(results)

# =====================================================================
# 3. COMPARE DAY VS NIGHT REGIMES
# =====================================================================
summary = (
    df_eval.groupby("regime")
    .agg(
        scenes=("sample_id", "count"),
        total_gt_fires=("gt_fires", "sum"),
        total_tp=("tp", "sum"),
        total_fp=("fp", "sum"),
        mean_dice=("dice", "mean"),
        mean_recall=("recall", "mean"),
        mean_fp_per_scene=("fp", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 78)
print("DIURNAL MODEL PERFORMANCE: DAY (Sunlit) vs. NIGHT (Pure Thermal)")
print("=" * 78)
print(summary.to_string(index=False))
print("=" * 78)

# Calculate reduction in daytime false alarms compared to previous 3-channel model
prev_day_fp = 1389499
curr_day_fp = summary.loc[summary["regime"] == "DAY", "total_fp"].values[0]
reduction = (prev_day_fp - curr_day_fp) / prev_day_fp * 100.0

print(f"\n--- COMPARISON ON DAYTIME FALSE ALARMS ---")
print(f"Old 3-Channel Daytime False Positives : {prev_day_fp:,} pixels")
print(f"New 4-Channel Daytime False Positives : {curr_day_fp:,} pixels")
print(f"False Alarm Reduction                 : {reduction:.2f}%")

df_eval.to_csv("diurnal_model_diagnosis.csv", index=False)
print(
    "\nFull per-scene evaluation table saved to 'diurnal_model_diagnosis.csv'."
)