import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unet_model import UNet
from wildfire_day_dataset import WildfireDayDataset

# =====================================================================
# 1. SETUP & MODEL LOADING
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_unet_day.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CHECKPOINT_PATH):
  raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

print("==================================================")
print(f"Evaluating UNet_Day Checkpoint on: {device}")
print("==================================================")

# Load checkpoint (with weights_only=False for PyTorch 2.6)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

# Initialize 4-channel U-Net
model = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded Daytime Model from Epoch {checkpoint['epoch']}")

# Load test dataset
test_dataset = WildfireDayDataset(INDEX_PATH, split="test", augment=False)
print(f"Total Untouched Daytime Test Scenes: {len(test_dataset)}\n")

# =====================================================================
# 2. RUN INFERENCE & ACCUMULATE METRICS
# =====================================================================
TAU = 0.50  # Decision threshold
eps = 1e-6

scene_records = []
total_tp = 0
total_fp = 0
total_fn = 0

with torch.no_grad():
  for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    record = test_dataset.records.iloc[idx]

    image_tensor = image.unsqueeze(0).to(device)  # [1, 4, 128, 256]
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)

    # Crop symmetric padding back to native Northern Algeria grid (120 x 250)
    # Padding applied was: (left=3, right=3, top=4, bottom=4)
    probs_cropped = probs[0, 0, 4:124, 3:253].cpu()
    target_cropped = mask[0, 4:124, 3:253].cpu()

    # Binarize
    preds = (probs_cropped >= TAU).float()

    tp = torch.sum(preds * target_cropped).item()
    fp = torch.sum(preds * (1.0 - target_cropped)).item()
    fn = torch.sum((1.0 - preds) * target_cropped).item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    scene_records.append({
        "sample_id": record["sample_id"],
        "timestamp": record["timestamp"],
        "gt_fires": int(target_cropped.sum().item()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "dice": dice,
        "recall": recall,
        "precision": precision,
    })

df_results = pd.DataFrame(scene_records)

# =====================================================================
# 3. PRINT COMPARISON SUMMARY
# =====================================================================
mean_dice = df_results["dice"].mean()
mean_recall = df_results["recall"].mean()
mean_precision = df_results["precision"].mean()
mean_fp_per_scene = df_results["fp"].mean()

print("=" * 60)
print("UNet_Day PERFORMANCE ON UNSEEN DAYTIME TEST SCENES")
print("=" * 60)
print(f"Total Scenes Evaluated : {len(df_results)}")
print(f"Total True Positives   : {int(total_tp):,}")
print(f"Total False Positives  : {int(total_fp):,}")
print(f"Total Missed Fires     : {int(total_fn):,}")
print(f"Mean Dice Score        : {mean_dice:.4f} ({mean_dice*100:.2f}%)")
print(f"Mean Recall            : {mean_recall:.4f} ({mean_recall*100:.2f}%)")
print(
    f"Mean Precision         : {mean_precision:.4f}"
    f" ({mean_precision*100:.2f}%)"
)
print(f"Mean FP per Scene      : {mean_fp_per_scene:.1f} pixels")
print("=" * 60)

# Compare directly against the old model's blowout
old_day_fp = 1389499
curr_day_fp = int(total_fp)
reduction = (old_day_fp - curr_day_fp) / old_day_fp * 100.0

print(f"\n--- DIRECT COMPARISON: DAYTIME FALSE ALARM SUPPRESSION ---")
print(f"Old Model Total False Positives : {old_day_fp:,} pixels")
print(f"New Model Total False Positives : {curr_day_fp:,} pixels")
print(f"False Alarm Reduction           : {reduction:.2f}%\n")