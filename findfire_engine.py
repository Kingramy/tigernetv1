from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label
import torch
import torch.nn.functional as F

from unet_model import UNet

# =====================================================================
# 1. PATHS & INITIALIZATION
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
NIGHT_CHECKPOINT = os.path.join("checkpoints", "best_wildfire_unet.pth")
DAY_CHECKPOINT = os.path.join("checkpoints", "best_unet_day.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==================================================")
print(f"FINDFIRE DUAL-REGIME ENGINE INITIALIZING ON: {device}")
print("==================================================")

# Load Night Model (3 channels)
print("Loading Nighttime U-Net (3 Channels)...")
night_ckpt = torch.load(
    NIGHT_CHECKPOINT, map_location=device, weights_only=False
)
model_night = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model_night.load_state_dict(night_ckpt["model_state_dict"])
model_night.eval()

# Load Day Model (4 channels)
print("Loading Daytime U-Net (4 Channels)...")
day_ckpt = torch.load(DAY_CHECKPOINT, map_location=device, weights_only=False)
model_day = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model_day.load_state_dict(day_ckpt["model_state_dict"])
model_day.eval()

print("Both models loaded successfully!\n")

# Spatial Bounds for Northern Algeria (from Phase 3)
LON_MIN, LON_MAX = -2.0, 8.5
LAT_MIN, LAT_MAX = 34.0, 37.5
GRID_H, GRID_W = 120, 250


# =====================================================================
# 2. INFERENCE CORE FUNCTION
# =====================================================================
def detect_wildfires(raw_image_3ch, timestamp_dt):
  """Routes an incoming satellite frame to the appropriate model based on solar time.

  Args:
      raw_image_3ch: NumPy array [3, 120, 250] in Kelvin (T3.9, T10.8, Delta_T)
      timestamp_dt: datetime object (UTC)

  Returns:
      binary_mask: [120, 250] binary prediction (1.0 = Fire, 0.0 = Safe)
      prob_map: [120, 250] confidence probabilities [0.0 to 1.0]
      regime: "NIGHT" or "DAY"
  """
  hour = timestamp_dt.hour
  # Night regime: UTC < 6 or UTC > 18
  is_night = hour < 6 or hour > 18
  regime = "NIGHT" if is_night else "DAY"

  image_tensor = torch.from_numpy(raw_image_3ch).float()

  if is_night:
    # ---------------- NIGHT REGIME ----------------
    # 3 Channels, standard normalization
    means = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
    stds = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)
    norm_img = (image_tensor - means) / stds

    # Symmetric pad: (left=3, right=3, top=4, bottom=4) -> [1, 3, 128, 256]
    padded = F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(
        device
    )

    with torch.no_grad():
      logits = model_night(padded)
      probs = torch.sigmoid(logits)

    probs_cropped = probs[0, 0, 4:124, 3:253].cpu().numpy()
    tau = 0.85  # Calibrated night threshold

  else:
    # ---------------- DAY REGIME ----------------
    # Compute 11x11 Local Background Contrast Channel
    delta_t = image_tensor[2:3, :, :].unsqueeze(0)  # [1, 1, 120, 250]
    local_bg = F.avg_pool2d(
        delta_t, kernel_size=11, stride=1, padding=5
    ).squeeze(0)
    contrast = image_tensor[2:3, :, :] - local_bg  # [1, 120, 250]

    image_4ch = torch.cat([image_tensor, contrast], dim=0)  # [4, 120, 250]

    # Day normalization
    means = torch.tensor([315.0, 300.0, 15.0, 0.0]).view(4, 1, 1)
    stds = torch.tensor([15.0, 12.0, 8.0, 4.0]).view(4, 1, 1)
    norm_img = (image_4ch - means) / stds

    padded = F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(
        device
    )

    with torch.no_grad():
      logits = model_day(padded)
      probs = torch.sigmoid(logits)

    probs_cropped = probs[0, 0, 4:124, 3:253].cpu().numpy()
    tau = 0.92  # Calibrated daytime threshold

  binary_mask = (probs_cropped >= tau).astype(np.float32)
  return binary_mask, probs_cropped, regime, tau


# =====================================================================
# 3. RUN FULL BENCHMARK ON UNTOUCHED TEST SCENES
# =====================================================================
index_df = pd.read_csv(INDEX_PATH)
test_records = index_df[index_df["split"] == "test"].reset_index(drop=True)

print(
    f"Running Unified Dual-Model Engine across all {len(test_records)} test"
    " scenes..."
)

test_stats = []
eps = 1e-6

for idx, row in test_records.iterrows():
  data = np.load(row["file_path"])
  raw_image = data["image"]  # [3, 120, 250]
  gt_mask = data["mask"]  # [120, 250]
  dt = pd.to_datetime(row["timestamp"])

  pred_mask, prob_map, regime, tau = detect_wildfires(raw_image, dt)

  tp = np.sum((pred_mask == 1.0) & (gt_mask == 1.0))
  fp = np.sum((pred_mask == 1.0) & (gt_mask == 0.0))
  fn = np.sum((pred_mask == 0.0) & (gt_mask == 1.0))

  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
  recall = (tp + eps) / (tp + fn + eps)
  precision = (tp + eps) / (tp + fp + eps)

  test_stats.append({
      "sample_id": row["sample_id"],
      "timestamp": str(dt),
      "regime": regime,
      "gt_fires": int(gt_mask.sum()),
      "tp": int(tp),
      "fp": int(fp),
      "fn": int(fn),
      "dice": dice,
      "recall": recall,
      "precision": precision,
  })

df_unified = pd.DataFrame(test_stats)

# Grouped Summary
summary = (
    df_unified.groupby("regime")
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

print("\n" + "=" * 80)
print("FINAL UNIFIED DUAL-ENGINE RESULTS ACROSS TEST DATASET")
print("=" * 80)
print(summary.to_string(index=False))
print("=" * 80)

# =====================================================================
# 4. VISUALIZE A REAL OPERATIONAL WILDFIRE DETECTION CASE
# =====================================================================
# Pick an intense daytime wildfire event from late August 2026
fire_samples = df_unified[
    (df_unified["regime"] == "DAY") & (df_unified["gt_fires"] > 30)
]
if len(fire_samples) > 0:
  sample_row = fire_samples.iloc[0]
  sample_idx = sample_row.name
else:
  sample_row = df_unified.iloc[0]
  sample_idx = 0

sample_data = np.load(test_records.iloc[sample_idx]["file_path"])
raw_img = sample_data["image"]
target_gt = sample_data["mask"]
sample_dt = pd.to_datetime(test_records.iloc[sample_idx]["timestamp"])

pred_mask, probs, regime, tau_used = detect_wildfires(raw_img, sample_dt)

# Find cluster coordinates (GPS conversion)
labeled_array, num_features = label(pred_mask)
detected_hotspots = []

for cluster_id in range(1, num_features + 1):
  y_coords, x_coords = np.where(labeled_array == cluster_id)
  # Calculate center of mass in grid pixels
  center_y = np.mean(y_coords)
  center_x = np.mean(x_coords)

  # Convert grid pixel to GPS (Lat/Lon)
  lat = LAT_MAX - (center_y / GRID_H) * (LAT_MAX - LAT_MIN)
  lon = LON_MIN + (center_x / GRID_W) * (LON_MAX - LON_MIN)
  cluster_size = len(y_coords)
  detected_hotspots.append((lat, lon, cluster_size))

print(
    f"\nOperational Incident Report for Scene {test_records.iloc[sample_idx]['sample_id']} ({sample_dt}):"
)
print(f"Regime Detected: {regime} | Threshold Applied: {tau_used}")
print(f"Active Wildfire Clusters Found: {num_features}")
for i, (lat, lon, size) in enumerate(detected_hotspots[:5], start=1):
  print(
      f"  Hotspot #{i}: Lat {lat:.4f}°N, Lon {lon:.4f}°E (Area: ~{size * 9} km²)"
  )

# Plot Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Physical Infrared Channel (Delta T)
im0 = axes[0].imshow(raw_img[2], cmap="inferno", origin="upper")
axes[0].set_title(
    f"Input: Delta T (T_3.9 - T_10.8) [{sample_dt.strftime('%Y-%m-%d %H:%M')}]"
)
fig.colorbar(im0, ax=axes[0], label="Kelvin")

# 2. Ground Truth Target (NASA VIIRS)
im1 = axes[1].imshow(target_gt, cmap="gray", origin="upper")
axes[1].set_title(f"Target Ground Truth (NASA VIIRS Fire Pixels)")
fig.colorbar(im1, ax=axes[1])

# 3. Model Alert Detection Map
axes[2].imshow(raw_img[2], cmap="inferno", origin="upper")
# Overlay detected fires
fire_y, fire_x = np.where(pred_mask == 1.0)
axes[2].scatter(
    fire_x,
    fire_y,
    color="cyan",
    s=25,
    marker="o",
    label=f"Unified Alert (τ={tau_used})",
)
axes[2].set_title(f"Operational Detections (Found {num_features} Fire Fronts)")
axes[2].legend(loc="lower left")

for ax in axes:
  ax.set_xlabel("Longitude Column")
  ax.set_ylabel("Latitude Row")

plt.tight_layout()
plt.savefig("operational_incident_report.png", dpi=200)
print(
    "\nSaved operational visual report to 'operational_incident_report.png'!"
)
plt.show()