import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label
import torch
import torch.nn.functional as F

from unet_model import UNet

# =====================================================================
# 1. SETUP & MODEL LOADING
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed_800\dataset_index_800.csv"
NIGHT_CKPT = os.path.join("checkpoints", "best_unet_night_800.pth")
DAY_CKPT = os.path.join("checkpoints", "best_unet_day_800.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==================================================")
print(f"PRODUCTION DUAL ENGINE EVALUATION ON: {device}")
print("==================================================")

# Load UNet_Night (3 channels)
night_state = torch.load(NIGHT_CKPT, map_location=device, weights_only=False)
model_night = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model_night.load_state_dict(night_state["model_state_dict"])
model_night.eval()

# Load UNet_Day (4 channels)
day_state = torch.load(DAY_CKPT, map_location=device, weights_only=False)
model_day = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model_day.load_state_dict(day_state["model_state_dict"])
model_day.eval()

print("Loaded UNet_Night and UNet_Day checkpoints successfully.\n")

# Grid bounds
LAT_MIN, LAT_MAX = 34.0, 37.5
LON_MIN, LON_MAX = -2.0, 8.5
GRID_H, GRID_W = 120, 250

# Precompute Static Land Mask (Latitude <= 36.9°N represents the African landmass)
lat_per_row = np.linspace(LAT_MAX, LAT_MIN, GRID_H)
LAND_MASK = np.repeat((lat_per_row <= 36.9)[:, None], GRID_W, axis=1)

# Normalization constants
NIGHT_MEANS = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
NIGHT_STDS = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)

DAY_MEANS = torch.tensor([315.0, 300.0, 15.0, 0.0]).view(4, 1, 1)
DAY_STDS = torch.tensor([15.0, 12.0, 8.0, 4.0]).view(4, 1, 1)


# =====================================================================
# 2. UNIFIED PRODUCTION INFERENCE FUNCTION
# =====================================================================
def run_production_inference(raw_3ch, dt_timestamp):
  hour = dt_timestamp.hour
  is_night = hour < 6 or hour > 18
  regime = "NIGHT" if is_night else "DAY"

  img_tensor = torch.from_numpy(raw_3ch).float()

  if is_night:
    norm_img = (img_tensor - NIGHT_MEANS) / NIGHT_STDS
    padded = F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)

    with torch.no_grad():
      logits = model_night(padded)
      probs = torch.sigmoid(logits)[0, 0, 4:124, 3:253].cpu().numpy()

    # Night rules: tau=0.95, T3.9 >= 285 K, must be on land
    tau = 0.95
    t_39 = raw_3ch[0]
    binary_pred = (probs >= tau) & (t_39 >= 285.0) & LAND_MASK

  else:
    # Day rules: 11x11 contrast, tau=0.90, must be on land
    delta_t = img_tensor[2:3, :, :].unsqueeze(0)
    local_bg = F.avg_pool2d(delta_t, kernel_size=11, stride=1, padding=5).squeeze(0)
    contrast = img_tensor[2:3, :, :] - local_bg
    img_4ch = torch.cat([img_tensor, contrast], dim=0)

    norm_img = (img_4ch - DAY_MEANS) / DAY_STDS
    padded = F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)

    with torch.no_grad():
      logits = model_day(padded)
      probs = torch.sigmoid(logits)[0, 0, 4:124, 3:253].cpu().numpy()

    tau = 0.90
    binary_pred = (probs >= tau) & LAND_MASK

  return binary_pred.astype(np.float32), probs, regime, tau


# =====================================================================
# 3. BENCHMARK ON ALL 170 UNSEEN TEST SCENES
# =====================================================================
df_all = pd.read_csv(INDEX_PATH)
df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
print(f"Evaluating Unified Engine on {len(df_test)} Untouched Test Scenes...")

records = []
eps = 1e-6

for idx, row in df_test.iterrows():
  data = np.load(row["file_path"])
  raw_image = data["image"]
  target = data["mask"] * LAND_MASK  # Ground truth evaluated on land
  dt = pd.to_datetime(row["timestamp"])

  pred, probs, regime, tau = run_production_inference(raw_image, dt)

  tp = np.sum((pred == 1.0) & (target == 1.0))
  fp = np.sum((pred == 1.0) & (target == 0.0))
  fn = np.sum((pred == 0.0) & (target == 1.0))

  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
  recall = (tp + eps) / (tp + fn + eps)
  precision = (tp + eps) / (tp + fp + eps)

  records.append({
      "sample_id": row["sample_id"],
      "timestamp": str(dt),
      "regime": regime,
      "gt_fires": int(target.sum()),
      "tp": int(tp),
      "fp": int(fp),
      "fn": int(fn),
      "dice": dice,
      "recall": recall,
      "precision": precision,
  })

df_bench = pd.DataFrame(records)

# =====================================================================
# 4. FINAL PRODUCTION REPORT
# =====================================================================
summary = (
    df_bench.groupby("regime")
    .agg(
        scenes=("sample_id", "count"),
        total_gt_fires=("gt_fires", "sum"),
        total_tp=("tp", "sum"),
        total_fp=("fp", "sum"),
        mean_dice=("dice", "mean"),
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_fp_per_scene=("fp", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 90)
print("FINAL PRODUCTION PERFORMANCE ACROSS 170 UNTOUCHED TEST SCENES")
print("=" * 90)
print(summary.to_string(index=False))
print("=" * 90)

# =====================================================================
# 5. REAL INCIDENT EXTRACTION
# =====================================================================
day_fires = df_bench[(df_bench["regime"] == "DAY") & (df_bench["gt_fires"] >= 10)]
sample_idx = day_fires.index[0] if len(day_fires) > 0 else 0

sample_row = df_test.iloc[sample_idx]
sample_data = np.load(sample_row["file_path"])
sample_img = sample_data["image"]
sample_gt = sample_data["mask"] * LAND_MASK
sample_dt = pd.to_datetime(sample_row["timestamp"])

pred, probs, regime, tau = run_production_inference(sample_img, sample_dt)
labeled_arr, num_blobs = label(pred)

print(f"\nOperational Incident Report for Test Scene: {sample_row['sample_id']} ({sample_dt})")
print(f"Regime: {regime} | Fire Fronts Identified: {num_blobs}")

for c_id in range(1, min(num_blobs + 1, 6)):
  ys, xs = np.where(labeled_arr == c_id)
  lat = LAT_MAX - (np.mean(ys) / GRID_H) * (LAT_MAX - LAT_MIN)
  lon = LON_MIN + (np.mean(xs) / GRID_W) * (LON_MAX - LON_MIN)
  area = len(ys) * 9
  print(f"  Front #{c_id}: Lat {lat:.4f}°N, Lon {lon:.4f}°E | Estimated Area: ~{area} km²")

# Render Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(sample_img[2] * LAND_MASK, cmap="inferno", origin="upper")
axes[0].set_title(f"Input: Delta T [{sample_dt.strftime('%Y-%m-%d %H:%M')}]")
fig.colorbar(im0, ax=axes[0], label="Kelvin")

im1 = axes[1].imshow(sample_gt, cmap="gray", origin="upper")
axes[1].set_title("NASA Ground Truth (VIIRS)")
fig.colorbar(im1, ax=axes[1])

axes[2].imshow(sample_img[2] * LAND_MASK, cmap="inferno", origin="upper")
f_y, f_x = np.where(pred == 1.0)
axes[2].scatter(f_x, f_y, color="cyan", s=25, label=f"Detections (τ={tau})")
axes[2].set_title(f"Operational Alerts ({num_blobs} Fire Fronts)")
axes[2].legend(loc="lower left")

for ax in axes:
  ax.set_xlabel("Longitude Index")
  ax.set_ylabel("Latitude Index")

plt.tight_layout()
plt.savefig("final_production_report_800.png", dpi=200)
print("\nSaved final report visualization to 'final_production_report_800.png'!")