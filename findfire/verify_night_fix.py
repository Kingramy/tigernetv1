import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unet_model import UNet

INDEX_PATH = r"D:\FINDFIRE_DATA\processed_800\dataset_index_800.csv"
NIGHT_CKPT = os.path.join("checkpoints", "best_unet_night_800.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
night_state = torch.load(NIGHT_CKPT, map_location=device, weights_only=False)
model_night = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model_night.load_state_dict(night_state["model_state_dict"])
model_night.eval()

df_all = pd.read_csv(INDEX_PATH)
df_all["hour"] = pd.to_datetime(df_all["timestamp"]).dt.hour
df_night_test = df_all[
    (df_all["split"] == "test")
    & ((df_all["hour"] < 6) | (df_all["hour"] > 18))
].reset_index(drop=True)

print(f"Testing Physical Temperature Gating on {len(df_night_test)} Night Scenes...")

means = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
stds = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)

# Compare Raw vs Filtered
total_tp_raw, total_fp_raw = 0, 0
total_tp_filt, total_fp_filt = 0, 0
total_gt = 0
eps = 1e-6

with torch.no_grad():
  for idx, row in df_night_test.iterrows():
    data = np.load(row["file_path"])
    raw_img = data["image"]  # [3, 120, 250], raw_img[0] is T_3.9 in Kelvin
    target = data["mask"]
    total_gt += target.sum()

    img_tensor = torch.from_numpy(raw_img).float()
    norm_img = (img_tensor - means) / stds
    padded = F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)

    logits = model_night(padded)
    probs = torch.sigmoid(logits)[0, 0, 4:124, 3:253].cpu().numpy()

    # Raw model prediction at tau=0.85
    pred_raw = (probs >= 0.85).astype(np.float32)

    # Physical filter: Must have confidence >= 0.85 AND absolute T_3.9 >= 285 K (12°C)
    t_39_kelvin = raw_img[0]
    pred_filtered = pred_raw * (t_39_kelvin >= 285.0).astype(np.float32)

    # Accumulate raw
    total_tp_raw += np.sum((pred_raw == 1.0) & (target == 1.0))
    total_fp_raw += np.sum((pred_raw == 1.0) & (target == 0.0))

    # Accumulate filtered
    total_tp_filt += np.sum((pred_filtered == 1.0) & (target == 1.0))
    total_fp_filt += np.sum((pred_filtered == 1.0) & (target == 0.0))

print("\n" + "=" * 65)
print("NIGHTTIME PHYSICAL CONSISTENCY FILTER COMPARISON")
print("=" * 65)
print(f"Total True NASA Ground Truth Fires: {int(total_gt)}")
print("-" * 65)
print(f"RAW UNET_NIGHT (No Filter):")
print(f"  • True Positives  : {int(total_tp_raw)} ({total_tp_raw/total_gt*100:.2f}% recall)")
print(f"  • False Positives : {int(total_fp_raw):,} pixels ({total_fp_raw/len(df_night_test):.1f} px/scene)")
print(f"  • Dice Score      : {(2*total_tp_raw)/(2*total_tp_raw + total_fp_raw + (total_gt-total_tp_raw)):.4f}")
print("-" * 65)
print(f"FILTERED (T_3.9 >= 285.0 K / 12°C):")
print(f"  • True Positives  : {int(total_tp_filt)} ({total_tp_filt/total_gt*100:.2f}% recall)")
print(f"  • False Positives : {int(total_fp_filt):,} pixels ({total_fp_filt/len(df_night_test):.1f} px/scene)")
print(f"  • Dice Score      : {(2*total_tp_filt)/(2*total_tp_filt + total_fp_filt + (total_gt-total_tp_filt)):.4f}")
print("=" * 65)