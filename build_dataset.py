from datetime import datetime, timedelta
import glob
import os
import re
import numpy as np
import pandas as pd
from pyresample import create_area_def
from satpy import Scene

# =====================================================================
# 1. PATHS AND SPATIAL CONFIGURATION
# =====================================================================
RAW_DATA_DIR = r"D:\FINDFIRE_DATA"
EXCEL_FILE = "nasa data set.xlsx"
OUTPUT_DIR = r"D:\FINDFIRE_DATA\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Northern Algeria Bounding Box (Phase 3)
WIDTH = 250
HEIGHT = 120
LON_MIN, LON_MAX = -2.0, 8.5
LAT_MIN, LAT_MAX = 34.0, 37.5

algeria_area = create_area_def(
    area_id="algeria_north",
    projection={"proj": "latlong", "datum": "WGS84"},
    area_extent=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX],
    shape=(HEIGHT, WIDTH),
    units="degrees",
)

# =====================================================================
# 2. LOAD AND INDEX NASA GROUND TRUTH TARGETS
# =====================================================================
print(f"Loading NASA targets from: {EXCEL_FILE}...")
df_nasa = pd.read_excel(EXCEL_FILE)

# Construct a single UTC datetime column for every detection
time_str = df_nasa["acq_time"].astype(int).astype(str).str.zfill(4)
date_str = df_nasa["acq_date"].astype(str).str[:10]
df_nasa["dt"] = pd.to_datetime(
    date_str + " " + time_str, format="%Y-%m-%d %H%M"
)

# Keep only detections inside Northern Algeria
df_nasa = df_nasa[
    (df_nasa["latitude"] >= LAT_MIN)
    & (df_nasa["latitude"] <= LAT_MAX)
    & (df_nasa["longitude"] >= LON_MIN)
    & (df_nasa["longitude"] <= LON_MAX)
].copy()
print(
    f"Loaded {len(df_nasa):,} valid fire points within Northern Algeria"
    " bounds.\n"
)

# =====================================================================
# 3. DISCOVER ALL DOWNLOADED .NAT FILES
# =====================================================================
nat_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.nat")))
print(f"Found {len(nat_files)} satellite .nat files to process.")

if not nat_files:
  raise FileNotFoundError(
      f"No .nat files found in {RAW_DATA_DIR}. Check your folder path!"
  )


# Helper: Extract datetime from EUMETSAT filename
def parse_sat_filename(filename):
  basename = os.path.basename(filename)
  # Pattern matches: -YYYYMMDDHHMMSS.
  match = re.search(
      r"-(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.", basename
  )
  if match:
    y, m, d, h, mn, s = match.groups()
    return datetime(int(y), int(m), int(d), int(h), int(mn), int(s))
  return None


# =====================================================================
# 4. PROCESSING LOOP (TENSOR EXTRACTION & PAIRING)
# =====================================================================
dataset_records = []
total_files = len(nat_files)

print("Beginning automated extraction and spatial alignment...\n")

for idx, nat_path in enumerate(nat_files):
  sat_dt = parse_sat_filename(nat_path)
  if sat_dt is None:
    print(f"[{idx+1}/{total_files}] Skipping unparseable file: {nat_path}")
    continue

  sample_id = f"sample_{idx:03d}"
  output_npz_path = os.path.join(OUTPUT_DIR, f"{sample_id}.npz")

  # Skip if already processed
  if os.path.exists(output_npz_path):
    print(f"[{idx+1}/{total_files}] {sample_id} already exists. Skipping.")
    continue

  # 4A. Query matching NASA fire detections (+/- 15 minutes of satellite scan)
  window_start = sat_dt - timedelta(minutes=15)
  window_end = sat_dt + timedelta(minutes=15)
  matching_fires = df_nasa[
      (df_nasa["dt"] >= window_start) & (df_nasa["dt"] <= window_end)
  ]

  # 4B. Build the Ground Truth Binary Mask Y [120, 250]
  mask_y = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
  for _, fire in matching_fires.iterrows():
    lat = fire["latitude"]
    lon = fire["longitude"]

    # Mapping equations
    x = int(((lon - LON_MIN) / (LON_MAX - LON_MIN)) * WIDTH)
    y = int(((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * HEIGHT)

    # Clamp within array bounds
    x = min(max(x, 0), WIDTH - 1)
    y = min(max(y, 0), HEIGHT - 1)

    mask_y[y, x] = 1.0

  fire_pixel_count = int(np.sum(mask_y == 1.0))

  # 4C. Load and Resample Satellite Bands via Satpy
  try:
    scn = Scene(filenames=[nat_path], reader="seviri_l1b_native")
    scn.load(["IR_039", "IR_108"])
    local_scn = scn.resample(algeria_area, resampler="nearest")

    t_39 = local_scn["IR_039"].values.astype(np.float32)
    t_108 = local_scn["IR_108"].values.astype(np.float32)

    # Compute physical differential feature
    delta_t = (t_39 - t_108).astype(np.float32)

    # Replace boundary NaNs with 0.0 to prevent gradient corruption
    t_39 = np.nan_to_num(t_39, nan=0.0)
    t_108 = np.nan_to_num(t_108, nan=0.0)
    delta_t = np.nan_to_num(delta_t, nan=0.0)

    # Stack into 3-channel input tensor X [3, 120, 250]
    tensor_x = np.stack([t_39, t_108, delta_t], axis=0)

  except Exception as e:
    print(f"[{idx+1}/{total_files}] Failed to process {nat_path}: {e}")
    continue

  # 4D. Determine Split (Spatial-Temporal splitting by Year)
  if sat_dt.year in [2024, 2025]:
    split = "train"
  elif sat_dt.year == 2026 and sat_dt < datetime(2026, 7, 25):
    split = "val"
  else:
    split = "test"

  # 4E. Save Compressed Pair (.npz)
  np.savez_compressed(
      output_npz_path,
      image=tensor_x,  # Shape: [3, 120, 250]
      mask=mask_y,  # Shape: [120, 250]
      timestamp=sat_dt.isoformat(),
      fire_pixels=fire_pixel_count,
  )

  dataset_records.append({
      "sample_id": sample_id,
      "file_path": output_npz_path,
      "timestamp": sat_dt.isoformat(),
      "year": sat_dt.year,
      "fire_pixels": fire_pixel_count,
      "split": split,
  })

  print(
      f"[{idx+1}/{total_files}] Created {sample_id} ({sat_dt.strftime('%Y-%m-%d %H:%M')}) "
      f"-> Fires: {fire_pixel_count} px | Split: {split}"
  )

# =====================================================================
# 5. SAVE DATASET INDEX CSV
# =====================================================================
if dataset_records:
  index_df = pd.DataFrame(dataset_records)
  index_path = os.path.join(OUTPUT_DIR, "dataset_index.csv")
  index_df.to_csv(index_path, index=False)

  print("\n==========================================")
  print("DATASET GENERATION COMPLETE!")
  print(f"Total Samples Generated: {len(index_df)}")
  print("\nSplit Distribution:")
  print(index_df["split"].value_counts())
  print(f"\nIndex saved to: {index_path}")
  print(f"Processed .npz files saved to: {OUTPUT_DIR}")
  print("==========================================")