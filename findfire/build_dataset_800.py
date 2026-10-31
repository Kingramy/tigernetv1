from datetime import datetime, timedelta
import glob
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from pyresample.geometry import AreaDefinition
from satpy import Scene

# =====================================================================
# 1. PATHS & TEMPORARY SCRATCH DIRECTORY
# =====================================================================
RAW_DIR = r"D:\FINDFIRE_DATA\raw_800"
PROCESSED_DIR = r"D:\FINDFIRE_DATA\processed_800"
SCRATCH_DIR = r"D:\FINDFIRE_DATA\temp_unpack"
EXCEL_PATH = "nasa data set.xlsx"
INDEX_CSV = os.path.join(PROCESSED_DIR, "dataset_index_800.csv")

if not os.path.exists(EXCEL_PATH):
  EXCEL_PATH = os.path.join("..", "nasa data set.xlsx")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Northern Algeria Equidistant Grid (120 rows x 250 cols)
LAT_MIN, LAT_MAX = 34.0, 37.5
LON_MIN, LON_MAX = -2.0, 8.5
GRID_H, GRID_W = 120, 250

area_def = AreaDefinition(
    area_id="northern_algeria",
    description="Northern Algeria Target Domain",
    proj_id="latlong",
    projection={"proj": "latlong", "datum": "WGS84"},
    width=GRID_W,
    height=GRID_H,
    area_extent=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX],
)

# =====================================================================
# 2. LOAD NASA VIIRS GROUND TRUTH
# =====================================================================
print("==================================================")
print("BUILDING 800-SCENE NPZ DATASET")
print("==================================================")
print(f"Reading NASA Ground Truth: {EXCEL_PATH}")

df_nasa = pd.read_excel(EXCEL_PATH)

df_alg = df_nasa[
    (df_nasa["latitude"] >= LAT_MIN)
    & (df_nasa["latitude"] <= LAT_MAX)
    & (df_nasa["longitude"] >= LON_MIN)
    & (df_nasa["longitude"] <= LON_MAX)
].copy()

time_str = df_alg["acq_time"].astype(int).astype(str).str.zfill(4)
date_str = df_alg["acq_date"].astype(str).str[:10]
df_alg["timestamp"] = pd.to_datetime(
    date_str + " " + time_str, format="%Y-%m-%d %H%M"
)
print(f"NASA active fire records in Northern Algeria: {len(df_alg):,}")

# Clean up any leftover temp files
for leftover in glob.glob(os.path.join(RAW_DIR, "temp_*")):
  try:
    os.remove(leftover)
  except:
    pass

# =====================================================================
# 3. LOCATE DOWNLOADED RAW FILES
# =====================================================================
raw_files = sorted(
    glob.glob(os.path.join(RAW_DIR, "*.nat"))
    + glob.glob(os.path.join(RAW_DIR, "*.zip"))
)
print(f"Found {len(raw_files)} raw files to process in: {RAW_DIR}\n")

# =====================================================================
# 4. PROCESSING LOOP (EXTRACT -> RESAMPLE -> NPZ)
# =====================================================================
records = []
total_files = len(raw_files)

for idx, filepath in enumerate(raw_files, start=1):
  file_name = os.path.basename(filepath)
  sample_id = f"sample_800_{idx:04d}"
  out_npz = os.path.join(PROCESSED_DIR, f"{sample_id}.npz")

  # Skip if already converted
  if os.path.exists(out_npz):
    continue

  target_nat_path = None
  needs_cleanup = False

  try:
    # 1. Check if the file is a zip archive (regardless of file extension)
    if zipfile.is_zipfile(filepath):
      with zipfile.ZipFile(filepath, "r") as z:
        # Find the .nat file inside the archive
        nat_names = [
            n for n in z.namelist() if n.endswith(".nat") or "MSG" in n
        ]
        if len(nat_names) == 0:
          print(f"[{idx:03d}/{total_files}] No .nat found in {file_name}")
          continue

        # Extract strictly with the original EUMETSAT filename into SCRATCH_DIR
        extracted_name = nat_names[0]
        z.extract(extracted_name, SCRATCH_DIR)
        target_nat_path = os.path.join(SCRATCH_DIR, extracted_name)
        needs_cleanup = True
    else:
      target_nat_path = filepath

    # 2. Read with Satpy using the verified filename
    scn = Scene(filenames=[target_nat_path], reader="seviri_l1b_native")
    scn.load(["IR_039", "IR_108"])
    resampled = scn.resample(area_def)

    t_39 = resampled["IR_039"].values.astype(np.float32)
    t_108 = resampled["IR_108"].values.astype(np.float32)

    t_39 = np.nan_to_num(t_39, nan=0.0)
    t_108 = np.nan_to_num(t_108, nan=0.0)
    delta_t = t_39 - t_108

    image_3ch = np.stack([t_39, t_108, delta_t], axis=0)  # [3, 120, 250]
    scene_start_dt = scn.start_time

    # 3. Match NASA Ground Truth (15-minute window)
    win_start = scene_start_dt - timedelta(minutes=15)
    win_end = scene_start_dt + timedelta(minutes=15)

    matches = df_alg[
        (df_alg["timestamp"] >= win_start) & (df_alg["timestamp"] <= win_end)
    ]
    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for _, fire in matches.iterrows():
      row_px = int(
          (LAT_MAX - fire["latitude"]) / (LAT_MAX - LAT_MIN) * (GRID_H - 1)
      )
      col_px = int(
          (fire["longitude"] - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_W - 1)
      )
      row_px = max(0, min(GRID_H - 1, row_px))
      col_px = max(0, min(GRID_W - 1, col_px))
      mask[row_px, col_px] = 1.0

    fire_pixel_count = int(mask.sum())

    # 4. Save clean compressed .npz
    np.savez_compressed(out_npz, image=image_3ch, mask=mask)

    records.append({
        "sample_id": sample_id,
        "file_path": out_npz,
        "timestamp": scene_start_dt.isoformat(),
        "fire_pixels": fire_pixel_count,
        "hour": scene_start_dt.hour,
    })

    if idx % 20 == 0 or idx == total_files:
      print(
          f"[{idx:03d}/{total_files}] Processed {sample_id} | Time:"
          f" {scene_start_dt.strftime('%Y-%m-%d %H:%M')} | Fire Pixels:"
          f" {fire_pixel_count}"
      )

  except Exception as e:
    print(f"[{idx:03d}/{total_files}] Error processing {file_name}: {e}")

  finally:
    # 5. Instantly delete the temporary extracted .nat file to keep drive empty
    if needs_cleanup and target_nat_path and os.path.exists(target_nat_path):
      try:
        os.remove(target_nat_path)
      except:
        pass

# Clean up scratch folder
try:
  shutil.rmtree(SCRATCH_DIR)
except:
  pass

# =====================================================================
# 5. CHRONOLOGICAL SPLIT (600 TRAIN / REST TEST)
# =====================================================================
df_index = pd.DataFrame(records)

if len(df_index) > 0:
  df_index = df_index.sort_values(by="timestamp").reset_index(drop=True)

  # Assign roughly 530-600 to train, and remaining to test
  # Following your 600/rest split:
  split_point = min(600, int(len(df_index) * 0.75))
  df_index["split"] = "train"
  df_index.loc[split_point:, "split"] = "test"

  df_index.to_csv(INDEX_CSV, index=False)

  print("\n" + "=" * 60)
  print("DATASET PROCESSING COMPLETE")
  print("=" * 60)
  print(f"Total .npz files created : {len(df_index)}")
  print(f"Training split scenes    : {(df_index['split'] == 'train').sum()}")
  print(f"Test split scenes        : {(df_index['split'] == 'test').sum()}")
  print(f"Saved dataset index to   : {INDEX_CSV}")
  print("==================================================")