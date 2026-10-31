import os
import shutil
import time
from datetime import datetime, timedelta
import eumdac
import pandas as pd

# =====================================================================
# 1. CREDENTIALS & DIRECTORY CONFIGURATION
# =====================================================================
# Enter your EUMETSAT API credentials (same as in eumetsat_downloader.py)

CONSUMER_KEY = "EhZhSU9MOW6R_hZAWre2FfXwBzka"
CONSUMER_SECRET = "bxkGaNxkn7ocgn3eSjxurOwPdOoa"

# Target download directory for the new 800 scenes
RAW_DOWNLOAD_DIR = r"D:\FINDFIRE_DATA\raw_800"
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
EXCEL_PATH = "nasa data set.xlsx"

if not os.path.exists(EXCEL_PATH):
  EXCEL_PATH = os.path.join("..", "nasa data set.xlsx")

os.makedirs(RAW_DOWNLOAD_DIR, exist_ok=True)

TARGET_SCENE_COUNT = 800
COLLECTION_ID = "EO:EUM:DAT:MSG:MSG15-RSS"  # Meteosat Rapid Scan Service (5-min)

# Geographic boundaries for Northern Algeria
LAT_MIN, LAT_MAX = 34.0, 37.5
LON_MIN, LON_MAX = -2.0, 8.5

# =====================================================================
# 2. SELECT THE 800 HIGHEST-FRP UNSEEN SCENES
# =====================================================================
print("==================================================")
print("FINDFIRE BATCH DOWNLOADER (800 NEW SCENES)")
print("==================================================")
print(f"Reading candidate fire events from: {EXCEL_PATH}")

df_nasa = pd.read_excel(EXCEL_PATH)

# Filter spatial bounding box
df_algeria = df_nasa[
    (df_nasa["latitude"] >= LAT_MIN)
    & (df_nasa["latitude"] <= LAT_MAX)
    & (df_nasa["longitude"] >= LON_MIN)
    & (df_nasa["longitude"] <= LON_MAX)
].copy()

# Parse datetime
time_str = df_algeria["acq_time"].astype(int).astype(str).str.zfill(4)
date_str = df_algeria["acq_date"].astype(str).str[:10]
df_algeria["timestamp"] = pd.to_datetime(
    date_str + " " + time_str, format="%Y-%m-%d %H%M"
)

# Group observations into individual satellite overpass scenes
scenes = (
    df_algeria.groupby(["acq_date", "acq_time", "daynight"])
    .agg(
        fire_count=("latitude", "count"),
        total_frp=("frp", "sum"),
        max_frp=("frp", "max"),
        timestamp=("timestamp", "first"),
    )
    .reset_index()
    .sort_values(by="total_frp", ascending=False)
)

# Exclude scenes already present in the existing dataset index
if os.path.exists(INDEX_PATH):
  df_existing = pd.read_csv(INDEX_PATH)
  existing_timestamps = pd.to_datetime(df_existing["timestamp"])

  is_unseen = []
  for _, row in scenes.iterrows():
    diffs = (existing_timestamps - row["timestamp"]).abs()
    already_downloaded = (diffs <= pd.Timedelta(minutes=15)).any()
    is_unseen.append(not already_downloaded)

  candidate_scenes = scenes[is_unseen].reset_index(drop=True)
else:
  candidate_scenes = scenes.reset_index(drop=True)

# Select top target scenes by FRP
target_scenes = candidate_scenes.head(TARGET_SCENE_COUNT).copy()
print(f"Selected {len(target_scenes)} new scenes based on maximum FRP.")
print(f"  • Daytime scenes  : {(target_scenes['daynight'] == 'D').sum()}")
print(f"  • Nighttime scenes: {(target_scenes['daynight'] == 'N').sum()}\n")

# Save the target download manifest for traceability
manifest_path = os.path.join(RAW_DOWNLOAD_DIR, "download_manifest_800.csv")
target_scenes.to_csv(manifest_path, index=False)
print(f"Saved download manifest to: {manifest_path}\n")

# =====================================================================
# 3. INITIALIZE EUMETSAT DATASTORE CONNECTION
# =====================================================================
print("Connecting to EUMETSAT Data Store API...")
token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
datastore = eumdac.DataStore(token)
selected_collection = datastore.get_collection(COLLECTION_ID)
print("Connected successfully.\n")

# =====================================================================
# 4. DOWNLOAD PIPELINE WITH AUTO-RESUME
# =====================================================================
successful_downloads = 0
failed_downloads = 0
skipped_downloads = 0

print("Starting batch download...")
print("-" * 75)

for i, row in target_scenes.reset_index(drop=True).iterrows():
  scene_idx = i + 1
  scene_dt = row["timestamp"]
  date_tag = scene_dt.strftime("%Y%m%d_%H%M")
  dest_file_prefix = f"MSG_RSS_{date_tag}"

  # Check if scene file already exists in raw_800 folder (resume capability)
  existing_matches = [
      f for f in os.listdir(RAW_DOWNLOAD_DIR) if f.startswith(dest_file_prefix)
  ]
  if len(existing_matches) > 0:
    skipped_downloads += 1
    continue

  # Query window: +/- 7 minutes around the target NASA observation
  start_window = scene_dt - timedelta(minutes=7)
  end_window = scene_dt + timedelta(minutes=7)

  try:
    products = selected_collection.search(dtstart=start_window, dtend=end_window)
    matched_product = None

    for prod in products:
      matched_product = prod
      break

    if matched_product is None:
      print(f"[{scene_idx:03d}/{TARGET_SCENE_COUNT}] {date_tag} -> No satellite product found.")
      failed_downloads += 1
      continue

    # Download product entry
    start_time = time.time()
    with matched_product.open() as f_src:
      filename = matched_product.entries[0] if hasattr(matched_product, "entries") else str(matched_product)
      ext = ".nat" if ".nat" in filename else ".zip"
      out_filepath = os.path.join(RAW_DOWNLOAD_DIR, f"{dest_file_prefix}{ext}")

      with open(out_filepath, "wb") as f_dst:
        shutil.copyfileobj(f_src, f_dst)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(out_filepath) / (1024 * 1024)
    successful_downloads += 1

    print(
        f"[{scene_idx:03d}/{TARGET_SCENE_COUNT}] {date_tag} ({row['daynight']}) | "
        f"FRP: {row['total_frp']:6.1f} MW | Size: {file_size_mb:4.1f} MB | {elapsed:4.1f}s -> Saved"
    )

  except Exception as e:
    print(f"[{scene_idx:03d}/{TARGET_SCENE_COUNT}] {date_tag} -> Error: {e}")
    failed_downloads += 1
    time.sleep(2)  # Short pause before retrying next query

print("-" * 75)
print("\n==================================================")
print("DOWNLOAD BATCH COMPLETE")
print(f"Successfully Downloaded : {successful_downloads}")
print(f"Already Existed (Skipped): {skipped_downloads}")
print(f"Failed / Unavailable    : {failed_downloads}")
print(f"Files saved in          : {RAW_DOWNLOAD_DIR}")
print("==================================================")