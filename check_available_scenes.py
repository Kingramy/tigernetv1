import os
import pandas as pd

# 1. Load the NASA ground truth dataset
EXCEL_PATH = "nasa data set.xlsx"
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"

if not os.path.exists(EXCEL_PATH):
  # Check one folder up if not found
  EXCEL_PATH = os.path.join("..", "nasa data set.xlsx")

print(f"Reading NASA dataset from: {EXCEL_PATH}...")
df_nasa = pd.read_excel(EXCEL_PATH)

# Filter for Northern Algeria bounding box
LAT_MIN, LAT_MAX = 34.0, 37.5
LON_MIN, LON_MAX = -2.0, 8.5

df_algeria = df_nasa[
    (df_nasa["latitude"] >= LAT_MIN)
    & (df_nasa["latitude"] <= LAT_MAX)
    & (df_nasa["longitude"] >= LON_MIN)
    & (df_nasa["longitude"] <= LON_MAX)
].copy()

# Build timestamps
time_str = df_algeria["acq_time"].astype(int).astype(str).str.zfill(4)
date_str = df_algeria["acq_date"].astype(str).str[:10]
df_algeria["timestamp"] = pd.to_datetime(
    date_str + " " + time_str, format="%Y-%m-%d %H%M"
)

# Group into satellite scenes
scenes = (
    df_algeria.groupby(["acq_date", "acq_time", "daynight"])
    .agg(
        fire_points=("latitude", "count"),
        total_frp=("frp", "sum"),
        max_frp=("frp", "max"),
        timestamp=("timestamp", "first"),
    )
    .reset_index()
    .sort_values(by="total_frp", ascending=False)
)

print(
    f"Total available fire scenes in Northern Algeria: {len(scenes)} scenes"
)

# 2. Check already downloaded scenes
if os.path.exists(INDEX_PATH):
  df_existing = pd.read_csv(INDEX_PATH)
  existing_timestamps = pd.to_datetime(df_existing["timestamp"])

  # Filter out timestamps already within 15 minutes of existing scenes
  is_new = []
  for _, row in scenes.iterrows():
    diffs = (existing_timestamps - row["timestamp"]).abs()
    already_have = (diffs <= pd.Timedelta(minutes=15)).any()
    is_new.append(not already_have)

  new_scenes = scenes[is_new].reset_index(drop=True)
  print(f"Scenes already processed in dataset: {len(df_existing)}")
  print(f"Completely NEW scenes available to download: {len(new_scenes)}")
  print(f"  - Daytime new scenes : {(new_scenes['daynight'] == 'D').sum()}")
  print(f"  - Nighttime new scenes: {(new_scenes['daynight'] == 'N').sum()}")
else:
  print(
      f"Warning: Could not find existing index at {INDEX_PATH} to cross-check."
  )