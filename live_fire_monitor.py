import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label
import torch
import torch.nn.functional as F

from unet_model import UNet

# =====================================================================
# 1. OPERATIONAL CONFIGURATION
# =====================================================================
MONITORING_MODE = "LIVE"  # "LIVE" to stream from orbit, or "SIMULATION"

# EUMETSAT API Credentials (Only needed for "LIVE" mode)
CONSUMER_KEY = "EhZhSU9MOW6R_hZAWre2FfXwBzka"
CONSUMER_SECRET = "bxkGaNxkn7ocgn3eSjxurOwPdOoa"

# Model Checkpoints
NIGHT_CKPT = os.path.join("checkpoints", "best_unet_night_800.pth")
DAY_CKPT = os.path.join("checkpoints", "best_unet_day_800.pth")
INDEX_PATH = r"D:\FINDFIRE_DATA\processed_800\dataset_index_800.csv"

# Alert Storage
ALERT_DIR = "live_alerts"
ALERT_LOG_FILE = os.path.join(ALERT_DIR, "active_incidents.jsonl")
INDUSTRIAL_LOG_FILE = os.path.join(ALERT_DIR, "industrial_hotspots.jsonl")
os.makedirs(ALERT_DIR, exist_ok=True)

# Northern Algeria Equidistant Grid
LAT_MIN, LAT_MAX = 34.0, 37.5
LON_MIN, LON_MAX = -2.0, 8.5
GRID_H, GRID_W = 120, 250
PIXEL_AREA_KM2 = 9.0  # ~3 km x 3 km per pixel

# Detection Thresholds
TAU_DAY = 0.90
TAU_NIGHT = 0.95
TAU_NASCENT_DAY = 0.70
TAU_NASCENT_NIGHT = 0.75
RATE_OF_RISE_THRESHOLD_K = 1.5

# Normalization Constants
NIGHT_MEANS = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
NIGHT_STDS = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)
DAY_MEANS = torch.tensor([315.0, 300.0, 15.0, 0.0]).view(4, 1, 1)
DAY_STDS = torch.tensor([15.0, 12.0, 8.0, 4.0]).view(4, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 2. STATIC LAND & INDUSTRIAL EXCLUSION MASKS
# =====================================================================
# 1. Land Mask: Suppress Mediterranean open water above 36.9°N
lat_per_row = np.linspace(LAT_MAX, LAT_MIN, GRID_H)
LAND_MASK = np.repeat((lat_per_row <= 36.9)[:, None], GRID_W, axis=1)

# 2. Algerian Heavy Industrial Heat Sources (Flares, Steel Mills, Cement Kilns)
KNOWN_INDUSTRIAL_SITES = [
    # Petrochemical & Gas Flares
    {
        "name": "Skikda Sonatrach Refinery & LNG Flares",
        "lat": 36.8875,
        "lon": 6.9250,
        "radius_km": 5.0,
    },
    {
        "name": "Arzew / Bethioua Petrochemical & LNG Terminal",
        "lat": 35.8300,
        "lon": -0.2800,
        "radius_km": 6.0,
    },
    # Steel & Metallurgy
    {
        "name": "Bellara Steel Complex (AQS - El Milia, Jijel)",
        "lat": 36.7708,
        "lon": 6.1900,
        "radius_km": 5.0,
    },
    {
        "name": "El Hadjar Sider Steelworks (Annaba)",
        "lat": 36.8000,
        "lon": 7.7200,
        "radius_km": 5.0,
    },
    # Rotary Cement & Lime Kilns (>1,400°C)
    {
        "name": "SCHB Hamma Bouziane Cement (Constantine)",
        "lat": 36.4500,
        "lon": 6.5680,
        "radius_km": 4.5,
    },
    {
        "name": "GICA Sigus Mega Cement Plant (Oum El Bouaghi)",
        "lat": 36.1000,
        "lon": 6.7360,
        "radius_km": 4.5,
    },
    {
        "name": "Hammam Delaa Brick & Thermal Works (M'Sila)",
        "lat": 35.8958,
        "lon": 4.3840,
        "radius_km": 4.5,
    },
    # Inside KNOWN_INDUSTRIAL_SITES:
    {
    "name": "Skikda Sonatrach Refinery & LNG Flares",
    "lat": 36.8875,
    "lon": 6.9250,
    "radius_km": 8.0,  # Expands past the adjacent 6.9460°E pixel
    },
    # ...
    {
    "name": "CILAS Mega Cement Plant (Branis / Djemorah, Biskra)",
    "lat": 35.0208,
    "lon": 5.6650,
    "radius_km": 7.5,  # Expands past the adjacent 5.6860°E pixel
    },
    {
        "name": "Lafarge Cement Plant (M'Sila)",
        "lat": 35.7500,
        "lon": 4.4260,
        "radius_km": 4.5,
    },
    {
        "name": "SCIMAT Ain Touta Cement Plant (Batna)",
        "lat": 35.3708,
        "lon": 5.7280,
        "radius_km": 4.5,
    },
    {
        "name": "El Kantara Quarry & Lime Kilns (Biskra)",
        "lat": 35.1375,
        "lon": 5.6020,
        "radius_km": 4.5,
    },
    {
        "name": "CILAS Mega Cement Plant (Branis / Djemorah, Biskra)",
        "lat": 35.0208,
        "lon": 5.6650,
        "radius_km": 4.5,
    },
    {
        "name": "Boutlelis / Misserghin Quarry Works (Oran)",
        "lat": 35.6042,
        "lon": -0.9500,
        "radius_km": 4.5,
    },
    {
        "name": "Sour El Ghozlane Cement Plant (Bouira)",
        "lat": 36.1500,
        "lon": 3.6900,
        "radius_km": 4.5,
    },
    {
        "name": "ECDE Oued Sly Cement Plant (Chlef)",
        "lat": 36.1000,
        "lon": 1.2000,
        "radius_km": 4.5,
    },
    {
        "name": "Zahana Cement Plant (Mascara)",
        "lat": 35.4500,
        "lon": -0.5000,
        "radius_km": 4.5,
    },
    {
        "name": "Beni Saf Cement Plant (Ain Temouchent)",
        "lat": 35.2800,
        "lon": -1.3500,
        "radius_km": 4.5,
    },
    {
        "name": "Meftah Cement Plant (Blida)",
        "lat": 36.6200,
        "lon": 3.2300,
        "radius_km": 4.5,
    },
]


def build_industrial_exclusion_mask():
  """Constructs a binary mask where True = Forest/Land, False = Industrial Exclusion Zone."""
  mask = np.ones((GRID_H, GRID_W), dtype=bool)

  # Precompute meshgrid of geographic coordinates
  lats = np.linspace(LAT_MAX, LAT_MIN, GRID_H)
  lons = np.linspace(LON_MIN, LON_MAX, GRID_W)
  lon_grid, lat_grid = np.meshgrid(lons, lats)

  # Approximate flat-earth metric conversion: 1 deg lat ~ 111 km, 1 deg lon ~ 89 km (at 36°N)
  km_per_deg_lat = 111.0
  km_per_deg_lon = 89.0

  for site in KNOWN_INDUSTRIAL_SITES:
    d_lat = (lat_grid - site["lat"]) * km_per_deg_lat
    d_lon = (lon_grid - site["lon"]) * km_per_deg_lon
    dist_km = np.sqrt(d_lat**2 + d_lon**2)

    # Exclude pixels within the operational radius
    site_mask = dist_km <= site["radius_km"]
    mask[site_mask] = False

  return mask


INDUSTRIAL_MASK = build_industrial_exclusion_mask()
VALID_TERRESTRIAL_DOMAIN = LAND_MASK & INDUSTRIAL_MASK
print(f"Constructed Industrial Exclusion Mask:")
print(f"  • Registered Heavy Industrial Facilities: {len(KNOWN_INDUSTRIAL_SITES)}")
print(
    f"  • Active Forest/Wildfire Search Area   :"
    f" {np.sum(VALID_TERRESTRIAL_DOMAIN):,} pixels (~{np.sum(VALID_TERRESTRIAL_DOMAIN)*9:,} km²)"
)
print(
    f"  • Suppressed Industrial Buffer Pixels  :"
    f" {np.sum(LAND_MASK & ~INDUSTRIAL_MASK)} pixels\n"
)

# =====================================================================
# 3. MODEL INITIALIZATION
# =====================================================================
print("=" * 75)
print("FINDFIRE: REAL-TIME OPERATIONAL WILDFIRE DAEMON (V2)")
print(f"Hardware Engine: {device} | Mode: {MONITORING_MODE}")
print("=" * 75)

if not os.path.exists(NIGHT_CKPT) or not os.path.exists(DAY_CKPT):
  sys.exit("Error: Production checkpoints not found.")

print("Loading UNet_Night (3 Channels)...")
night_state = torch.load(NIGHT_CKPT, map_location=device, weights_only=False)
model_night = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model_night.load_state_dict(night_state["model_state_dict"])
model_night.eval()

print("Loading UNet_Day (4 Channels)...")
day_state = torch.load(DAY_CKPT, map_location=device, weights_only=False)
model_day = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model_day.load_state_dict(day_state["model_state_dict"])
model_day.eval()
print("Neural models loaded successfully.\n")


# =====================================================================
# 4. SPATIAL & INFERENCE CORE
# =====================================================================
def run_spatial_inference(raw_3ch, dt_timestamp):
  hour = dt_timestamp.hour
  is_night = (hour < 6) or (hour >= 18)
  regime = "NIGHT" if is_night else "DAY"
  img_tensor = torch.from_numpy(raw_3ch).float()

  if is_night:
    norm_img = (img_tensor - NIGHT_MEANS) / NIGHT_STDS
    padded = (
        F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)
    )
    with torch.no_grad():
      logits = model_night(padded)
      probs = torch.sigmoid(logits)[0, 0, 4:124, 3:253].cpu().numpy()

    # Absolute temperature sanity gate
    t_39 = raw_3ch[0]
    probs = probs * (t_39 >= 285.0).astype(np.float32)
  else:
    # 11x11 Local Background Contrast
    delta_t = img_tensor[2:3, :, :].unsqueeze(0)
    local_bg = F.avg_pool2d(
        delta_t, kernel_size=11, stride=1, padding=5
    ).squeeze(0)
    contrast = img_tensor[2:3, :, :] - local_bg
    img_4ch = torch.cat([img_tensor, contrast], dim=0)

    norm_img = (img_4ch - DAY_MEANS) / DAY_STDS
    padded = (
        F.pad(norm_img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)
    )
    with torch.no_grad():
      logits = model_day(padded)
      probs = torch.sigmoid(logits)[0, 0, 4:124, 3:253].cpu().numpy()

  # Enforce static land barrier
  probs = probs * LAND_MASK
  return probs, regime


# =====================================================================
# 5. TEMPORAL RATE-OF-RISE TRACKER
# =====================================================================
class TemporalFireTracker:

  def __init__(self):
    self.prev_timestamp = None
    self.prev_delta_t = None
    self.prev_probs = None

  def analyze_frame(self, raw_3ch, current_timestamp):
    current_delta_t = raw_3ch[2]
    current_probs, regime = run_spatial_inference(raw_3ch, current_timestamp)

    tau_standard = TAU_NIGHT if regime == "NIGHT" else TAU_DAY
    tau_nascent = TAU_NASCENT_NIGHT if regime == "NIGHT" else TAU_NASCENT_DAY

    rate_of_rise = np.zeros_like(current_delta_t)
    time_delta_mins = 5.0

    if self.prev_delta_t is not None and self.prev_timestamp is not None:
      time_delta_mins = (
          current_timestamp - self.prev_timestamp
      ).total_seconds() / 60.0
      if 3.0 <= time_delta_mins <= 15.0:
        dt_factor = 5.0 / time_delta_mins
        rate_of_rise = (current_delta_t - self.prev_delta_t) * dt_factor
        rate_of_rise = np.clip(rate_of_rise, -20.0, 50.0)

    # All detections on land
    standard_mask = current_probs >= tau_standard
    nascent_mask = (current_probs >= tau_nascent) & (
        rate_of_rise >= RATE_OF_RISE_THRESHOLD_K
    )
    raw_detections = (standard_mask | nascent_mask) & LAND_MASK

    # Split into Wildfire Incidents vs Suppressed Industrial Hotspots
    wildfire_mask = raw_detections & INDUSTRIAL_MASK
    industrial_mask = raw_detections & (~INDUSTRIAL_MASK)

    # Update frame buffer
    self.prev_timestamp = current_timestamp
    self.prev_delta_t = current_delta_t.copy()
    self.prev_probs = current_probs.copy()

    return {
        "wildfire_mask": wildfire_mask.astype(np.float32),
        "industrial_mask": industrial_mask.astype(np.float32),
        "nascent_mask": (nascent_mask & INDUSTRIAL_MASK).astype(np.float32),
        "standard_mask": (standard_mask & INDUSTRIAL_MASK).astype(np.float32),
        "probs": current_probs,
        "rate_of_rise": rate_of_rise,
        "regime": regime,
        "time_delta_mins": time_delta_mins,
    }


# =====================================================================
# 6. INCIDENT DISPATCHER
# =====================================================================
def dispatch_incident_alerts(analysis, raw_3ch, timestamp):
  wildfire_mask = analysis["wildfire_mask"]
  industrial_mask = analysis["industrial_mask"]
  nascent_mask = analysis["nascent_mask"]
  rate_of_rise = analysis["rate_of_rise"]
  probs = analysis["probs"]
  regime = analysis["regime"]

  # Check industrial suppression
  ind_pixels = int(np.sum(industrial_mask))
  if ind_pixels > 0:
    ind_y, ind_x = np.where(industrial_mask == 1.0)
    for y, x in zip(ind_y, ind_x):
      lat = LAT_MAX - (y / GRID_H) * (LAT_MAX - LAT_MIN)
      lon = LON_MIN + (x / GRID_W) * (LON_MAX - LON_MIN)
      # Find matching factory
      matched_name = "Industrial Area"
      for s in KNOWN_INDUSTRIAL_SITES:
        d = np.sqrt(
            ((lat - s["lat"]) * 111) ** 2 + ((lon - s["lon"]) * 89) ** 2
        )
        if d <= s["radius_km"] + 3.0:
          matched_name = s["name"]
          break
      with open(INDUSTRIAL_LOG_FILE, "a") as f_ind:
        f_ind.write(
            json.dumps({
                "timestamp": timestamp.isoformat(),
                "site": matched_name,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "confidence": round(float(probs[y, x]), 3),
            })
            + "\n"
        )
    print(
        f"[{timestamp.strftime('%H:%M:%S')} UTC] Notice: Filtered {ind_pixels}"
        " permanent industrial heat signature(s) (Flares/Cement Kilns)."
    )

  # Check real wildfire fronts
  labeled_array, num_features = label(wildfire_mask)
  if num_features == 0:
    print(
        f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Regime: {regime:5s} |"
        " Status: All Clear (0 Active Wildfires)"
    )
    return

  incidents = []
  print(f"\n" + "!" * 80)
  print(
      f"WILDFIRE EMERGENCY ALERT: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
  )
  print(
      f"Operational Regime: {regime} | Active Wildfire Fronts: {num_features}"
  )
  print("!" * 80)

  for cluster_id in range(1, num_features + 1):
    ys, xs = np.where(labeled_array == cluster_id)
    cluster_pixels = len(ys)
    area_km2 = cluster_pixels * PIXEL_AREA_KM2

    lat = LAT_MAX - (float(np.mean(ys)) / GRID_H) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (float(np.mean(xs)) / GRID_W) * (LON_MAX - LON_MIN)

    max_t39 = float(np.max(raw_3ch[0][ys, xs]))
    max_rise = float(np.max(rate_of_rise[ys, xs]))
    mean_conf = float(np.mean(probs[ys, xs]))
    is_nascent = bool(np.any(nascent_mask[ys, xs]))

    if is_nascent and cluster_pixels <= 3:
      classification = "NASCENT_IGNITION_CRITICAL"
      action = "IMMEDIATE EVACUATION & AIR TANKER DISPATCH"
    elif cluster_pixels > 6:
      classification = "MAJOR_EXPANDING_WILDFIRE"
      action = "REGIONAL PERIMETER CONTAINMENT"
    else:
      classification = "ACTIVE_FLAMING_FRONT"
      action = "GROUND UNIT INTERVENTION"

    incident_record = {
        "incident_id": (
            f"INC_{timestamp.strftime('%Y%m%d_%H%M')}_{cluster_id:02d}"
        ),
        "timestamp": timestamp.isoformat(),
        "regime": regime,
        "classification": classification,
        "directive": action,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "estimated_area_km2": round(area_km2, 1),
        "pixel_count": cluster_pixels,
        "confidence": round(mean_conf, 3),
        "peak_t39_K": round(max_t39, 1),
        "rate_of_rise_K_5min": round(max_rise, 2),
    }
    incidents.append(incident_record)

    print(f" > FRONT #{cluster_id:02d} [{classification}]:")
    print(f"   Coordinates : Lat {lat:.4f}°N, Lon {lon:.4f}°E")
    print(
        f"   Front Size  : ~{area_km2:.1f} km² ({cluster_pixels} px) | Peak"
        f" T3.9: {max_t39:.1f} K"
    )
    print(
        f"   Rate of Rise: +{max_rise:.2f} K / 5 min | Confidence:"
        f" {mean_conf*100:.1f}%"
    )
    print(f"   Directive   : {action}\n")

  with open(ALERT_LOG_FILE, "a") as f_out:
    for inc in incidents:
      f_out.write(json.dumps(inc) + "\n")

  render_alert_snapshot(raw_3ch, analysis, incidents, timestamp)


def render_alert_snapshot(raw_3ch, analysis, incidents, timestamp):
  wildfire_mask = analysis["wildfire_mask"]
  rate_of_rise = analysis["rate_of_rise"]
  date_tag = timestamp.strftime("%Y%m%d_%H%M%S")
  out_png = os.path.join(ALERT_DIR, f"incident_{date_tag}.png")

  fig, axes = plt.subplots(1, 3, figsize=(20, 5))

  # Delta T Channel
  im0 = axes[0].imshow(
      raw_3ch[2] * VALID_TERRESTRIAL_DOMAIN, cmap="inferno", origin="upper"
  )
  axes[0].set_title(
      f"Thermal Channel: Delta T [{timestamp.strftime('%H:%M:%S')} UTC]"
  )
  fig.colorbar(im0, ax=axes[0], label="Kelvin")

  # Rate of Rise
  im1 = axes[1].imshow(
      rate_of_rise * VALID_TERRESTRIAL_DOMAIN,
      cmap="coolwarm",
      vmin=-2.0,
      vmax=5.0,
      origin="upper",
  )
  axes[1].set_title("Temporal Rate-of-Rise: d(Delta T)/dt (K / 5 min)")
  fig.colorbar(im1, ax=axes[1], label="d(Kelvin)/dt")

  # Civil Protection Overlay
  axes[2].imshow(
      raw_3ch[2] * VALID_TERRESTRIAL_DOMAIN, cmap="gray", origin="upper"
  )

  std_y, std_x = np.where(
      (analysis["standard_mask"] == 1.0) & (analysis["nascent_mask"] == 0.0)
  )
  if len(std_y) > 0:
    axes[2].scatter(std_x, std_y, color="orange", s=30, label="Wildfire Front")

  nas_y, nas_x = np.where(analysis["nascent_mask"] == 1.0)
  if len(nas_y) > 0:
    axes[2].scatter(
        nas_x,
        nas_y,
        color="red",
        s=70,
        marker="*",
        label="Nascent Ignition (Early)",
    )

  for inc in incidents:
    c_x = int((inc["longitude"] - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_W - 1))
    c_y = int((LAT_MAX - inc["latitude"]) / (LAT_MAX - LAT_MIN) * (GRID_H - 1))
    axes[2].text(
        c_x + 3,
        c_y - 3,
        inc["incident_id"][-5:],
        color="cyan",
        fontsize=9,
        weight="bold",
    )

  axes[2].set_title(
      f"Civil Protection Wildfire Map ({len(incidents)} Real Incidents)"
  )
  axes[2].legend(loc="lower left")

  plt.tight_layout()
  plt.savefig(out_png, dpi=200)
  plt.close()
  print(f"Tactical map saved to: {out_png}")


# =====================================================================
# 7. RUNTIME ENGINE
# =====================================================================
def run_live_daemon():
  import gc
  import eumdac
  from pyresample.geometry import AreaDefinition
  from satpy import Scene

  area_def = AreaDefinition(
      area_id="northern_algeria",
      description="Northern Algeria Target Domain",
      proj_id="latlong",
      projection={"proj": "latlong", "datum": "WGS84"},
      width=GRID_W,
      height=GRID_H,
      area_extent=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX],
  )

  tracker = TemporalFireTracker()
  last_processed_time = None

  # Track token renewal time (refresh every 45 minutes)
  token_issued_at = datetime.now(timezone.utc)
  token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
  datastore = eumdac.DataStore(token)
  collection = datastore.get_collection("EO:EUM:DAT:MSG:MSG15-RSS")
  print("Connected to EUMETSAT. Running 24/7 continuous surveillance...\n")

  while True:
    try:
      now_utc = datetime.now(timezone.utc)

      # 1. AUTO-REFRESH TOKEN BEFORE 60-MIN EXPIRY
      if (now_utc - token_issued_at).total_seconds() > 2700:  # 45 minutes
        print(
            f"[{now_utc.strftime('%H:%M:%S')} UTC] Refreshing EUMETSAT"
            " security token..."
        )
        token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection("EO:EUM:DAT:MSG:MSG15-RSS")
        token_issued_at = now_utc

      # 2. QUERY RECENT PRODUCTS
      start_window = now_utc - timedelta(minutes=15)
      products = collection.search(dtstart=start_window, dtend=now_utc)

      latest_product = None
      for prod in products:
        latest_product = prod
        break

      if latest_product is None:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Polling... Waiting for"
            " new RSS product."
        )
        time.sleep(60)
        continue

      prod_dt = latest_product.sensing_start
      if last_processed_time is not None and prod_dt <= last_processed_time:
        time.sleep(45)
        continue

      print(
          f"\nNew Satellite Frame Acquired:"
          f" {prod_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"
      )

      # 3. ISOLATED IN-MEMORY PROCESSING
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_archive = os.path.join(temp_dir, "incoming.zip")
        with latest_product.open() as f_src, open(temp_archive, "wb") as f_dst:
          shutil.copyfileobj(f_src, f_dst)

        target_nat = None
        if zipfile.is_zipfile(temp_archive):
          with zipfile.ZipFile(temp_archive, "r") as z:
            for n in z.namelist():
              if n.endswith(".nat") or "MSG" in n:
                z.extract(n, temp_dir)
                target_nat = os.path.join(temp_dir, n)
                break
        else:
          target_nat = temp_archive

        scn = Scene(filenames=[target_nat], reader="seviri_l1b_native")
        scn.load(["IR_039", "IR_108"])
        resampled = scn.resample(area_def)

        t39 = np.nan_to_num(
            resampled["IR_039"].values.astype(np.float32), nan=0.0
        )
        t108 = np.nan_to_num(
            resampled["IR_108"].values.astype(np.float32), nan=0.0
        )
        delta_t = t39 - t108
        raw_3ch = np.stack([t39, t108, delta_t], axis=0)

        # Free memory handles
        del scn, resampled
        gc.collect()

      analysis = tracker.analyze_frame(raw_3ch, prod_dt)
      dispatch_incident_alerts(analysis, raw_3ch, prod_dt)
      last_processed_time = prod_dt

    except Exception as e:
      print(f"[{datetime.now().strftime('%H:%M:%S')}] Runtime exception: {e}")
      print("Recovering connection... Retrying in 30 seconds.")
      # Invalidate token on error so it forces a fresh connection
      token_issued_at = datetime.now(timezone.utc) - timedelta(hours=2)
      time.sleep(30)

if __name__ == "__main__":
  if MONITORING_MODE == "LIVE":
    run_live_daemon()
  else:
    sys.exit("Please select MONITORING_MODE = 'LIVE'.")