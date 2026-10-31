# tigernetv1

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TigerNet (FindFire)** is an operational dual-regime deep learning system designed for real-time wildfire detection and continuous monitoring across Northern Algeria using Meteosat Second Generation (MSG / SEVIRI) geostationary satellite telemetry.

---

## Overview

Wildfire detection via geostationary orbit provides high temporal resolution (15-minute cadence), enabling early intervention before fires grow catastrophic. However, satellite fire detection faces severe physical challenges:
- **Diurnal Disparity**: Solar reflection during daytime corrupts mid-wave infrared (IR 3.9µm) readings, producing severe false alarms that do not exist at night.
- **Extreme Class Imbalance**: Active fire pixels represent $< 0.01\%$ of all monitored land pixels.
- **Thermal Anomalies & Artifacts**: Solar glint, hot bare soil, and static industrial flare stacks trigger false alerts without temporal filtering and spatial land masking.

TigerNet addresses these challenges via a **Dual-Regime U-Net Architecture** coupled with temporal differentiation ($\Delta T$) and automated land/sea masking.

---

## Key Features

- **Dual-Regime Deep Learning**:
  - **Day Model**: Trained on multi-channel inputs (IR 3.9, IR 10.8, VIS 0.6, Solar Zenith Angle, $\Delta T_{3.9}$) to isolate true combustion from solar scattering.
  - **Night Model**: Calibrated for high-sensitivity thermal anomaly detection with low false alarm rates ($\tau_{\text{night}} = 0.95$).
- **Temporal Differentiation ($\Delta T$)**: Compares consecutive 15-minute scans to detect rapid thermal rise ($\Delta T > 1.5\,\text{K}$), differentiating nascent fires from steady-state industrial hotspots.
- **Automated Live Monitor Daemon (`live_fire_monitor.py`)**:
  - Connects to EUMETSAT Data Store via `eumdac`.
  - Downloads, decompresses, and crops satellite scans in real-time.
  - Runs dual-regime inference, extracts fire clusters, and publishes incident alerts (`live_alerts/active_incidents.jsonl`).
- **Production Performance**:
  - Evaluated on benchmark scenes across Algeria with ground-truth validation against NASA FIRMS (VIIRS / MODIS) hotspots.

---

## Target Geographic Coverage

- **Region**: Northern Algeria
- **Bounding Box**: Lat $34.0^\circ\text{N} - 37.5^\circ\text{N}$, Lon $-2.0^\circ\text{W} - 8.5^\circ\text{E}$
- **Grid Resolution**: $120 \times 250$ equidistant projection ($\sim 3\,\text{km} \times 3\,\text{km}$ per pixel, $9\,\text{km}^2$)

---

## Repository Structure

```
tigernetv1/
├── findfire/                       # Core python package & operational modules
│   ├── unet_model.py               # U-Net architecture implementation
│   ├── live_fire_monitor.py        # Real-time daemon polling EUMETSAT orbit streams
│   ├── findfire_engine.py          # Unified dual-regime inference engine
│   ├── losses_and_metrics.py       # Custom loss functions (Focal, Dice, Tversky)
│   ├── wildfire_dataset.py         # PyTorch dataset for nocturnal scenes
│   ├── wildfire_day_dataset.py     # PyTorch dataset for diurnal scenes
│   ├── train_day_800.py            # Daytime model training pipeline
│   ├── train_night_800.py          # Nighttime model training pipeline
│   └── benchmark_800.py            # Large-scale evaluation & metrics benchmark
├── FINDFIRE_REPORT/                # Technical research papers, reports & slide decks
│   ├── AgentGuard_Project_Explanation.pdf
│   ├── final_production_pipeline_land_masking_and_threshold_calibration.pdf
│   └── ...
├── checkpoints/                    # Directory for model weights (*.pth)
├── live_alerts/                    # Generated alert previews and incident logs
├── batch_download.py               # Bulk scene downloader from EUMETSAT
├── simulation.py                   # Offline playback and alert simulator
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Kingramy/tigernetv1.git
cd tigernetv1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install torch torchvision numpy scipy pandas matplotlib satpy eumdac openpyxl
```

### 2. Configuration & Credentials

To enable real-time satellite streaming from EUMETSAT, obtain API credentials from the [EUMETSAT Data Store](https://data.eumetsat.int/) and configure your environment:

```bash
export EUMETSAT_CONSUMER_KEY="your_consumer_key"
export EUMETSAT_CONSUMER_SECRET="your_consumer_secret"
```

### 3. Running Live Monitoring Daemon

```bash
python live_fire_monitor.py
```

### 4. Running Simulation Mode

To simulate real-time detection on historical test scenes:

```bash
python simulation.py
```

---

## Model Architecture & Loss Formulation

TigerNet utilizes a symmetric U-Net encoder-decoder network optimized for dense segmentation of small target features:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{Focal}} + \beta \mathcal{L}_{\text{Dice}}$$

- **Focal Loss** addresses the extreme background-to-fire pixel imbalance by down-weighting well-classified background regions.
- **Dice Loss** penalizes boundary mismatch on localized fire clusters.

---

## Reports & Documentation

Comprehensive technical analysis, architectural decisions, and diagnostic reports are documented in [`FINDFIRE_REPORT/`](FINDFIRE_REPORT/).

---

## Author & Acknowledgements

Developed as part of the AI Engineering Wildfire Early Warning Research Initiative for Algeria.
Satellite data provided by [EUMETSAT](https://www.eumetsat.int/). Reference hotspot data provided by [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/).

