## Weather Forecasting (Belgium wind) — interview-ready walkthrough

This repo contains:
- **Python ML pipeline** (`weather-ml-project/`): download GFS → crop Belgium → build dataset → train CNN → evaluate + plots → **postprocessing correction**
- **Frontend UI** (`frontend/`): optional Vite web app

### What you can present (7-step story)

1. **Download FULL GFS** (`src/download_gfs.py`)
   - Downloads **exactly 5 full GRIB2 files** (1 date × 5 forecast hours).
   - Small enough to run quickly before an interview.
2. **Crop to Belgium** (`src/preprocess.py`)
   - Reads GRIB2 with `cfgrib`, selects a Belgium bounding box, saves NetCDF (`data/processed/*.nc`).
3. **Dataset** (`src/dataset.py`)
   - Stacks time steps into a tensor time-series.
   - Creates samples: “use last 2 steps to predict 1 step ahead”.
4. **Model** (`src/model.py`)
   - A small CNN with residual blocks that maps input wind maps → next-step wind maps.
5. **Training** (`src/train.py`)
   - Chronological split (forecasting), early stopping, saves `saved_models/wind_forecast_cnn.pth`.
6. **Evaluation + plots** (`src/evaluate.py`)
   - Computes MSE/MAE over the validation set and saves “best/worst” maps to `outputs/`.
7. **Postprocessing correction (ML calibration)** (`src/evaluate.py` section)
   - Implemented directly inside `src/evaluate.py` (no extra files).
   - After the CNN predicts \( \hat{y} \), we fit a simple correction on a calibration set:
     \( y \approx a \hat{y} + b \)
   - This reduces systematic bias (too strong/weak winds, offsets).

---

## Run the Python pipeline

### Option A: Run in WSL (Linux terminal)
From `.../weather-ml-project`:

```bash
python3 -m pip install -r requirements.txt
python3 ./main.py
```

Important: in WSL, use Linux paths: `python3 ./main.py` (not `.\main.py`).

### Option B: Run in Windows PowerShell (recommended)

```powershell
cd C:\Users\tzoan\Desktop\Weather_Forecasting\weather-ml-project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python .\main.py
```

---

## Useful “run one step” commands

From `weather-ml-project/`:

```bash
python3 ./src/download_gfs.py
python3 ./src/preprocess.py
python3 ./src/train.py
python3 ./src/evaluate.py
```

Outputs:
- `data/raw/` = downloaded GRIB2 (wind-only)
- `data/processed/` = Belgium NetCDF files
- `saved_models/` = trained CNN + postprocessing correction JSON
- `outputs/` = metrics JSON + maps PNGs

---

## Run the frontend (optional)

```powershell
cd C:\Users\tzoan\Desktop\Weather_Forecasting\frontend
npm install
npm run dev
```