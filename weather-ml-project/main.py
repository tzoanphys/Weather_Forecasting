"""
main.py — Single entry point for the full weather ML pipeline.

Steps
-----
1. Download   : fetch raw GFS GRIB2 files from NOAA S3
2. Plots      : generate exploratory wind plots from the processed NetCDF
3. Preprocess : convert raw GRIB2 files to Belgium NetCDF subsets
4. Dataset    : load the processed files and print a data summary
5. Model      : print the CNN architecture
6. Train      : train the BetterWindCNN and save weights
7. Evaluate   : run inference, plot prediction maps and error maps
"""

import runpy
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make src/ importable regardless of how the script is launched
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _banner(step: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {step}")
    print(f"{'=' * 64}\n")


def _run_script(step: str, script: Path) -> None:
    """Execute a standalone script in-process via runpy."""
    _banner(step)
    runpy.run_path(str(script), run_name="__main__")


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def main() -> None:

    # 1 ─ Download raw GFS data ----------------------------------------
    _run_script("STEP 1 — Download raw GFS data", SRC / "download_gfs.py")

    # 2 ─ Exploratory plots --------------------------------------------
    _run_script("STEP 2 — Generate EDA wind plots", SRC / "download_plots.py")

    # 3 ─ Preprocess ---------------------------------------------------
    _run_script("STEP 3 — Preprocess GRIB2 → Belgium NetCDF", SRC / "preprocess.py")

    # 4 ─ Dataset summary ----------------------------------------------
    _banner("STEP 4 — Dataset summary")
    from dataset import load_wind_time_series, WindForecastDataset

    processed_dir = ROOT / "data" / "processed"
    data, files, latitudes, longitudes = load_wind_time_series(processed_dir)

    print(f"  Files loaded      : {len(files)}")
    print(f"  Time steps        : {data.shape[0]}")
    print(f"  Channels (u10/v10): {data.shape[1]}")
    print(f"  Spatial grid      : {data.shape[2]} lat × {data.shape[3]} lon")
    print(f"  Latitude range    : {latitudes.min():.2f} → {latitudes.max():.2f} °N")
    print(f"  Longitude range   : {longitudes.min():.2f} → {longitudes.max():.2f} °E")

    # 5 ─ Model architecture -------------------------------------------
    _banner("STEP 5 — Model architecture")
    from model import BetterWindCNN

    INPUT_STEPS = 2          # must match train.py INPUT_STEPS
    preview_model = BetterWindCNN(in_channels=INPUT_STEPS * 2, out_channels=2)
    print(preview_model)

    # 6 ─ Train --------------------------------------------------------
    _banner("STEP 6 — Train")
    from train import train_model
    train_model()

    # 7 ─ Evaluate -----------------------------------------------------
    _banner("STEP 7 — Evaluate")
    from evaluate import main as evaluate_main
    evaluate_main()

    # Done
    _banner("Pipeline complete! All outputs saved to outputs/")


if __name__ == "__main__":
    main()
