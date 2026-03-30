"""
run_postprocessing.py
=====================
Main postprocessing runner.  Executes AFTER training (src/train.py).

Pipeline
--------
1. Load the trained CNN model (saved_models/wind_forecast_cnn.pth).
2. Rebuild the full dataset with the same chronological split as train.py.
3. Sub-split the validation set into:
      - fitting set  (first FIT_RATIO of val samples) → train postprocessors
      - test   set   (remaining val samples)          → evaluate improvement
4. Fit two postprocessors on the fitting set:
      a) MeanBiasCorrection   (Vannitsem et al. 2021, simplest method)
      b) LinearCalibration    (MOS-style, Glahn & Lowry 1972 as cited in paper)
5. Run all three variants (raw, bias-corrected, MOS) on the test set.
6. Print a comparison table and save the fitted postprocessors to disk.

Why this split?
---------------
Postprocessors must be fitted on samples the MAIN MODEL has already seen
(validation set) but evaluated on samples neither the model NOR the
postprocessor has seen before (test portion of validation).  Using the
same samples for fitting and testing would give over-optimistic results.

References
----------
Vannitsem et al. (2021). Statistical Postprocessing for Weather Forecasts.
    BAMS, March 2021, E681-E699.
Glahn & Lowry (1972). The use of model output statistics (MOS) in
    objective weather forecasting.  J. Appl. Meteor., 11, 1203-1211.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------
# Make src/ importable when run directly
# ------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import BetterWindCNN
from dataset import WindForecastDataset, load_wind_time_series
from postprocessing.bias_correction   import MeanBiasCorrection
from postprocessing.linear_calibration import LinearCalibration

# ------------------------------------------------------------------
# Configuration — must match train.py
# ------------------------------------------------------------------
INPUT_STEPS    = 2
TARGET_OFFSET  = 1
TRAIN_RATIO    = 0.8    # same chronological split as train.py
FIT_RATIO      = 0.5    # fraction of *validation* set used to FIT postprocessors
                        # the remaining (1 - FIT_RATIO) is the true test set
MODEL_FILENAME = "wind_forecast_cnn.pth"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Device: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def _predict(model, dataset, index, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) for one dataset sample."""
    x, y_true = dataset[index]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(x).squeeze(0).cpu().numpy()
    return y_true.numpy(), y_pred


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return mse, mae


def _summarise(label: str, mse_list: list, mae_list: list) -> None:
    mse = np.array(mse_list)
    mae = np.array(mae_list)
    print(
        f"  {label:<22}  "
        f"MSE  mean={mse.mean():.4f}  std={mse.std():.4f}  "
        f"min={mse.min():.4f}  max={mse.max():.4f}  |  "
        f"MAE  mean={mae.mean():.4f}  std={mae.std():.4f}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data"    / "processed"
    model_path    = project_root / "saved_models" / MODEL_FILENAME
    pp_dir        = project_root / "saved_models" / "postprocessing"
    pp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Postprocessing pipeline")
    print("=" * 64)
    print(f"  Model       : {model_path}")
    print(f"  Data        : {processed_dir}")
    print(f"  PP save dir : {pp_dir}")

    # 1. Device & dataset
    device = _get_device()
    data, _, latitudes, longitudes = load_wind_time_series(processed_dir)
    dataset = WindForecastDataset(data=data, input_steps=INPUT_STEPS,
                                  target_offset=TARGET_OFFSET)
    n = len(dataset)
    print(f"\n  Total dataset samples : {n}")

    # 2. Chronological split (same as train.py)
    train_size = int(TRAIN_RATIO * n)
    if train_size >= n:
        train_size = n - 1
    val_indices = list(range(train_size, n))
    print(f"  Training samples      : {train_size}")
    print(f"  Validation samples    : {len(val_indices)}")

    # 3. Sub-split validation → fitting set + test set
    fit_size  = max(1, int(FIT_RATIO * len(val_indices)))
    fit_indices  = val_indices[:fit_size]
    test_indices = val_indices[fit_size:]

    if len(test_indices) == 0:
        print("\n  WARNING: test set is empty — increase validation data or "
              "reduce FIT_RATIO.")
        return

    print(f"  PP fitting  samples   : {len(fit_indices)}")
    print(f"  PP test     samples   : {len(test_indices)}")

    # 4. Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    sample_x, sample_y = dataset[0]
    model = BetterWindCNN(
        in_channels=sample_x.shape[0],
        out_channels=sample_y.shape[0]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("\n  Model loaded successfully.")

    # 5. Collect raw predictions on the fitting set
    print("\n  Running inference on fitting set ...")
    fit_true, fit_pred = [], []
    for idx in fit_indices:
        yt, yp = _predict(model, dataset, idx, device)
        fit_true.append(yt)
        fit_pred.append(yp)

    # 6. Fit postprocessors
    print("\n  Fitting postprocessors ...")

    bc = MeanBiasCorrection()
    bc.fit(fit_true, fit_pred)
    bc.save(pp_dir / "bias_correction.npy")

    lc = LinearCalibration()
    lc.fit(fit_true, fit_pred)
    lc.save(pp_dir / "linear_calibration.npz")

    # 7. Evaluate on test set
    print("\n  Evaluating on test set ...")
    raw_mse, raw_mae     = [], []
    bc_mse,  bc_mae      = [], []
    lc_mse,  lc_mae      = [], []

    for idx in test_indices:
        yt, yp = _predict(model, dataset, idx, device)

        m, a = _metrics(yt, yp)
        raw_mse.append(m); raw_mae.append(a)

        m, a = _metrics(yt, bc.apply(yp))
        bc_mse.append(m);  bc_mae.append(a)

        m, a = _metrics(yt, lc.apply(yp))
        lc_mse.append(m);  lc_mae.append(a)

    # 8. Print comparison table
    print(f"\n  {'Method':<22}  {'MSE':^50}  {'MAE':^40}")
    print("  " + "-" * 110)
    _summarise("Raw CNN",            raw_mse, raw_mae)
    _summarise("+ Bias Correction",  bc_mse,  bc_mae)
    _summarise("+ Linear Calib(MOS)",lc_mse,  lc_mae)

    # 9. Print improvement
    raw_mean_mse = np.mean(raw_mse)
    bc_improvement  = 100 * (raw_mean_mse - np.mean(bc_mse))  / raw_mean_mse
    lc_improvement  = 100 * (raw_mean_mse - np.mean(lc_mse))  / raw_mean_mse
    print(f"\n  MSE improvement — Bias Correction : {bc_improvement:+.1f}%")
    print(f"  MSE improvement — Linear Calib    : {lc_improvement:+.1f}%")
    print("\n  Postprocessing finished.")


if __name__ == "__main__":
    main()
