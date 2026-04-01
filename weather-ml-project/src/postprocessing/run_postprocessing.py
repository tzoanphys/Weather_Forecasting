"""
run_postprocessing.py
=====================

This script runs a simple postprocessing study AFTER the CNN model has already
been trained in train.py.

Main idea
---------
The CNN gives a raw prediction, but that prediction may still contain
systematic errors. In weather forecasting, it is common to apply an extra
correction step after the main model. This extra step is called
postprocessing.

In this script, we test whether two simple postprocessing methods can improve
the raw CNN predictions:

1. MeanBiasCorrection
   Learns the average error of the CNN and adds it back as a correction.

2. LinearCalibration
   Learns a simple linear relationship between prediction and truth:
       y_true ≈ a * y_pred + b
   This is similar to MOS-style calibration.

What this script does
---------------------
1. Load the trained CNN model.
2. Rebuild the dataset using the same chronological split as train.py.
3. Take the validation part of the dataset and split it again into:
      - fitting set: used to learn the postprocessing corrections
      - test set: used to check whether the corrections really help
4. Fit the two postprocessing methods on the fitting set.
5. Compare three versions on the test set:
      - raw CNN prediction
      - bias-corrected prediction
      - linearly calibrated prediction
6. Print the results and save the fitted postprocessors.

Why do we split the validation set again?
-----------------------------------------
Because a postprocessor must not be evaluated on the same samples it was fitted on.

If we used the same samples both to fit and to test the postprocessor,
the results would look better than they really are.

So:
- fitting set  -> used to learn the correction
- test set     -> used to evaluate the correction fairly

References
----------
Vannitsem et al. (2021). Statistical Postprocessing for Weather Forecasts.
Glahn & Lowry (1972). The use of model output statistics (MOS) in objective
weather forecasting.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------
# Make the src/ folder importable when this script is run directly
# ------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import BetterWindCNN
from dataset import WindForecastDataset, load_wind_time_series
from postprocessing.bias_correction import MeanBiasCorrection
from postprocessing.linear_calibration import LinearCalibration

# ------------------------------------------------------------------
# Configuration
# These values must match the ones used in train.py
# ------------------------------------------------------------------
INPUT_STEPS = 2
TARGET_OFFSET = 1
TRAIN_RATIO = 0.8   # same chronological train split as in train.py
FIT_RATIO = 0.5     # use half of the validation set to fit postprocessors
                    # the rest becomes the final test set
MODEL_FILENAME = "wind_forecast_cnn.pth"


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _get_device() -> torch.device:
    """
    Choose the device used for inference.

    If a CUDA GPU is available, use it.
    Otherwise, fall back to CPU.
    """
    if torch.cuda.is_available():
        print(f"Device: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")

    print("Device: CPU")
    return torch.device("cpu")


def _predict(model, dataset, index, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on one dataset sample.

    Returns
    -------
    y_true : np.ndarray
        The true target for this sample.
    y_pred : np.ndarray
        The CNN prediction for this sample.
    """
    x, y_true = dataset[index]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x).squeeze(0).cpu().numpy()

    return y_true.numpy(), y_pred


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute two simple regression metrics:

    - MSE: mean squared error
    - MAE: mean absolute error
    """
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return mse, mae


def _summarise(label: str, mse_list: list, mae_list: list) -> None:
    """
    Print a compact summary of the errors for one method.
    """
    mse = np.array(mse_list)
    mae = np.array(mae_list)

    print(
        f"  {label:<22}  "
        f"MSE  mean={mse.mean():.4f}  std={mse.std():.4f}  "
        f"min={mse.min():.4f}  max={mse.max():.4f}  |  "
        f"MAE  mean={mae.mean():.4f}  std={mae.std():.4f}"
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main() -> None:
    """
    Run the full postprocessing experiment.

    Steps
    -----
    1. Load data and trained model
    2. Recreate the same chronological split used in training
    3. Split validation data into:
         - postprocessing fitting set
         - postprocessing test set
    4. Fit the two postprocessing methods
    5. Evaluate raw CNN vs corrected versions
    6. Print and save results
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "saved_models" / MODEL_FILENAME
    pp_dir = project_root / "saved_models" / "postprocessing"
    pp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Postprocessing pipeline")
    print("=" * 64)
    print(f"  Model       : {model_path}")
    print(f"  Data        : {processed_dir}")
    print(f"  PP save dir : {pp_dir}")

    # 1. Load device and dataset
    device = _get_device()
    data, _, latitudes, longitudes = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET
    )

    n = len(dataset)
    print(f"\n  Total dataset samples : {n}")

    # 2. Recreate the same chronological split as in train.py
    train_size = int(TRAIN_RATIO * n)
    if train_size >= n:
        train_size = n - 1

    val_indices = list(range(train_size, n))

    print(f"  Training samples      : {train_size}")
    print(f"  Validation samples    : {len(val_indices)}")

    # 3. Split validation data into:
    #    - fitting set for learning the postprocessing
    #    - test set for fair evaluation
    fit_size = max(1, int(FIT_RATIO * len(val_indices)))
    fit_indices = val_indices[:fit_size]
    test_indices = val_indices[fit_size:]

    if len(test_indices) == 0:
        print(
            "\n  WARNING: test set is empty — increase validation data or "
            "reduce FIT_RATIO."
        )
        return

    print(f"  PP fitting  samples   : {len(fit_indices)}")
    print(f"  PP test     samples   : {len(test_indices)}")

    # 4. Load the trained CNN model
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

    # 5. Run the CNN on the fitting set
    #    We need predictions + truths in order to fit the postprocessors
    print("\n  Running inference on fitting set ...")
    fit_true, fit_pred = [], []

    for idx in fit_indices:
        yt, yp = _predict(model, dataset, idx, device)
        fit_true.append(yt)
        fit_pred.append(yp)

    # 6. Fit the two postprocessing methods
    print("\n  Fitting postprocessors ...")

    bc = MeanBiasCorrection()
    bc.fit(fit_true, fit_pred)
    bc.save(pp_dir / "bias_correction.npy")

    lc = LinearCalibration()
    lc.fit(fit_true, fit_pred)
    lc.save(pp_dir / "linear_calibration.npz")

    # 7. Evaluate all methods on the test set
    print("\n  Evaluating on test set ...")

    raw_mse, raw_mae = [], []
    bc_mse, bc_mae = [], []
    lc_mse, lc_mae = [], []

    for idx in test_indices:
        yt, yp = _predict(model, dataset, idx, device)

        # Raw CNN result
        m, a = _metrics(yt, yp)
        raw_mse.append(m)
        raw_mae.append(a)

        # Bias-corrected result
        m, a = _metrics(yt, bc.apply(yp))
        bc_mse.append(m)
        bc_mae.append(a)

        # Linearly calibrated result
        m, a = _metrics(yt, lc.apply(yp))
        lc_mse.append(m)
        lc_mae.append(a)

    # 8. Print a comparison table
    print(f"\n  {'Method':<22}  {'MSE':^50}  {'MAE':^40}")
    print("  " + "-" * 110)
    _summarise("Raw CNN", raw_mse, raw_mae)
    _summarise("+ Bias Correction", bc_mse, bc_mae)
    _summarise("+ Linear Calib(MOS)", lc_mse, lc_mae)

    # 9. Print percentage improvement in mean MSE
    raw_mean_mse = np.mean(raw_mse)
    bc_improvement = 100 * (raw_mean_mse - np.mean(bc_mse)) / raw_mean_mse
    lc_improvement = 100 * (raw_mean_mse - np.mean(lc_mse)) / raw_mean_mse

    print(f"\n  MSE improvement — Bias Correction : {bc_improvement:+.1f}%")
    print(f"  MSE improvement — Linear Calib    : {lc_improvement:+.1f}%")
    print("\n  Postprocessing finished.")


if __name__ == "__main__":
    main()