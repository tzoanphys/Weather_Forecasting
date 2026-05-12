from pathlib import Path
import urllib.request
import zipfile
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import BetterWindCNN
from dataset import WindForecastDataset, load_wind_time_series

INPUT_STEPS    = 2
TARGET_OFFSET  = 1
MODEL_FILENAME = "wind_forecast_cnn.pth"
POSTPROCESS_FILENAME = "wind_postprocess_affine.json"


# ============================================================
# Postprocessing (ML correction) — kept inside this file
# ============================================================
#
# Purpose:
# The CNN prediction can have a systematic bias (too strong/weak winds).
# We add a *very simple* ML correction:
#
#     corrected = a * raw_prediction + b
#
# We fit (a, b) on a calibration set and then report corrected metrics.
# This is interview-friendly because it is explainable and small.
# ============================================================

def split_calibration_and_test_indices(
    indices: list[int],
    calibration_fraction: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Chronological split: earlier indices calibrate, later indices test."""
    if not indices:
        return [], []
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be between 0 and 1")
    # Need at least 2 validation samples to hold out a test set; otherwise
    # calibration and test would be the same single index (no crash).
    if len(indices) == 1:
        return [indices[0]], [indices[0]]
    cut = int(round(len(indices) * calibration_fraction))
    cut = max(1, min(cut, len(indices) - 1))
    return indices[:cut], indices[cut:]


def _fit_affine_1d(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Fit y_true ≈ a*y_pred + b in least squares (closed-form).
    Returns (a, b).
    """
    pred = y_pred.reshape(-1).astype(np.float64)
    true = y_true.reshape(-1).astype(np.float64)

    pred_mean = float(pred.mean())
    true_mean = float(true.mean())
    pred_var = float(pred.var())

    if pred_var < 1e-12:
        a = 1.0
        b = true_mean - pred_mean
        return a, b

    cov = float(np.mean((pred - pred_mean) * (true - true_mean)))
    a = cov / pred_var
    b = true_mean - a * pred_mean
    return float(a), float(b)


def fit_postprocessor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Fit affine correction separately for u10 and v10.
    Expects y_true/y_pred shaped [N, 2, lat, lon].
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 4 or y_true.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2,lat,lon], got {y_true.shape}")

    a_u, b_u = _fit_affine_1d(y_true[:, 0], y_pred[:, 0])
    a_v, b_v = _fit_affine_1d(y_true[:, 1], y_pred[:, 1])
    return {"u10": {"a": a_u, "b": b_u}, "v10": {"a": a_v, "b": b_v}}


def apply_postprocessor(y_pred: np.ndarray, post: dict) -> np.ndarray:
    """Apply affine correction to one sample shaped [2,lat,lon]."""
    out = np.empty_like(y_pred, dtype=np.float32)
    out[0] = post["u10"]["a"] * y_pred[0] + post["u10"]["b"]
    out[1] = post["v10"]["a"] * y_pred[1] + post["v10"]["b"]
    return out


def save_postprocessor(post: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(post, indent=2), encoding="utf-8")


def load_postprocessor(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple GPU (MPS)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def build_dataset(processed_dir: Path):
    data, _, latitudes, longitudes = load_wind_time_series(processed_dir)
    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET,
    )
    print(f"\nStacked time-series shape:\n{data.shape}")
    print(f"\nNumber of dataset samples created:\n{len(dataset)}")
    return dataset, latitudes, longitudes


# ============================================================
# Configuration
# ============================================================

TRAIN_RATIO  = 0.8   # must match train.py


def _chronological_train_size(n: int) -> int:
    """Same train/val boundary as train.py (chronological forecasting split)."""
    if n < 2:
        raise ValueError("Need at least 2 samples to create train/validation splits.")
    train_size = int(TRAIN_RATIO * n)
    if train_size == 0:
        train_size = 1
    if train_size == n:
        train_size = n - 1
    return train_size


# ============================================================
# Paths
# ============================================================

def get_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "saved_models" / MODEL_FILENAME
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, model_path, output_dir


# ============================================================
# Model
# ============================================================

def load_model(
    model_path: Path,
    dataset: WindForecastDataset,
    device: torch.device
) -> BetterWindCNN:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sample_x, sample_y = dataset[0]
    in_channels  = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    model = BetterWindCNN(in_channels=in_channels, out_channels=out_channels).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"\nLoaded model from: {model_path}")
    return model


# ============================================================
# Prediction
# ============================================================

def predict_one_sample(
    model: BetterWindCNN,
    dataset: WindForecastDataset,
    sample_index: int,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range. "
            f"Valid range is 0 to {len(dataset) - 1}."
        )

    x, y_true = dataset[sample_index]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x)

    return y_true.cpu().numpy(), y_pred.squeeze(0).cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    return mse, mae



# ============================================================
# Plotting
# ============================================================

def _load_natural_earth_countries() -> gpd.GeoDataFrame:
    """
    Download and cache the Natural Earth 50m countries shapefile.
    Stored in the project data/processed directory.
    """
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "natural_earth"
    shp_path = cache_dir / "ne_50m_admin_0_countries.shp"

    if not shp_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
        zip_path = cache_dir / "ne_50m_admin_0_countries.zip"
        print("Downloading Natural Earth 50m countries (one-time)...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
        zip_path.unlink()
        print("Done.")

    return gpd.read_file(shp_path)


# Dark navy background matching professional NWP style
_BG_COLOR = "#0b0e1a"

# Larger typography for readability in exported PNGs (dashboard viewing)
_MAP_TICK_FONTSIZE = 18
_MAP_LABEL_FONTSIZE = 24
_MAP_TITLE_FONTSIZE = 26
_MAP_SUPTITLE_FONTSIZE = 30
_CB_LABEL_FONTSIZE = 20
_CB_TICK_FONTSIZE = 18

# Also raise default matplotlib fonts as a baseline.
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": _MAP_TITLE_FONTSIZE,
        "axes.labelsize": _MAP_LABEL_FONTSIZE,
        "xtick.labelsize": _MAP_TICK_FONTSIZE,
        "ytick.labelsize": _MAP_TICK_FONTSIZE,
    }
)


def add_map_background(ax: plt.Axes, extent: list) -> None:
    """
    Draw a dark-style map background:
    - dark navy fill
    - thin white country borders
    - subtle white lat/lon gridlines
    """
    world = _load_natural_earth_countries()
    lon_min, lon_max, lat_min, lat_max = extent

    from shapely.geometry import box as shapely_box
    region = shapely_box(lon_min - 1, lat_min - 1, lon_max + 1, lat_max + 1)
    visible = world[world.geometry.intersects(region)]

    ax.set_facecolor(_BG_COLOR)

    # All country borders in white, Belgium slightly bolder
    visible[visible["NAME"] != "Belgium"].plot(
        ax=ax, color="none", edgecolor="white", linewidth=0.6, zorder=4, alpha=0.7
    )
    visible[visible["NAME"] == "Belgium"].plot(
        ax=ax, color="none", edgecolor="white", linewidth=1.8, zorder=5
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Subtle lat/lon grid
    lon_ticks = np.arange(int(np.ceil(lon_min)), int(np.floor(lon_max)) + 1, 1)
    lat_ticks = np.arange(int(np.ceil(lat_min)), int(np.floor(lat_max)) + 1, 1)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.grid(True, color="white", linewidth=0.3, linestyle="-", alpha=0.25, zorder=6)
    ax.tick_params(labelsize=_MAP_TICK_FONTSIZE, colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")


def _add_streamlines(ax: plt.Axes, u: np.ndarray, v: np.ndarray,
                     longitudes: np.ndarray, latitudes: np.ndarray) -> None:
    """
    Overlay smooth white streamlines showing wind flow direction.
    streamplot requires strictly increasing y (latitudes), so flip if needed.
    """
    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        u = u[::-1, :]
        v = v[::-1, :]

    speed = np.sqrt(u ** 2 + v ** 2)
    speed_norm = speed / (speed.max() + 1e-6)  # 0-1 for linewidth scaling
    ax.streamplot(
        longitudes, latitudes, u, v,
        color="white",
        linewidth=0.6 + 1.4 * speed_norm,
        density=1.8,
        arrowsize=0.8,
        arrowstyle="->",
        zorder=7,
    )


def plot_prediction_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_dir: Path,
    sample_index: int,
    tag: str = "raw",
) -> None:
    """
    Plot true and predicted wind speed (magnitude) side-by-side.
    Style: dark background, viridis colormap, smooth bicubic rendering,
    white streamlines for wind flow direction.
    """
    extent = [
        longitudes.min(), longitudes.max(),
        latitudes.min(), latitudes.max()
    ]

    true_u10 = y_true[0]
    true_v10 = y_true[1]
    pred_u10 = y_pred[0]
    pred_v10 = y_pred[1]

    true_speed = np.sqrt(true_u10 ** 2 + true_v10 ** 2)
    pred_speed = np.sqrt(pred_u10 ** 2 + pred_v10 ** 2)

    # Shared color range so both panels are comparable
    vmax = max(true_speed.max(), pred_speed.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(_BG_COLOR)
    cmap = "viridis"

    for ax in axes:
        add_map_background(ax, extent)

    # --- True wind speed ---
    im0 = axes[0].imshow(
        true_speed, origin="lower", extent=extent, aspect="auto",
        alpha=0.92, zorder=2, cmap=cmap, vmin=0, vmax=vmax,
        interpolation="bicubic"
    )
    _add_streamlines(axes[0], true_u10, true_v10, longitudes, latitudes)
    axes[0].set_title("True Wind Speed", fontsize=_MAP_TITLE_FONTSIZE, fontweight="bold", pad=12)
    axes[0].set_xlabel("Longitude", fontsize=_MAP_LABEL_FONTSIZE)
    axes[0].set_ylabel("Latitude", fontsize=_MAP_LABEL_FONTSIZE)
    cb = plt.colorbar(im0, ax=axes[0], pad=0.02)
    cb.set_label("m/s", color="white", fontsize=_CB_LABEL_FONTSIZE)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=_CB_TICK_FONTSIZE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # --- Predicted wind speed ---
    im1 = axes[1].imshow(
        pred_speed, origin="lower", extent=extent, aspect="auto",
        alpha=0.92, zorder=2, cmap=cmap, vmin=0, vmax=vmax,
        interpolation="bicubic"
    )
    _add_streamlines(axes[1], pred_u10, pred_v10, longitudes, latitudes)
    axes[1].set_title(f"Predicted Wind Speed ({tag})", fontsize=_MAP_TITLE_FONTSIZE, fontweight="bold", pad=12)
    axes[1].set_xlabel("Longitude", fontsize=_MAP_LABEL_FONTSIZE)
    axes[1].set_ylabel("Latitude", fontsize=_MAP_LABEL_FONTSIZE)
    cb = plt.colorbar(im1, ax=axes[1], pad=0.02)
    cb.set_label("m/s", color="white", fontsize=_CB_LABEL_FONTSIZE)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=_CB_TICK_FONTSIZE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Wind Speed Forecast — Belgium — Sample {sample_index} — {tag}",
        fontsize=_MAP_SUPTITLE_FONTSIZE, fontweight="bold", color="white", y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / f"evaluation_sample_{sample_index}_{tag}_belgium_only.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved prediction maps to: {output_path}")


def plot_error_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_dir: Path,
    sample_index: int,
    tag: str = "raw",
) -> None:
    extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]

    true_speed  = np.sqrt(y_true[0] ** 2 + y_true[1] ** 2)
    pred_speed  = np.sqrt(y_pred[0] ** 2 + y_pred[1] ** 2)
    speed_error = np.abs(pred_speed - true_speed)
    error_u10   = np.abs(y_pred[0] - y_true[0])
    error_v10   = np.abs(y_pred[1] - y_true[1])

    panels = [
        (speed_error, "Wind Speed Error"),
        (error_u10,   "Absolute Error — u10"),
        (error_v10,   "Absolute Error — v10"),
    ]
    error_max = max(d.max() for d, _ in panels)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(_BG_COLOR)

    for ax, (data, title) in zip(axes, panels):
        add_map_background(ax, extent)
        im = ax.imshow(
            data, origin="lower", extent=extent, aspect="auto",
            alpha=0.92, zorder=2, cmap="viridis", vmin=0, vmax=error_max,
            interpolation="bicubic"
        )
        ax.set_title(title, fontsize=_MAP_TITLE_FONTSIZE, fontweight="bold", pad=10)
        ax.set_xlabel("Longitude", fontsize=_MAP_LABEL_FONTSIZE)
        ax.set_ylabel("Latitude", fontsize=_MAP_LABEL_FONTSIZE)
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("m/s", color="white", fontsize=_CB_LABEL_FONTSIZE)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=_CB_TICK_FONTSIZE)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Error Maps — Belgium — Sample {sample_index} — {tag}",
        fontsize=_MAP_SUPTITLE_FONTSIZE - 1, fontweight="bold", color="white", y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / f"error_maps_sample_{sample_index}_{tag}_belgium_only.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved error maps to: {output_path}")

# ============================================================
# Bulk validation evaluation
# ============================================================

def evaluate_validation_set(
    model: BetterWindCNN,
    dataset: WindForecastDataset,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """
    Run inference on every validation sample (chronological split)
    and return per-sample MSE and MAE lists (same order as validation indices).
    """
    n = len(dataset)
    train_size = _chronological_train_size(n)
    val_indices = list(range(train_size, n))

    mse_list, mae_list = [], []
    for idx in val_indices:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        mse, mae = compute_metrics(y_true, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)

    return mse_list, mae_list


def _gather_predictions(
    model: BetterWindCNN,
    dataset: WindForecastDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect y_true and y_pred arrays for a list of dataset indices.
    Returns arrays shaped [N, 2, lat, lon].
    """
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for idx in indices:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        y_true_list.append(y_true.astype(np.float32))
        y_pred_list.append(y_pred.astype(np.float32))

    return np.stack(y_true_list, axis=0), np.stack(y_pred_list, axis=0)


def _metrics_over_indices(
    model: BetterWindCNN,
    dataset: WindForecastDataset,
    indices: list[int],
    device: torch.device,
    postprocessor: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample MSE/MAE for a set of indices.
    If postprocessor is provided, it is applied to y_pred before metrics.
    """
    mse_list: list[float] = []
    mae_list: list[float] = []

    for idx in indices:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        if postprocessor is not None:
            y_pred = apply_postprocessor(y_pred, postprocessor)
        mse, mae = compute_metrics(y_true, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)

    return np.array(mse_list, dtype=np.float64), np.array(mae_list, dtype=np.float64)


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    processed_dir, model_path, output_dir = get_paths()
    postprocess_path = Path(__file__).resolve().parent.parent / "saved_models" / POSTPROCESS_FILENAME

    print(f"\nUsing data from   : {processed_dir}")
    print(f"Loading model from: {model_path}")
    print(f"Saving figures to : {output_dir}")

    dataset, latitudes, longitudes = build_dataset(processed_dir)
    model = load_model(model_path, dataset, device)

    # ----------------------------------------------------------
    # 1) Aggregate metrics over all validation samples
    # ----------------------------------------------------------
    print("\nEvaluating all validation samples...")
    mse_list, mae_list = evaluate_validation_set(model, dataset, device)

    mse_arr = np.array(mse_list)
    mae_arr = np.array(mae_list)

    if mse_arr.size == 0:
        raise ValueError(
            "No validation samples to evaluate. "
            "You need more time steps / dataset samples than the training window."
        )

    train_size = _chronological_train_size(len(dataset))
    val_indices = list(range(train_size, len(dataset)))

    print(f"\nValidation samples evaluated : {len(mse_list)}")
    print(f"MSE  — mean : {mse_arr.mean():.4f}  std : {mse_arr.std():.4f}  "
          f"min : {mse_arr.min():.4f}  max : {mse_arr.max():.4f}")
    print(f"MAE  — mean : {mae_arr.mean():.4f}  std : {mae_arr.std():.4f}  "
          f"min : {mae_arr.min():.4f}  max : {mae_arr.max():.4f}")

    # Best / worst map to the same order as val_indices (mse_list[i] == sample val_indices[i])
    k_best = int(np.argmin(mse_arr))
    k_worst = int(np.argmax(mse_arr))
    best_idx = val_indices[k_best]
    worst_idx = val_indices[k_worst]
    print(f"Best  sample (lowest  MSE): dataset index {best_idx}  MSE={mse_arr[k_best]:.6f}")
    print(f"Worst sample (highest MSE): dataset index {worst_idx}  MSE={mse_arr[k_worst]:.6f}")

    # ----------------------------------------------------------
    # 2) Postprocessing correction (ML calibration)
    # ----------------------------------------------------------
    calib_indices, test_indices = split_calibration_and_test_indices(val_indices, calibration_fraction=0.5)
    print("\nPostprocessing (affine bias/scale correction)")
    print(f"Validation split: {len(val_indices)} samples total")
    print(f"  Calibration set: {len(calib_indices)} samples (fit correction)")
    print(f"  Test set       : {len(test_indices)} samples (report corrected metrics)")

    post_stale = (
        postprocess_path.exists()
        and model_path.exists()
        and model_path.stat().st_mtime > postprocess_path.stat().st_mtime
    )

    if postprocess_path.exists() and not post_stale:
        post = load_postprocessor(postprocess_path)
        print(f"Loaded existing postprocessor: {postprocess_path}")
    else:
        if post_stale:
            print("Model is newer than saved postprocessor; refitting correction.")
        y_true_calib, y_pred_calib = _gather_predictions(model, dataset, calib_indices, device)
        post = fit_postprocessor(y_true=y_true_calib, y_pred=y_pred_calib)
        save_postprocessor(post, postprocess_path)
        print(f"Fitted and saved postprocessor: {postprocess_path}")

    raw_mse, raw_mae = _metrics_over_indices(model, dataset, test_indices, device, postprocessor=None)
    cor_mse, cor_mae = _metrics_over_indices(model, dataset, test_indices, device, postprocessor=post)

    print("\nPostprocessing results (on postprocess test set)")
    print(f"RAW      MSE mean: {raw_mse.mean():.4f}   MAE mean: {raw_mae.mean():.4f}")
    print(f"CORRECT  MSE mean: {cor_mse.mean():.4f}   MAE mean: {cor_mae.mean():.4f}")
    print(f"Δ (improvement)  : {(raw_mse.mean() - cor_mse.mean()):.4f} MSE, {(raw_mae.mean() - cor_mae.mean()):.4f} MAE")

    summary = {
        "train_ratio": float(TRAIN_RATIO),
        "num_validation_samples": int(len(mse_list)),
        "mse": {
            "mean": float(mse_arr.mean()),
            "std": float(mse_arr.std()),
            "min": float(mse_arr.min()),
            "max": float(mse_arr.max()),
        },
        "mae": {
            "mean": float(mae_arr.mean()),
            "std": float(mae_arr.std()),
            "min": float(mae_arr.min()),
            "max": float(mae_arr.max()),
        },
        "best": {
            "dataset_index": int(best_idx),
            "evaluation_image_raw": f"evaluation_sample_{best_idx}_raw_belgium_only.png",
            "error_map_image_raw": f"error_maps_sample_{best_idx}_raw_belgium_only.png",
            "evaluation_image_corrected": f"evaluation_sample_{best_idx}_corrected_belgium_only.png",
            "error_map_image_corrected": f"error_maps_sample_{best_idx}_corrected_belgium_only.png",
        },
        "worst": {
            "dataset_index": int(worst_idx),
            "evaluation_image_raw": f"evaluation_sample_{worst_idx}_raw_belgium_only.png",
            "error_map_image_raw": f"error_maps_sample_{worst_idx}_raw_belgium_only.png",
            "evaluation_image_corrected": f"evaluation_sample_{worst_idx}_corrected_belgium_only.png",
            "error_map_image_corrected": f"error_maps_sample_{worst_idx}_corrected_belgium_only.png",
        },
        "postprocessing": {
            "type": "affine_per_component",
            "path": str(postprocess_path),
            "calibration_fraction_of_validation": 0.5,
            "test_set_size": int(len(test_indices)),
            "raw_test": {
                "mse_mean": float(raw_mse.mean()),
                "mae_mean": float(raw_mae.mean()),
            },
            "corrected_test": {
                "mse_mean": float(cor_mse.mean()),
                "mae_mean": float(cor_mae.mean()),
            },
        },
    }

    # ----------------------------------------------------------
    # 3) Plot maps for the best and worst validation samples
    # ----------------------------------------------------------
    for label, idx in [("best", best_idx), ("worst", worst_idx)]:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        y_pred_corr = apply_postprocessor(y_pred, post)
        plot_prediction_maps(
            y_true=y_true, y_pred=y_pred,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=idx, tag="raw"
        )
        plot_error_maps(
            y_true=y_true, y_pred=y_pred,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=idx, tag="raw"
        )
        # Also save corrected plots for a clean interview story: "ML + calibration"
        plot_prediction_maps(
            y_true=y_true, y_pred=y_pred_corr,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=int(idx), tag="corrected"
        )
        plot_error_maps(
            y_true=y_true, y_pred=y_pred_corr,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=int(idx), tag="corrected"
        )
        print(f"Plots saved for {label} sample (index {idx}).")

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to: {summary_path}")


if __name__ == "__main__":
    main()