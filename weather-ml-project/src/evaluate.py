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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
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
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    ax.tick_params(labelsize=8, colors="white")
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
    sample_index: int
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
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
    axes[0].set_title("True Wind Speed", fontsize=13, fontweight="bold", pad=8)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cb = plt.colorbar(im0, ax=axes[0], pad=0.02)
    cb.set_label("m/s", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # --- Predicted wind speed ---
    im1 = axes[1].imshow(
        pred_speed, origin="lower", extent=extent, aspect="auto",
        alpha=0.92, zorder=2, cmap=cmap, vmin=0, vmax=vmax,
        interpolation="bicubic"
    )
    _add_streamlines(axes[1], pred_u10, pred_v10, longitudes, latitudes)
    axes[1].set_title("Predicted Wind Speed", fontsize=13, fontweight="bold", pad=8)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    cb = plt.colorbar(im1, ax=axes[1], pad=0.02)
    cb.set_label("m/s", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Wind Speed Forecast — Belgium — Sample {sample_index}",
        fontsize=15, fontweight="bold", color="white", y=1.01
    )
    plt.tight_layout()

    output_path = output_dir / f"evaluation_sample_{sample_index}_belgium_only.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved prediction maps to: {output_path}")


def plot_error_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_dir: Path,
    sample_index: int
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(_BG_COLOR)

    for ax, (data, title) in zip(axes, panels):
        add_map_background(ax, extent)
        im = ax.imshow(
            data, origin="lower", extent=extent, aspect="auto",
            alpha=0.92, zorder=2, cmap="viridis", vmin=0, vmax=error_max,
            interpolation="bicubic"
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("m/s", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"Error Maps — Belgium — Sample {sample_index}",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )
    plt.tight_layout()

    output_path = output_dir / f"error_maps_sample_{sample_index}_belgium_only.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved error maps to: {output_path}")

# ============================================================
# Bulk validation evaluation
# ============================================================

def evaluate_validation_set(
    model: BetterWindCNN,
    dataset: WindForecastDataset,
    device: torch.device,
) -> tuple[list[float], list[float], int]:
    """
    Run inference on every validation sample (chronological split)
    and return per-sample MSE list, MAE list, and the first val index.
    """
    n = len(dataset)
    train_size = int(TRAIN_RATIO * n)
    if train_size == n:
        train_size = n - 1
    val_indices = list(range(train_size, n))

    mse_list, mae_list = [], []
    for idx in val_indices:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        mse, mae = compute_metrics(y_true, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)

    return mse_list, mae_list, val_indices[0]


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    processed_dir, model_path, output_dir = get_paths()

    print(f"\nUsing data from   : {processed_dir}")
    print(f"Loading model from: {model_path}")
    print(f"Saving figures to : {output_dir}")

    dataset, latitudes, longitudes = build_dataset(processed_dir)
    model = load_model(model_path, dataset, device)

    # ----------------------------------------------------------
    # Aggregate metrics over all validation samples
    # ----------------------------------------------------------
    print("\nEvaluating all validation samples...")
    mse_list, mae_list, first_val_idx = evaluate_validation_set(model, dataset, device)

    mse_arr = np.array(mse_list)
    mae_arr = np.array(mae_list)

    print(f"\nValidation samples evaluated : {len(mse_list)}")
    print(f"MSE  — mean : {mse_arr.mean():.4f}  std : {mse_arr.std():.4f}  "
          f"min : {mse_arr.min():.4f}  max : {mse_arr.max():.4f}")
    print(f"MAE  — mean : {mae_arr.mean():.4f}  std : {mae_arr.std():.4f}  "
          f"min : {mae_arr.min():.4f}  max : {mae_arr.max():.4f}")

    # Worst and best sample indices (relative to full dataset)
    best_idx  = first_val_idx + int(np.argmin(mse_arr))
    worst_idx = first_val_idx + int(np.argmax(mse_arr))
    print(f"Best  sample (lowest  MSE): dataset index {best_idx}")
    print(f"Worst sample (highest MSE): dataset index {worst_idx}")

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
            "evaluation_image": f"evaluation_sample_{best_idx}_belgium_only.png",
            "error_map_image": f"error_maps_sample_{best_idx}_belgium_only.png",
        },
        "worst": {
            "dataset_index": int(worst_idx),
            "evaluation_image": f"evaluation_sample_{worst_idx}_belgium_only.png",
            "error_map_image": f"error_maps_sample_{worst_idx}_belgium_only.png",
        },
    }

    # ----------------------------------------------------------
    # Plot maps for the best and worst validation samples
    # ----------------------------------------------------------
    for label, idx in [("best", best_idx), ("worst", worst_idx)]:
        y_true, y_pred = predict_one_sample(model, dataset, idx, device)
        plot_prediction_maps(
            y_true=y_true, y_pred=y_pred,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=idx
        )
        plot_error_maps(
            y_true=y_true, y_pred=y_pred,
            latitudes=latitudes, longitudes=longitudes,
            output_dir=output_dir, sample_index=idx
        )
        print(f"Plots saved for {label} sample (index {idx}).")

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to: {summary_path}")


if __name__ == "__main__":
    main()