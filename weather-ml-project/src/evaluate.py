from pathlib import Path
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch


from model import BetterWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# ============================================================
# Configuration
# ============================================================

INPUT_STEPS = 2
TARGET_OFFSET = 1
MODEL_FILENAME = "wind_forecast_cnn.pth"
SAMPLE_INDEX = 0


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    """
    Select the best available device.
    """
    if torch.backends.mps.is_available():
        print("Using Apple GPU with MPS.")
        return torch.device("mps")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using NVIDIA GPU: {gpu_name}")
        return torch.device("cuda")

    print("No GPU available. Using CPU.")
    return torch.device("cpu")


# ============================================================
# Paths
# ============================================================

def get_paths() -> tuple[Path, Path, Path]:
    """
    Return:
    - processed data directory
    - saved model path
    - output directory
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "saved_models" / MODEL_FILENAME
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, model_path, output_dir


# ============================================================
# Data
# ============================================================

def build_dataset(processed_dir: Path) -> tuple[WindForecastDataset, np.ndarray, np.ndarray]:
    """
    Load processed files and create dataset.
    Also return latitude and longitude for plotting.
    """
    data, _, latitudes, longitudes = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET
    )

    if len(dataset) == 0:
        raise ValueError("No samples were created from the dataset.")

    print(f"\nEvaluation dataset contains {len(dataset)} sample(s).")
    return dataset, latitudes, longitudes


# ============================================================
# Model
# ============================================================

def load_model(
    model_path: Path,
    dataset: WindForecastDataset,
    device: torch.device
) -> BetterWindCNN:
    """
    Rebuild the model and load trained weights.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sample_x, sample_y = dataset[0]
    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    model = BetterWindCNN(
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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
    """
    Predict one sample and return:
    - true target
    - predicted target
    """
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range. "
            f"Valid range is 0 to {len(dataset) - 1}."
        )

    x, y_true = dataset[sample_index]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.squeeze(0).cpu().numpy()

    return y_true_np, y_pred_np


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute MSE and MAE.
    """
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


def add_map_background(ax: plt.Axes, extent: list) -> None:
    """
    Draw a Natural Earth map background:
    - surrounding countries in semi-transparent light grey
    - Belgium with a bold black border
    - lat/lon gridlines
    """
    world = _load_natural_earth_countries()
    lon_min, lon_max, lat_min, lat_max = extent

    # Clip to a slightly larger area than the plot extent
    from shapely.geometry import box as shapely_box
    region = shapely_box(lon_min - 1, lat_min - 1, lon_max + 1, lat_max + 1)
    visible = world[world.geometry.intersects(region)]

    visible[visible["NAME"] != "Belgium"].plot(
        ax=ax, color="#d0d0d0", edgecolor="#777777", linewidth=0.7, zorder=3, alpha=0.45
    )
    visible[visible["NAME"] == "Belgium"].plot(
        ax=ax, color="none", edgecolor="black", linewidth=2.2, zorder=4
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Lat/lon gridlines
    lon_ticks = np.arange(int(np.ceil(lon_min)), int(np.floor(lon_max)) + 1, 1)
    lat_ticks = np.arange(int(np.ceil(lat_min)), int(np.floor(lat_max)) + 1, 1)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.grid(True, color="white", linewidth=0.5, linestyle="--", alpha=0.6, zorder=5)
    ax.tick_params(labelsize=7)


def plot_prediction_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_dir: Path,
    sample_index: int
) -> None:
    """
    Plot true and predicted u10 / v10 maps over the Belgium domain.
    - Shared symmetric color scale per variable for honest True vs Predicted comparison
    - Diverging colormap (RdBu_r): blue = wind blowing west/south, red = east/north
    - Quiver arrows showing wind direction and relative speed
    """
    extent = [
        longitudes.min(), longitudes.max(),
        latitudes.min(), latitudes.max()
    ]

    true_u10 = y_true[0]
    true_v10 = y_true[1]
    pred_u10 = y_pred[0]
    pred_v10 = y_pred[1]

    # Symmetric color range shared between True and Predicted for each component
    u_abs = max(abs(true_u10).max(), abs(pred_u10).max())
    v_abs = max(abs(true_v10).max(), abs(pred_v10).max())

    # Quiver arrow grid — subsample so arrows don't overlap
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    step = max(1, min(len(latitudes), len(longitudes)) // 8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = "RdBu_r"

    for ax in axes.flat:
        add_map_background(ax, extent)

    # --- True u10 ---
    im0 = axes[0, 0].imshow(true_u10, origin="lower", extent=extent, aspect="auto",
                             alpha=0.85, zorder=2, cmap=cmap, vmin=-u_abs, vmax=u_abs)
    axes[0, 0].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                      true_u10[::step, ::step], true_v10[::step, ::step],
                      scale=60, width=0.004, color="k", zorder=6)
    axes[0, 0].set_title("True u10", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    cb = plt.colorbar(im0, ax=axes[0, 0])
    cb.set_label("m/s")

    # --- Predicted u10 ---
    im1 = axes[0, 1].imshow(pred_u10, origin="lower", extent=extent, aspect="auto",
                             alpha=0.85, zorder=2, cmap=cmap, vmin=-u_abs, vmax=u_abs)
    axes[0, 1].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                      pred_u10[::step, ::step], pred_v10[::step, ::step],
                      scale=60, width=0.004, color="k", zorder=6)
    axes[0, 1].set_title("Predicted u10", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    cb = plt.colorbar(im1, ax=axes[0, 1])
    cb.set_label("m/s")

    # --- True v10 ---
    im2 = axes[1, 0].imshow(true_v10, origin="lower", extent=extent, aspect="auto",
                             alpha=0.85, zorder=2, cmap=cmap, vmin=-v_abs, vmax=v_abs)
    axes[1, 0].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                      true_u10[::step, ::step], true_v10[::step, ::step],
                      scale=60, width=0.004, color="k", zorder=6)
    axes[1, 0].set_title("True v10", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    cb = plt.colorbar(im2, ax=axes[1, 0])
    cb.set_label("m/s")

    # --- Predicted v10 ---
    im3 = axes[1, 1].imshow(pred_v10, origin="lower", extent=extent, aspect="auto",
                             alpha=0.85, zorder=2, cmap=cmap, vmin=-v_abs, vmax=v_abs)
    axes[1, 1].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                      pred_u10[::step, ::step], pred_v10[::step, ::step],
                      scale=60, width=0.004, color="k", zorder=6)
    axes[1, 1].set_title("Predicted v10", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    cb = plt.colorbar(im3, ax=axes[1, 1])
    cb.set_label("m/s")

    fig.suptitle(f"Forecast Evaluation - Belgium - Sample {sample_index}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / f"evaluation_sample_{sample_index}_belgium_only.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    """
    Plot absolute error maps over the Belgium domain only.
    """
    extent = [
        longitudes.min(), longitudes.max(),
        latitudes.min(), latitudes.max()
    ]

    error_u10 = np.abs(y_pred[0] - y_true[0])
    error_v10 = np.abs(y_pred[1] - y_true[1])

    # Shared color scale so both panels are directly comparable
    error_max = max(error_u10.max(), error_v10.max())

    # Quiver arrows showing true wind direction for spatial reference
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    step = max(1, min(len(latitudes), len(longitudes)) // 8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax in axes:
        add_map_background(ax, extent)

    im0 = axes[0].imshow(error_u10, origin="lower", extent=extent, aspect="auto",
                         alpha=0.85, zorder=2, cmap="YlOrRd", vmin=0, vmax=error_max)
    axes[0].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                   y_true[0][::step, ::step], y_true[1][::step, ::step],
                   scale=60, width=0.004, color="k", zorder=6)
    axes[0].set_title("Absolute Error - u10", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cb = plt.colorbar(im0, ax=axes[0])
    cb.set_label("m/s")

    im1 = axes[1].imshow(error_v10, origin="lower", extent=extent, aspect="auto",
                         alpha=0.85, zorder=2, cmap="YlOrRd", vmin=0, vmax=error_max)
    axes[1].quiver(lon_grid[::step, ::step], lat_grid[::step, ::step],
                   y_true[0][::step, ::step], y_true[1][::step, ::step],
                   scale=60, width=0.004, color="k", zorder=6)
    axes[1].set_title("Absolute Error - v10", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    cb = plt.colorbar(im1, ax=axes[1])
    cb.set_label("m/s")

    fig.suptitle(f"Error Maps - Belgium - Sample {sample_index}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / f"error_maps_sample_{sample_index}_belgium_only.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved error maps to: {output_path}")

# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    processed_dir, model_path, output_dir = get_paths()

    print(f"\nUsing data from: {processed_dir}")
    print(f"Loading model from: {model_path}")
    print(f"Saving figures to: {output_dir}")

    dataset, latitudes, longitudes = build_dataset(processed_dir)
    model = load_model(model_path, dataset, device)

    y_true, y_pred = predict_one_sample(
        model=model,
        dataset=dataset,
        sample_index=SAMPLE_INDEX,
        device=device
    )

    mse, mae = compute_metrics(y_true, y_pred)

    print(f"\nEvaluating sample index: {SAMPLE_INDEX}")
    print(f"True target shape : {y_true.shape}")
    print(f"Pred target shape : {y_pred.shape}")
    print(f"MSE               : {mse:.6f}")
    print(f"MAE               : {mae:.6f}")

    plot_prediction_maps(
        y_true=y_true,
        y_pred=y_pred,
        latitudes=latitudes,
        longitudes=longitudes,
        output_dir=output_dir,
        sample_index=SAMPLE_INDEX
    )

    plot_error_maps(
        y_true=y_true,
        y_pred=y_pred,
        latitudes=latitudes,
        longitudes=longitudes,
        output_dir=output_dir,
        sample_index=SAMPLE_INDEX
    )


if __name__ == "__main__":
    main()