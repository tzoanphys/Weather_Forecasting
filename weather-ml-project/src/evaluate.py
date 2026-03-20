from pathlib import Path

import geopandas as gpd
import geodatasets

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

# ============================================================
# Plotting
# ============================================================

def plot_prediction_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    output_dir: Path,
    sample_index: int
) -> None:
    """
    Plot true and predicted u10 / v10 maps over the Belgium domain only.
    """
    extent = [
        longitudes.min(), longitudes.max(),
        latitudes.min(), latitudes.max()
    ]

    true_u10 = y_true[0]
    true_v10 = y_true[1]
    pred_u10 = y_pred[0]
    pred_v10 = y_pred[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im0 = axes[0, 0].imshow(true_u10, origin="lower", extent=extent, aspect="auto")
    axes[0, 0].set_xlim(longitudes.min(), longitudes.max())
    axes[0, 0].set_ylim(latitudes.min(), latitudes.max())
    axes[0, 0].set_title("True u10")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pred_u10, origin="lower", extent=extent, aspect="auto")
    axes[0, 1].set_xlim(longitudes.min(), longitudes.max())
    axes[0, 1].set_ylim(latitudes.min(), latitudes.max())
    axes[0, 1].set_title("Predicted u10")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(true_v10, origin="lower", extent=extent, aspect="auto")
    axes[1, 0].set_xlim(longitudes.min(), longitudes.max())
    axes[1, 0].set_ylim(latitudes.min(), latitudes.max())
    axes[1, 0].set_title("True v10")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(pred_v10, origin="lower", extent=extent, aspect="auto")
    axes[1, 1].set_xlim(longitudes.min(), longitudes.max())
    axes[1, 1].set_ylim(latitudes.min(), latitudes.max())
    axes[1, 1].set_title("Predicted v10")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(f"Forecast Evaluation - Belgium - Sample {sample_index}")
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(error_u10, origin="lower", extent=extent, aspect="auto")
    axes[0].set_xlim(longitudes.min(), longitudes.max())
    axes[0].set_ylim(latitudes.min(), latitudes.max())
    axes[0].set_title("Absolute Error - u10")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(error_v10, origin="lower", extent=extent, aspect="auto")
    axes[1].set_xlim(longitudes.min(), longitudes.max())
    axes[1].set_ylim(latitudes.min(), latitudes.max())
    axes[1].set_title("Absolute Error - v10")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(f"Error Maps - Belgium - Sample {sample_index}")
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