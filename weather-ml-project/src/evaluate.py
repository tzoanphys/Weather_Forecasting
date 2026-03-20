from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.simple_model import SimpleWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# ============================================================
# Configuration
# ============================================================

INPUT_STEPS = 2
TARGET_OFFSET = 1
MODEL_FILENAME = "wind_forecast_cnn.pth"
SAMPLE_INDEX = 0


# ============================================================
# Device selection
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

def get_project_paths() -> tuple[Path, Path, Path, Path]:
    """
    Return project paths:
    - project root
    - processed data directory
    - trained model path
    - output directory for evaluation figures
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "saved_models" / MODEL_FILENAME
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    return project_root, processed_dir, model_path, output_dir


# ============================================================
# Data loading
# ============================================================

def build_dataset(processed_dir: Path) -> WindForecastDataset:
    """
    Load processed wind files and create the evaluation dataset.
    """
    data, _ = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET
    )

    if len(dataset) == 0:
        raise ValueError(
            "The dataset contains zero samples. "
            "Check your processed files and INPUT_STEPS / TARGET_OFFSET."
        )

    print(f"\nEvaluation dataset contains {len(dataset)} sample(s).")
    return dataset


# ============================================================
# Model loading
# ============================================================

def load_trained_model(
    model_path: Path,
    dataset: WindForecastDataset,
    device: torch.device
) -> nn.Module:
    """
    Rebuild the model with the correct input/output shape
    and load the trained weights.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    sample_x, sample_y = dataset[0]
    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    model = SimpleWindCNN(
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

def predict_sample(
    model: nn.Module,
    dataset: WindForecastDataset,
    sample_index: int,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model on one sample and return:
    - input tensor
    - true target
    - predicted target
    """
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range. "
            f"Valid range: 0 to {len(dataset) - 1}"
        )

    x, y_true = dataset[sample_index]
    x_batch = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x_batch)

    x_np = x.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.squeeze(0).cpu().numpy()

    return x_np, y_true_np, y_pred_np


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute simple regression metrics.
    """
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    return mse, mae


# ============================================================
# Plotting
# ============================================================

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_index: int,
    output_dir: Path
) -> None:
    """
    Plot true and predicted u10 / v10 fields and save the figure.
    """
    true_u10 = y_true[0]
    true_v10 = y_true[1]
    pred_u10 = y_pred[0]
    pred_v10 = y_pred[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im0 = axes[0, 0].imshow(true_u10, origin="lower")
    axes[0, 0].set_title("True u10")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(pred_u10, origin="lower")
    axes[0, 1].set_title("Predicted u10")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(true_v10, origin="lower")
    axes[1, 0].set_title("True v10")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(pred_v10, origin="lower")
    axes[1, 1].set_title("Predicted v10")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Forecast Evaluation - Sample {sample_index}", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"evaluation_sample_{sample_index}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved prediction figure to: {output_path}")


def plot_error_maps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_index: int,
    output_dir: Path
) -> None:
    """
    Plot absolute error maps for u10 and v10 and save the figure.
    """
    error_u10 = np.abs(y_pred[0] - y_true[0])
    error_v10 = np.abs(y_pred[1] - y_true[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(error_u10, origin="lower")
    axes[0].set_title("Absolute Error - u10")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(error_v10, origin="lower")
    axes[1].set_title("Absolute Error - v10")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Prediction Error Maps - Sample {sample_index}", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"error_maps_sample_{sample_index}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved error figure to: {output_path}")


# ============================================================
# Main evaluation pipeline
# ============================================================

def main() -> None:
    device = get_device()
    _, processed_dir, model_path, output_dir = get_project_paths()

    print(f"\nUsing data from: {processed_dir}")
    print(f"Loading model from: {model_path}")
    print(f"Saving evaluation figures to: {output_dir}")

    dataset = build_dataset(processed_dir)
    model = load_trained_model(model_path, dataset, device)

    x, y_true, y_pred = predict_sample(
        model=model,
        dataset=dataset,
        sample_index=SAMPLE_INDEX,
        device=device
    )

    mse, mae = compute_metrics(y_true, y_pred)

    print(f"\nEvaluating sample index: {SAMPLE_INDEX}")
    print(f"Input shape       : {x.shape}")
    print(f"True target shape : {y_true.shape}")
    print(f"Pred target shape : {y_pred.shape}")
    print(f"MSE               : {mse:.6f}")
    print(f"MAE               : {mae:.6f}")

    plot_results(y_true, y_pred, SAMPLE_INDEX, output_dir)
    plot_error_maps(y_true, y_pred, SAMPLE_INDEX, output_dir)


if __name__ == "__main__":
    main()