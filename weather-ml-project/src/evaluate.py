from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import torch

from simple_model import SimpleWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# -----------------------------
# Settings
# -----------------------------
INPUT_STEPS = 2
TARGET_OFFSET = 1
TRAIN_RATIO = 0.8

MODEL_FILENAME = "simple_wind_cnn.pth"


def main() -> None:
    # -----------------------------
    # Device: CUDA → MPS → CPU
    # -----------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # -----------------------------
    # Paths
    # -----------------------------
    project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "saved_models" / MODEL_FILENAME
    output_dir = project_root / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    data, *_ = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET
    )

    print("Total samples:", len(dataset))

    # -----------------------------
    # Load model
    # -----------------------------
    sample_x, sample_y = dataset[0]

    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    model = SimpleWindCNN(
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Loaded model:")
    print(model_path)

    # -----------------------------
    # Validation indices
    # -----------------------------
    train_size = int(TRAIN_RATIO * len(dataset))

    if train_size == len(dataset):
        train_size = len(dataset) - 1

    val_indices = list(range(train_size, len(dataset)))

    print("Validation samples:", len(val_indices))

    # -----------------------------
    # Helper functions
    # -----------------------------
    def predict(index):
        x, y_true = dataset[index]

        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x)

        y_true = y_true.numpy()
        y_pred = y_pred.squeeze(0).cpu().numpy()

        return y_true, y_pred

    def compute_mse(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def compute_mae(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def wind_speed(y):
        u10 = y[0]
        v10 = y[1]
        return np.sqrt(u10 ** 2 + v10 ** 2)

    # -----------------------------
    # Evaluate raw predictions
    # -----------------------------
    all_true = []
    all_pred = []

    raw_mse_list = []
    raw_mae_list = []

    for index in val_indices:
        y_true, y_pred = predict(index)

        all_true.append(y_true)
        all_pred.append(y_pred)

        raw_mse_list.append(compute_mse(y_true, y_pred))
        raw_mae_list.append(compute_mae(y_true, y_pred))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    raw_mse_mean = float(np.mean(raw_mse_list))
    raw_mae_mean = float(np.mean(raw_mae_list))

    print("Raw MSE:", raw_mse_mean)
    print("Raw MAE:", raw_mae_mean)

    # -----------------------------
    # Simple postprocessing
    # corrected = a * prediction + b
    # -----------------------------
    pred_flat = all_pred.reshape(-1)
    true_flat = all_true.reshape(-1)

    a = np.cov(pred_flat, true_flat)[0, 1] / np.var(pred_flat)
    b = true_flat.mean() - a * pred_flat.mean()

    all_pred_corrected = a * all_pred + b

    corrected_mse_list = []
    corrected_mae_list = []

    for i in range(len(val_indices)):
        corrected_mse_list.append(compute_mse(all_true[i], all_pred_corrected[i]))
        corrected_mae_list.append(compute_mae(all_true[i], all_pred_corrected[i]))

    corrected_mse_mean = float(np.mean(corrected_mse_list))
    corrected_mae_mean = float(np.mean(corrected_mae_list))

    print("Corrected MSE:", corrected_mse_mean)
    print("Corrected MAE:", corrected_mae_mean)

    print("Postprocessing formula:")
    print("corrected = a * prediction + b")
    print("a =", a)
    print("b =", b)

    # -----------------------------
    # Choose one sample for final plot
    # -----------------------------
    best_position = int(np.argmin(corrected_mse_list))
    best_index = val_indices[best_position]

    y_true = all_true[best_position]
    y_pred = all_pred[best_position]
    y_corr = all_pred_corrected[best_position]

    true_speed = wind_speed(y_true)
    pred_speed = wind_speed(y_pred)
    corr_speed = wind_speed(y_corr)
    error_speed = np.abs(corr_speed - true_speed)

    # -----------------------------
    # Final plot
    # -----------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(true_speed)
    axes[0].set_title("True wind speed")

    axes[1].imshow(pred_speed)
    axes[1].set_title("Raw prediction")

    axes[2].imshow(corr_speed)
    axes[2].set_title("Corrected prediction")

    axes[3].imshow(error_speed)
    axes[3].set_title("Corrected error")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    plot_path = output_dir / "final_evaluation_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved final plot:")
    print(plot_path)

    # -----------------------------
    # Save summary
    # -----------------------------
    summary = {
        "validation_samples": len(val_indices),
        "raw_mse": raw_mse_mean,
        "raw_mae": raw_mae_mean,
        "corrected_mse": corrected_mse_mean,
        "corrected_mae": corrected_mae_mean,
        "postprocessing": {
            "formula": "corrected = a * prediction + b",
            "a": float(a),
            "b": float(b),
        },
        "plotted_sample_index": int(best_index),
        "final_plot": str(plot_path),
    }

    summary_path = output_dir / "evaluation_summary.json"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved evaluation summary:")
    print(summary_path)


if __name__ == "__main__":
    main()
