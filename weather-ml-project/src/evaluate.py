# evaluate.py
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

import config
from simple_model import SimpleWindCNN
from dataset import load_wind_time_series, WindForecastDataset

def main() -> None:
    print("Using device:", config.DEVICE)

    # --- Load Dataset ---
    data, _files, latitudes, longitudes = load_wind_time_series(config.PROCESSED_DIR)

    dataset = WindForecastDataset(
        data=data,
        input_steps=config.INPUT_STEPS,
        target_offset=config.TARGET_OFFSET
    )
    print("Total samples:", len(dataset))

    # --- Reconstruct Model Topology ---
    sample_x, sample_y = dataset[0]
    model = SimpleWindCNN(in_channels=sample_x.shape[0], out_channels=sample_y.shape[0]).to(config.DEVICE)
    
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    print("Loaded execution model state from:", config.MODEL_PATH)

    # --- Retrieve Identical Validation Boundaries ---
    _, val_indices = config.get_train_val_indices(len(dataset))
    print("Validation instances evaluating:", len(val_indices))

    # --- Prediction & Math Calculators ---
    def predict(index):
        x, y_true = dataset[index]
        x = x.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            y_pred = model(x)
        return y_true.numpy(), y_pred.squeeze(0).cpu().numpy()

    compute_mse = lambda true, pred: np.mean((pred - true) ** 2)
    compute_mae = lambda true, pred: np.mean(np.abs(pred - true))
    wind_speed  = lambda y: np.sqrt(y[0] ** 2 + y[1] ** 2)

    # --- Inference Evaluation Run ---
    all_true, all_pred, raw_mse_list, raw_mae_list = [], [], [], []

    for index in val_indices:
        y_true, y_pred = predict(index)
        all_true.append(y_true)
        all_pred.append(y_pred)
        raw_mse_list.append(compute_mse(y_true, y_pred))
        raw_mae_list.append(compute_mae(y_true, y_pred))

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    raw_mse_mean = float(np.mean(raw_mse_list))
    raw_mae_mean = float(np.mean(raw_mae_list))

    print(f"Raw Eval Measurements -> Mean MSE: {raw_mse_mean:.6f} | Mean MAE: {raw_mae_mean:.6f}")

    # --- Linear Recalibration Variance Adjustment ---
    pred_flat = all_pred.reshape(-1)
    true_flat = all_true.reshape(-1)

    a = np.cov(pred_flat, true_flat)[0, 1] / np.var(pred_flat)
    b = true_flat.mean() - a * pred_flat.mean()
    all_pred_corrected = a * all_pred + b

    corrected_mse_list = [compute_mse(all_true[i], all_pred_corrected[i]) for i in range(len(val_indices))]
    corrected_mae_list = [compute_mae(all_true[i], all_pred_corrected[i]) for i in range(len(val_indices))]
    
    corrected_mse_mean = float(np.mean(corrected_mse_list))
    corrected_mae_mean = float(np.mean(corrected_mae_list))

    # --- Matrix Graph Visual Execution Pipeline ---
    best_position = int(np.argmin(corrected_mse_list))
    best_index = val_indices[best_position]

    lon, lat = np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float)
    extent = (float(lon.min()), float(lon.max()), float(lat[-1] if lat[0] > lat[-1] else lat[0]), float(lat[0] if lat[0] > lat[-1] else lat[-1]))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    aspect = 1.0 / np.cos(np.radians(0.5 * (extent[2] + extent[3])))

    panels = [
        (axes[0], wind_speed(all_true[best_position]), "True wind speed (m/s)", "viridis", "m/s"),
        (axes[1], wind_speed(all_pred[best_position]), "Raw prediction (m/s)", "viridis", "m/s"),
        (axes[2], wind_speed(all_pred_corrected[best_position]), "Corrected prediction (m/s)", "viridis", "m/s"),
        (axes[3], np.abs(wind_speed(all_pred_corrected[best_position]) - wind_speed(all_true[best_position])), "|Corrected − true| (m/s)", "magma", "m/s"),
    ]

    for ax, field, title, cmap, cbar_label in panels:
        im = ax.imshow(field, origin="upper", extent=extent, cmap=cmap, aspect="auto")
        ax.set_aspect(aspect)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude °E", fontsize=9)
        ax.set_ylabel("Latitude °N", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    plt.tight_layout()
    plot_path = config.OUTPUT_DIR / "final_evaluation_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # --- Export Results Structural Output ---
    summary = {
        "validation_samples": len(val_indices),
        "raw_mse": raw_mse_mean,
        "raw_mae": raw_mae_mean,
        "corrected_mse": corrected_mse_mean,
        "corrected_mae": corrected_mae_mean,
        "postprocessing": {"formula": "corrected = a * prediction + b", "a": float(a), "b": float(b)},
        "plotted_sample_index": int(best_index),
        "final_plot": str(plot_path),
    }
    
    summary_path = config.OUTPUT_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Generated plots and summary configuration inside /outputs folder completely.")

if __name__ == "__main__":
    main()