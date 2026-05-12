from __future__ import annotations

"""
Neural network model for learning forecast error.

The network predicts the error:
    error = observation - forecast

Then we compute:
    corrected forecast = forecast + predicted error
"""

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import PostprocessingConfig


class ErrorNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class TrainedNeuralPostprocessor:
    model: ErrorNet
    scaler: StandardScaler
    feature_names: list[str]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "forecast_wind_speed_ms": df["forecast_wind_speed_ms"],
            "u10": df["u10"],
            "v10": df["v10"],
            "station_lat": df["station_lat"],
            "station_lon": df["station_lon"],
            "hour_sin": np.sin(2 * np.pi * df["hour"] / 24.0),
            "hour_cos": np.cos(2 * np.pi * df["hour"] / 24.0),
            "month_sin": np.sin(2 * np.pi * df["month"] / 12.0),
            "month_cos": np.cos(2 * np.pi * df["month"] / 12.0),
        }
    )
    return features


def train_neural_postprocessor(df: pd.DataFrame, config: PostprocessingConfig) -> TrainedNeuralPostprocessor:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    features = build_features(df)
    targets = df["error"].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test, df_train, df_test = train_test_split(
        features,
        targets,
        df,
        test_size=config.test_ratio,
        random_state=config.random_seed,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = ErrorNet(input_size=x_train_tensor.shape[1], hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_tensor)
        loss = loss_function(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_predictions = model(x_test_tensor)
                val_loss = loss_function(val_predictions, y_test_tensor)
            print(f"Epoch {epoch:03d} | train loss = {loss.item():.4f} | val loss = {val_loss.item():.4f}")

    feature_names = list(features.columns)

    save_bundle(model, scaler, feature_names, config)
    return TrainedNeuralPostprocessor(model=model, scaler=scaler, feature_names=feature_names)


def apply_neural_postprocessor(bundle: TrainedNeuralPostprocessor, df: pd.DataFrame) -> pd.DataFrame:
    features = build_features(df)
    x_scaled = bundle.scaler.transform(features)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    bundle.model.eval()
    with torch.no_grad():
        predicted_error = bundle.model(x_tensor).numpy().reshape(-1)

    result = df.copy()
    result["predicted_error"] = predicted_error
    result["corrected_wind_speed_ms"] = result["forecast_wind_speed_ms"] + result["predicted_error"]
    return result


def save_bundle(model: ErrorNet, scaler: StandardScaler, feature_names: list[str], config: PostprocessingConfig) -> None:
    model_path = config.model_dir / "neural_error_model.pt"
    scaler_path = config.model_dir / "neural_scaler.pkl"
    feature_path = config.model_dir / "feature_names.pkl"

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)
    with open(feature_path, "wb") as file:
        pickle.dump(feature_names, file)

    print(f"Saved neural model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved feature names to: {feature_path}")
