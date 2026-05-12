from __future__ import annotations

"""Simple baseline methods: mean bias correction and linear regression."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class MeanBiasModel:
    mean_error: float

    def predict(self, forecast_speed: np.ndarray) -> np.ndarray:
        return forecast_speed + self.mean_error


def fit_mean_bias(train_df: pd.DataFrame) -> MeanBiasModel:
    return MeanBiasModel(mean_error=float(train_df["error"].mean()))


class SimpleMOSModel:
    """Small linear MOS-style model."""

    def __init__(self) -> None:
        self.model = LinearRegression()
        self.feature_names = ["forecast_wind_speed_ms", "u10", "v10"]

    def fit(self, train_df: pd.DataFrame) -> None:
        x = train_df[self.feature_names]
        y = train_df["obs_wind_speed_ms"]
        self.model.fit(x, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_names]
        return self.model.predict(x)
