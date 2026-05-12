from __future__ import annotations

"""Evaluate raw forecasts, mean-bias correction, MOS, and neural postprocessing."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .baseline_bias import SimpleMOSModel, fit_mean_bias
from .config import PostprocessingConfig
from .neural_postprocessing import apply_neural_postprocessor, train_neural_postprocessor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_all_methods(df: pd.DataFrame, config: PostprocessingConfig) -> pd.DataFrame:
    train_df, test_df = train_test_split(df, test_size=config.test_ratio, random_state=config.random_seed)

    y_true = test_df["obs_wind_speed_ms"].to_numpy()

    # 1. Raw forecast
    raw_pred = test_df["forecast_wind_speed_ms"].to_numpy()

    # 2. Mean bias correction
    mean_bias_model = fit_mean_bias(train_df)
    mean_bias_pred = mean_bias_model.predict(test_df["forecast_wind_speed_ms"].to_numpy())

    # 3. Linear MOS
    mos_model = SimpleMOSModel()
    mos_model.fit(train_df)
    mos_pred = mos_model.predict(test_df)

    # 4. Neural network
    neural_bundle = train_neural_postprocessor(train_df, config)
    neural_df = apply_neural_postprocessor(neural_bundle, test_df)
    neural_pred = neural_df["corrected_wind_speed_ms"].to_numpy()

    results = pd.DataFrame(
        [
            {
                "method": "Raw forecast",
                "MAE": mean_absolute_error(y_true, raw_pred),
                "RMSE": rmse(y_true, raw_pred),
            },
            {
                "method": "Mean bias correction",
                "MAE": mean_absolute_error(y_true, mean_bias_pred),
                "RMSE": rmse(y_true, mean_bias_pred),
            },
            {
                "method": "Linear MOS",
                "MAE": mean_absolute_error(y_true, mos_pred),
                "RMSE": rmse(y_true, mos_pred),
            },
            {
                "method": "Neural postprocessing",
                "MAE": mean_absolute_error(y_true, neural_pred),
                "RMSE": rmse(y_true, neural_pred),
            },
        ]
    )

    output_path = config.outputs_dir / "postprocessing_scores.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved evaluation results to: {output_path}")
    return results
