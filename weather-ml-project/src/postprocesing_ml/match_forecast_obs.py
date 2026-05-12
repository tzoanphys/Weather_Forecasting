from __future__ import annotations

"""
Create forecast-observation pairs.

Expected forecast input
-----------------------
A CSV file in data/raw/forecast.csv with at least these columns:
- valid_time
- latitude
- longitude
- u10
- v10

If you later want, you can replace this CSV reading step with direct GFS reading.
For teaching purposes, CSV is easier to study first.
"""

import numpy as np
import pandas as pd

from .config import PostprocessingConfig


def load_forecast_csv(config: PostprocessingConfig) -> pd.DataFrame:
    forecast_path = config.raw_dir / "forecast.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Could not find {forecast_path}.\n"
            "Please save a forecast CSV there before running the matching step."
        )

    forecast = pd.read_csv(forecast_path)
    forecast["valid_time"] = pd.to_datetime(forecast["valid_time"])

    required = {"valid_time", "latitude", "longitude", "u10", "v10"}
    missing = required.difference(forecast.columns)
    if missing:
        raise ValueError(f"Forecast CSV is missing columns: {sorted(missing)}")

    forecast["forecast_wind_speed_ms"] = np.sqrt(forecast["u10"] ** 2 + forecast["v10"] ** 2)
    return forecast


def load_observations_csv(config: PostprocessingConfig) -> pd.DataFrame:
    obs_path = config.raw_dir / "observations.csv"
    if not obs_path.exists():
        raise FileNotFoundError(
            f"Could not find {obs_path}.\n"
            "Run fetch_observations.py first."
        )

    obs = pd.read_csv(obs_path)
    obs["valid_time"] = pd.to_datetime(obs["valid_time"])
    return obs


def nearest_gridpoint_match(forecast: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """
    Match each station observation to the nearest forecast grid point at the same time.

    This is a simple and pedagogical first method.
    Later, you could replace it with interpolation.
    """
    matched_rows = []

    for valid_time, obs_group in observations.groupby("valid_time"):
        fcst_group = forecast[forecast["valid_time"] == valid_time]
        if fcst_group.empty:
            continue

        fcst_lats = fcst_group["latitude"].to_numpy()
        fcst_lons = fcst_group["longitude"].to_numpy()

        for _, obs_row in obs_group.iterrows():
            distance_sq = (fcst_lats - obs_row["station_lat"]) ** 2 + (fcst_lons - obs_row["station_lon"]) ** 2
            nearest_index = int(np.argmin(distance_sq))
            nearest_fcst = fcst_group.iloc[nearest_index]

            matched_rows.append(
                {
                    "valid_time": valid_time,
                    "station_id": obs_row["station_id"],
                    "station_lat": obs_row["station_lat"],
                    "station_lon": obs_row["station_lon"],
                    "grid_lat": nearest_fcst["latitude"],
                    "grid_lon": nearest_fcst["longitude"],
                    "u10": nearest_fcst["u10"],
                    "v10": nearest_fcst["v10"],
                    "forecast_wind_speed_ms": nearest_fcst["forecast_wind_speed_ms"],
                    "obs_wind_speed_ms": obs_row["obs_wind_speed_ms"],
                }
            )

    if not matched_rows:
        raise RuntimeError("No forecast-observation pairs were created.")

    matched = pd.DataFrame(matched_rows)
    matched["error"] = matched["obs_wind_speed_ms"] - matched["forecast_wind_speed_ms"]
    matched["hour"] = pd.to_datetime(matched["valid_time"]).dt.hour
    matched["month"] = pd.to_datetime(matched["valid_time"]).dt.month
    return matched


def build_training_table(config: PostprocessingConfig) -> pd.DataFrame:
    forecast = load_forecast_csv(config)
    observations = load_observations_csv(config)
    matched = nearest_gridpoint_match(forecast, observations)

    output_path = config.processed_dir / "matched_forecast_observations.csv"
    matched.to_csv(output_path, index=False)
    print(f"Saved matched table to: {output_path}")
    return matched


if __name__ == "__main__":
    cfg = PostprocessingConfig()
    cfg.ensure_directories()
    build_training_table(cfg)
