from __future__ import annotations

"""
Download real wind observations with Meteostat.

This file is intentionally simple and heavily commented.
The goal is to show clearly:
1. how we find stations,
2. how we download hourly observations,
3. how we save a clean table for the later ML steps.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from meteostat import Hourly, Stations

from .config import PostprocessingConfig


def find_stations(config: PostprocessingConfig) -> pd.DataFrame:
    """Find weather stations inside the selected Europe box."""
    stations = Stations()
    stations = stations.bounds((config.west, config.south), (config.east, config.north))
    stations = stations.fetch(config.max_stations)

    if stations.empty:
        raise RuntimeError("No stations were found in the requested area.")

    stations = stations.reset_index().rename(columns={"id": "station_id"})
    return stations


def download_hourly_observations(config: PostprocessingConfig) -> pd.DataFrame:
    """
    Download recent hourly observations for several stations.

    Meteostat gives wind speed in km/h.
    For weather model work, we usually prefer m/s, so we convert it.
    """
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=config.obs_days)

    stations = find_stations(config)
    collected = []

    for _, station in stations.iterrows():
        station_id = station["station_id"]
        try:
            hourly = Hourly(station_id, start_time, end_time, timezone="UTC").fetch()
            if hourly.empty:
                continue

            hourly = hourly.reset_index().rename(columns={"time": "valid_time"})
            hourly["station_id"] = station_id
            hourly["station_lat"] = station["latitude"]
            hourly["station_lon"] = station["longitude"]

            # Keep only columns that are useful for our first prototype.
            keep_columns = [col for col in ["valid_time", "wspd", "wdir", "temp"] if col in hourly.columns]
            hourly = hourly[keep_columns + ["station_id", "station_lat", "station_lon"]].copy()

            if "wspd" not in hourly.columns:
                continue

            hourly = hourly.dropna(subset=["wspd"])
            hourly["obs_wind_speed_ms"] = hourly["wspd"] / 3.6
            collected.append(hourly)
        except Exception as exc:
            print(f"Skipping station {station_id}: {exc}")

    if not collected:
        raise RuntimeError("No observation data could be downloaded.")

    observations = pd.concat(collected, ignore_index=True)
    observations["valid_time"] = pd.to_datetime(observations["valid_time"]).dt.tz_localize(None)

    output_path = config.raw_dir / "observations.csv"
    observations.to_csv(output_path, index=False)
    print(f"Saved observations to: {output_path}")
    return observations


if __name__ == "__main__":
    cfg = PostprocessingConfig()
    cfg.ensure_directories()
    download_hourly_observations(cfg)
