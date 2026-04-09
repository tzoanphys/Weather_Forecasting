"""
=============================================================================
STEP 2 — Download real wind observations via Meteostat
=============================================================================

WHY WE NEED REAL OBSERVATIONS
------------------------------
Statistical postprocessing (as described in Vannitsem et al. 2021, BAMS)
works by learning the *error* between the NWP forecast and the observed
reality.  Without real observations we cannot know whether GFS is biased
high, low, or systematically wrong in a certain wind direction.

WHY METEOSTAT
--------------
Meteostat (https://meteostat.net) provides hourly and daily weather data
from thousands of WMO/SYNOP stations worldwide, gathered through open
sources such as NOAA ISD, DWD, and Copernicus.  Their Python library
(`meteostat`) gives direct pandas DataFrames — no API key required.

The library has a simple interface:
    from meteostat import Point, Hourly
    data = Hourly(Point(lat, lon), start, end).fetch()

It automatically picks the nearest station(s) and interpolates if needed.

WHAT THIS SCRIPT PRODUCES
--------------------------
For each station listed in STATIONS it creates a CSV file:
    data/obs/observations_<ICAO_or_name>.csv

Each CSV has columns:
    time, wspd (m/s), wdir (degrees), temp (°C), u10 (m/s), v10 (m/s)

The u10 / v10 columns are *derived* from wspd + wdir so they are directly
comparable with the GFS u10/v10 variables (standard meteorological convention).

Wind decomposition formula:
    Given wind speed (spd) and **meteorological** direction (dir — the
    direction the wind is COMING FROM, measured clockwise from North):

        u = -spd * sin(dir_rad)   ← eastward component
        v = -spd * cos(dir_rad)   ← northward component

    The minus signs are because dir is the source direction, not destination.

PARAMETERS YOU CAN CHANGE
--------------------------
STATIONS   — list of (name, lat, lon, elev_m) for your area of interest
START_DATE — beginning of the time window to download
END_DATE   — end of the time window (defaults to today)
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Meteostat must be installed:  pip install meteostat
try:
    from meteostat import Point, Hourly
except ImportError:
    raise ImportError(
        "Please install meteostat:  pip install meteostat\n"
        "Meteostat fetches free hourly weather observations from WMO stations."
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Station(NamedTuple):
    name: str
    lat: float
    lon: float
    elevation_m: float


# Belgian + nearby Dutch/German synoptic stations.
# The lat/lon is used to find the nearest Meteostat station.
STATIONS: list[Station] = [
    Station("Brussels",   50.90,  4.48,  58.0),
    Station("Liege",      50.63,  5.44,  200.0),
    Station("Ghent",      51.19,  3.82,   8.0),
    Station("Antwerp",    51.33,  4.47,  10.0),
    Station("Uccle",      50.80,  4.35,  100.0),
    Station("De_Bilt_NL", 52.10,  5.18,   2.0),   # famous Dutch reference station
]

# Time window — we go back enough to cover the 20 most recent GFS runs
# PLUS a decent training history for bias estimation.
END_DATE: datetime.datetime = datetime.datetime.utcnow()
START_DATE: datetime.datetime = END_DATE - datetime.timedelta(days=60)  # 2 months

# Output directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "obs"

# Minimum number of valid observation rows required to save a station file.
MIN_OBS_ROWS: int = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def wind_to_uv(wspd: pd.Series, wdir: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert wind speed (m/s) and meteorological wind direction (degrees,
    0° = from North, 90° = from East) into u10 and v10 components.

    Meteorological convention says direction is where the wind COMES FROM,
    so we negate:
        u = -spd * sin(dir_rad)
        v = -spd * cos(dir_rad)

    Both input Series may contain NaN — those rows will produce NaN output.
    """
    dir_rad = np.deg2rad(wdir.values)
    u = -wspd.values * np.sin(dir_rad)
    v = -wspd.values * np.cos(dir_rad)
    return pd.Series(u, index=wspd.index, name="u10"), \
           pd.Series(v, index=wspd.index, name="v10")


def fetch_station(station: Station, start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame | None:
    """
    Download hourly observations for a single station using Meteostat.

    Returns a cleaned DataFrame or None if no data was available.

    DataFrame columns:
        time (index), wspd_ms, wdir_deg, temp_c, u10, v10
    """
    print(f"\n  Fetching observations for: {station.name}  "
          f"({station.lat:.2f}°N, {station.lon:.2f}°E)")

    # Create a Meteostat Point — it automatically finds the nearest stations.
    location = Point(station.lat, station.lon, station.elevation_m)

    try:
        # Hourly() fetches observations from all stations near the Point
        # and merges them.  The resulting index is UTC datetime.
        data = Hourly(location, start, end).fetch()
    except Exception as exc:
        print(f"  [ERROR] Meteostat fetch failed: {exc}")
        return None

    if data.empty:
        print(f"  [WARN] No data returned for {station.name}.")
        return None

    print(f"  Raw rows fetched: {len(data)}")

    # The Meteostat hourly columns we care about:
    #   wspd = wind speed (km/h) — will convert to m/s
    #   wdir = wind direction (degrees)
    #   temp = temperature (°C)
    needed = ["wspd", "wdir"]
    available = [c for c in needed if c in data.columns]
    if not available:
        print(f"  [WARN] No wind columns found in data for {station.name}.")
        return None

    # Meteostat returns wind speed in km/h — convert to m/s (÷ 3.6).
    # Wind direction is already in meteorological degrees.
    df = pd.DataFrame(index=data.index)
    df.index.name = "time"

    if "wspd" in data.columns:
        df["wspd_ms"] = data["wspd"] / 3.6   # km/h → m/s
    else:
        df["wspd_ms"] = np.nan

    if "wdir" in data.columns:
        df["wdir_deg"] = data["wdir"]
    else:
        df["wdir_deg"] = np.nan

    if "temp" in data.columns:
        df["temp_c"] = data["temp"]
    else:
        df["temp_c"] = np.nan

    # Drop rows where wind data is entirely missing.
    df = df.dropna(subset=["wspd_ms", "wdir_deg"], how="all")

    if len(df) < MIN_OBS_ROWS:
        print(f"  [WARN] Too few valid rows ({len(df)}) for {station.name}.")
        return None

    # Derive u10 / v10 from speed + direction.
    df["u10"], df["v10"] = wind_to_uv(df["wspd_ms"], df["wdir_deg"])

    print(f"  Valid rows after cleaning: {len(df)}")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    print(f"  Mean wind speed: {df['wspd_ms'].mean():.2f} m/s")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Observation output directory: {OUTPUT_DIR}")
    print(f"Time window: {START_DATE.date()} → {END_DATE.date()}")
    print(f"Stations to process: {len(STATIONS)}")

    summary: list[dict] = []

    for station in STATIONS:
        df = fetch_station(station, START_DATE, END_DATE)

        if df is None:
            summary.append({"station": station.name, "rows": 0, "status": "FAILED"})
            continue

        out_path = OUTPUT_DIR / f"observations_{station.name}.csv"
        df.to_csv(out_path)
        summary.append({"station": station.name, "rows": len(df), "status": "OK"})
        print(f"  Saved → {out_path.name}")

    # Print a summary table.
    print("\n" + "=" * 50)
    print("Download summary")
    print("=" * 50)
    for s in summary:
        status = "✓" if s["status"] == "OK" else "✗"
        print(f"  {status}  {s['station']:<20}  {s['rows']:>5} rows")

    print(f"\nAll observation files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
