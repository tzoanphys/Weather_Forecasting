"""
=============================================================================
STEP 1 — Download the 20 most recent GFS forecasts from NOAA AWS
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
The Global Forecast System (GFS) is NOAA's main deterministic NWP model.
It runs 4 times per day (00z, 06z, 12z, 18z). Its outputs are public on the
AWS Open Data Registry:
    https://registry.opendata.aws/noaa-gfs-bdp-pds/

We want *wind* only — specifically u10 and v10 (horizontal wind components
at 10 m above ground). A full GFS GRIB2 file is ~400 MB. By reading the
companion `.idx` index file we can request only the byte range that contains
the wind variables, reducing the download to ~5–15 MB per forecast.

HOW THE INDEX TRICK WORKS
--------------------------
Each `.idx` file looks like:
    1:0:d=2026010100:UGRD:10 m above ground:anl:
    2:123456:d=2026010100:VGRD:10 m above ground:anl:
    3:234567:...

The second colon-separated field is the byte offset where that record starts.
The next record's offset is therefore the end of the current record.  We pass
`Range: bytes=START-END` in the HTTP request to download just that slice.

WHAT THIS SCRIPT PRODUCES
--------------------------
For each of the 20 most recent GFS analysis/forecast runs it downloads:
    data/raw_postproc/gfs.YYYYMMDD.tHHz.pgrb2.0p25.f000        ← 0-h analysis
    data/raw_postproc/gfs.YYYYMMDD.tHHz.pgrb2.0p25.f006        ← 6-h forecast
The files are partial GRIB2 (wind-only) but cfgrib reads them perfectly.

PARAMETERS YOU CAN CHANGE
--------------------------
N_RUNS    — how many recent GFS runs to download (default 20)
CYCLE     — which daily cycle to grab when falling back ("00", "06" …)
FORECAST_HOURS — which lead times to download per run
BBOX      — lat/lon bounding box for later cropping (Belgium + surroundings)
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import NamedTuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How many of the most recent GFS analysis runs to download.
N_RUNS: int = 20

# Which forecast hours to download per run.
# 000 = analysis (0-h lead time), 006 = 6-h forecast, etc.
FORECAST_HOURS: list[str] = ["000", "006", "012", "018", "024"]

# GFS model cycle (UTC hour).  "00" is the midnight run.
CYCLE: str = "00"

# The GFS 0.25-degree global grid product tag on AWS.
PRODUCT: str = "pgrb2.0p25"

# AWS base URL for GFS open data.
GFS_BASE_URL: str = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
)

# Variables to keep.  Each entry is (shortName, level-description in .idx).
TARGET_VARS: list[tuple[str, str]] = [
    ("UGRD", "10 m above ground"),   # eastward wind
    ("VGRD", "10 m above ground"),   # northward wind
]

# Output directory (separate from the main project's raw/ to avoid conflicts).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "raw_postproc"

# Minimum byte size to accept a downloaded file as valid.
MIN_BYTES: int = 4 * 1024   # 4 KB

# HTTP request headers — NOAA AWS prefers a proper User-Agent.
HEADERS: dict[str, str] = {"User-Agent": "weather-ml-postproc/1.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class GFSRun(NamedTuple):
    """A single GFS model run described by its date and UTC cycle."""
    date: str    # e.g. "20260101"
    cycle: str   # e.g. "00"

    @property
    def dir_url(self) -> str:
        """AWS S3 prefix for this run."""
        return f"{GFS_BASE_URL}/gfs.{self.date}/{self.cycle}/atmos"

    def file_stem(self, fhour: str) -> str:
        """Base filename without extension."""
        return f"gfs.{self.date}.t{self.cycle}z.{PRODUCT}.f{fhour}"


def recent_runs(n: int, cycle: str = "00") -> list[GFSRun]:
    """
    Return the `n` most recent GFS runs that are likely already published.

    GFS data appears on AWS about 4–5 hours after the model run starts.
    We look back from today, checking dates in reverse chronological order.
    We only grab the requested `cycle` (e.g. the 00z run) per calendar day
    so that we get one run per day spread over `n` consecutive days.
    """
    today = datetime.date.today()
    runs: list[GFSRun] = []
    day_offset = 0

    while len(runs) < n:
        d = today - datetime.timedelta(days=day_offset)
        runs.append(GFSRun(date=d.strftime("%Y%m%d"), cycle=cycle))
        day_offset += 1

    # Reverse so that the oldest is first — easier to track progress.
    return list(reversed(runs))


def fetch_idx(idx_url: str) -> list[str] | None:
    """
    Download the .idx index file and return its lines, or None on failure.

    The .idx file is only a few KB so we download it in full.
    """
    try:
        resp = requests.get(idx_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.text.splitlines()
    except requests.RequestException as exc:
        print(f"  [WARN] Could not fetch index: {exc}")
        return None


def parse_byte_range(lines: list[str], target_vars: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """
    Parse the .idx lines and return a list of (start_byte, end_byte) pairs
    for records matching `target_vars`.

    Each .idx line format:
        record_num:byte_offset:date_str:shortName:level:forecast_type:
    """
    ranges: list[tuple[int, int]] = []

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue

        short_name = parts[3].strip()
        level_desc = parts[4].strip()

        # Check if this record matches any of our target variables.
        for target_name, target_level in target_vars:
            if target_name == short_name and target_level in level_desc:
                start_byte = int(parts[1])
                # End byte = start of the NEXT record minus 1.
                if i + 1 < len(lines):
                    next_parts = lines[i + 1].split(":")
                    end_byte = int(next_parts[1]) - 1
                else:
                    # Last record — download to end of file.
                    end_byte = ""  # type: ignore[assignment]
                ranges.append((start_byte, end_byte))
                break  # no need to check other targets for this line

    return ranges


def download_byte_ranges(
    grib_url: str,
    ranges: list[tuple[int, int]],
    out_path: Path,
) -> bool:
    """
    Download multiple byte ranges from `grib_url` and concatenate them
    into a single GRIB2 file at `out_path`.

    Returns True on success, False on failure.
    """
    # If the file already looks complete, skip re-downloading.
    if out_path.exists() and out_path.stat().st_size > MIN_BYTES:
        print(f"  [SKIP] Already exists: {out_path.name}")
        return True

    chunks: list[bytes] = []
    for start, end in ranges:
        range_header = f"bytes={start}-{end}" if end != "" else f"bytes={start}-"
        try:
            resp = requests.get(
                grib_url,
                headers={**HEADERS, "Range": range_header},
                timeout=60,
            )
            # HTTP 206 = Partial Content (success), 200 = full file (fallback).
            if resp.status_code not in (200, 206):
                print(f"  [WARN] HTTP {resp.status_code} for range {range_header}")
                return False
            chunks.append(resp.content)
        except requests.RequestException as exc:
            print(f"  [WARN] Request failed: {exc}")
            return False

    total_bytes = sum(len(c) for c in chunks)
    if total_bytes < MIN_BYTES:
        print(f"  [WARN] Downloaded data too small ({total_bytes} bytes).")
        return False

    out_path.write_bytes(b"".join(chunks))
    print(f"  [OK]   {out_path.name}  ({total_bytes / 1024:.1f} KB)")
    return True


# ---------------------------------------------------------------------------
# Main download routine
# ---------------------------------------------------------------------------

def download_run(run: GFSRun, out_dir: Path) -> None:
    """Download all requested forecast hours for a single GFS run."""
    print(f"\n{'='*60}")
    print(f" GFS run  {run.date}  cycle {run.cycle}z")
    print(f"{'='*60}")

    for fhour in FORECAST_HOURS:
        stem = run.file_stem(fhour)
        grib_url = f"{run.dir_url}/{stem}"
        idx_url  = f"{grib_url}.idx"
        out_path = out_dir / stem

        print(f"\n  Forecast hour f{fhour}")
        print(f"  GRIB URL : {grib_url}")

        # Step A: Download the index file (a few KB).
        lines = fetch_idx(idx_url)
        if lines is None:
            print(f"  [SKIP] Index not available — run may not be published yet.")
            continue

        # Step B: Parse byte ranges for the wind variables.
        byte_ranges = parse_byte_range(lines, TARGET_VARS)
        if not byte_ranges:
            print(f"  [WARN] Could not find wind variables in index.")
            continue

        print(f"  Found {len(byte_ranges)} wind record(s) to download.")

        # Step C: Download only the wind bytes and save to disk.
        download_byte_ranges(grib_url, byte_ranges, out_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Downloading {N_RUNS} most recent GFS runs  (cycle {CYCLE}z)")
    print(f"Forecast hours per run: {FORECAST_HOURS}")

    runs = recent_runs(N_RUNS, cycle=CYCLE)

    for run in runs:
        download_run(run, OUTPUT_DIR)

    print("\n\nDone — all requested files have been processed.")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
