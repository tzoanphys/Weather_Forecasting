from pathlib import Path
import datetime
import requests

# -----------------------------------------
# NOAA GFS public AWS — wind-only download
#
# Instead of downloading full 300-500 MB GRIB2
# files, this script:
#   1. Fetches the lightweight .idx index file
#   2. Finds the byte offsets for UGRD and VGRD
#      at 10 m above ground (u10 / v10)
#   3. Downloads only those bytes (~5-15 MB)
#
# The preprocessor (preprocess.py) is unchanged
# because cfgrib reads partial GRIB2 files fine.
# -----------------------------------------

# Add or remove dates here to control how much data you download.
# Section 1: Seasonal diversity dates from 2025 for varied weather regimes.
_DIVERSITY_DATES = [
    # Winter 2024/2025
    "20250101", "20250115",
    # February 2025
    "20250201", "20250215",
    # March 2025 (original dates kept)
    "20250310", "20250311", "20250312", "20250313",
    "20250314", "20250315", "20250316", "20250317",
    # Spring 2025
    "20250415", "20250501",
    # Summer 2025
    "20250601", "20250715",
    # Autumn 2025
    "20250901", "20251015",
    # Winter 2025
    "20251201",
]

# Section 2: Every day from 1 Jan 2026 to 29 Mar 2026 (88 days).
_start = datetime.date(2026, 1, 1)
_end   = datetime.date(2026, 3, 29)
_daily_2026 = [
    (_start + datetime.timedelta(days=i)).strftime("%Y%m%d")
    for i in range((_end - _start).days + 1)
]

# Merge both lists, dropping any duplicates, preserving order.
_seen = set()
DATES = []
for _d in _DIVERSITY_DATES + _daily_2026:
    if _d not in _seen:
        _seen.add(_d)
        DATES.append(_d)

print(f"Total dates to process: {len(DATES)}  ({len(DATES) * 5} files)")
CYCLE = "00"

# Forecast hours (every 6 hours)
FORECAST_HOURS = ["000", "006", "012", "018", "024"]

# Variables to extract from each GRIB file
TARGET_VARS = [
    ("UGRD", "10 m above ground"),  # u10
    ("VGRD", "10 m above ground"),  # v10
]

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

# Wind-only files are a few MB, not 300-500 MB
MIN_FILE_SIZE = 10 * 1024  # 10 KB


def fetch_idx(idx_url):
    """Download and return lines of the .idx index file, or None on failure."""
    try:
        resp = requests.get(idx_url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text.splitlines()
    except Exception as e:
        print(f"  Could not fetch index: {e}")
        return None


def find_byte_ranges(idx_lines, targets):
    """
    Parse .idx lines and return (start, end) byte ranges for each target.
    Returns empty list if any target variable is not found.
    """
    records = []
    for line in idx_lines:
        parts = line.split(":")
        if len(parts) < 6:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        var   = parts[3].strip()
        level = parts[4].strip()
        records.append((offset, var, level))

    ranges = []
    for target_var, target_level in targets:
        found = False
        for i, (offset, var, level) in enumerate(records):
            if var == target_var and target_level in level:
                end = records[i + 1][0] - 1 if i + 1 < len(records) else -1
                ranges.append((offset, end))
                found = True
                break
        if not found:
            print(f"  Variable {target_var} at '{target_level}' not found in index.")
            return []

    return ranges


def download_byte_range(url, start, end):
    """Download a specific byte range from a URL."""
    range_header = f"bytes={start}-{end}" if end != -1 else f"bytes={start}-"
    try:
        resp = requests.get(
            url,
            headers={**headers, "Range": range_header},
            timeout=60
        )
        if resp.status_code not in (200, 206):
            print(f"  Unexpected status {resp.status_code} for range {range_header}")
            return None
        return resp.content
    except Exception as e:
        print(f"  Failed byte-range download ({range_header}): {e}")
        return None


# -----------------------------------------
# Download loop
# -----------------------------------------
for date in DATES:
    for forecast in FORECAST_HOURS:

        filename    = f"gfs.{date}.t{CYCLE}z.pgrb2.0p25.f{forecast}"
        noaa_name   = f"gfs.t{CYCLE}z.pgrb2.0p25.f{forecast}"
        base_url    = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{CYCLE}/atmos/{noaa_name}"
        idx_url     = base_url + ".idx"
        output_path = output_dir / filename
        tmp_path    = output_path.with_suffix(".tmp")

        print("\n----------------------------------------")
        print(f"Downloading (wind-only): {filename}")

        if output_path.exists() and output_path.stat().st_size >= MIN_FILE_SIZE:
            print(f"Already exists, skipping: {output_path}")
            continue

        # 1. Fetch the index file
        idx_lines = fetch_idx(idx_url)
        if idx_lines is None:
            print(f"  Skipping {filename}: index unavailable.")
            continue

        # 2. Locate byte ranges for u10 and v10
        byte_ranges = find_byte_ranges(idx_lines, TARGET_VARS)
        if not byte_ranges:
            print(f"  Skipping {filename}: could not locate u10/v10 in index.")
            continue

        # 3. Download each field and concatenate
        chunks = []
        ok = True
        for (start, end), (var, level) in zip(byte_ranges, TARGET_VARS):
            size_str = f"{end - start + 1:,} bytes" if end != -1 else "to EOF"
            print(f"  Fetching {var} @ {level}  (bytes {start}–{end if end != -1 else 'EOF'}  {size_str})")
            data = download_byte_range(base_url, start, end)
            if data is None:
                ok = False
                break
            chunks.append(data)

        if not ok:
            print(f"  Failed to download all fields for {filename}, skipping.")
            if tmp_path.exists():
                tmp_path.unlink()
            continue

        # 4. Write the concatenated GRIB records to disk
        try:
            with open(tmp_path, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
            tmp_path.rename(output_path)
            size_mb = output_path.stat().st_size / (1024 ** 2)
            print(f"  Saved: {output_path}  ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  Write error for {filename}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

print("\nAll downloads finished.")