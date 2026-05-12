from pathlib import Path

import requests

# ============================================================
# Download FULL GFS files ( 5 files)
# ============================================================
#
# Interview-friendly goal:
# - Download only a tiny dataset so the pipeline runs fast.
# - Use 1 date × 5 forecast hours = 5 GRIB2 files.
#
# These are the standard full GRIB2 files from the NOAA GFS public S3 bucket.
# ============================================================

# Pick one date you know exists in the NOAA archive.
# You can change this to any YYYYMMDD you want.
DATE = "20250310"

# Model cycle hour (00, 06, 12, 18). Keep it simple.
CYCLE = "00"

# 5 forecast lead times (0h, 6h, 12h, 18h, 24h) => exactly 5 files.
FORECAST_HOURS = ["000", "006", "012", "018", "024"]

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}
MIN_FILE_SIZE = 5 * 1024 * 1024  # 5 MB sanity check


def download_file(url: str, out_path: Path) -> None:
    """Download a URL to a local file (streaming, safe write)."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp_path.replace(out_path)


print(f"Downloading FULL GFS files for date={DATE} cycle={CYCLE}")
print(f"Total files to download: {len(FORECAST_HOURS)}")

for forecast in FORECAST_HOURS:
    filename = f"gfs.{DATE}.t{CYCLE}z.pgrb2.0p25.f{forecast}"
    noaa_name = f"gfs.t{CYCLE}z.pgrb2.0p25.f{forecast}"
    url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{DATE}/{CYCLE}/atmos/{noaa_name}"

    out_path = output_dir / filename
    print("\n" + "-" * 60)
    print(f"File: {filename}")

    if out_path.exists() and out_path.stat().st_size >= MIN_FILE_SIZE:
        size_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"Already downloaded, skipping ({size_mb:.1f} MB): {out_path}")
        continue

    print(f"Downloading from: {url}")
    download_file(url, out_path)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

print("\nAll downloads finished (5 files).")