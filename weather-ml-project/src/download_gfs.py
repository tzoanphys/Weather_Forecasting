from pathlib import Path
import requests

# -----------------------------------------
# NOAA GFS public AWS file download
# -----------------------------------------

# Add or remove dates here to control how much data you download.
# Each date = 5 files (~250–500 MB total). More dates = better model.
DATES = [
    "20250310", "20250311", "20250312", "20250313",
    "20250314", "20250315", "20250316", "20250317",
]
CYCLE = "00"

# Forecast hours (every 6 hours)
FORECAST_HOURS = ["000", "006", "012", "018", "024"]

# Always point to the project root
project_root = Path(__file__).resolve().parent.parent

# Save into weather-ml-project/data/raw
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------------------
# Download loop
# -----------------------------------------
for date in DATES:
    for forecast in FORECAST_HOURS:

        # Include date in filename so files from different dates don't overwrite each other
        filename    = f"gfs.{date}.t{CYCLE}z.pgrb2.0p25.f{forecast}"
        noaa_name   = f"gfs.t{CYCLE}z.pgrb2.0p25.f{forecast}"
        url         = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{CYCLE}/atmos/{noaa_name}"
        output_path = output_dir / filename

        print("\n----------------------------------------")
        print(f"Downloading: {filename}")

        if output_path.exists():
            print(f"Already exists, skipping: {output_path}")
            continue

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            print("Download complete.")
            print(f"Saved to: {output_path}")
            print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("\nAll downloads finished.")