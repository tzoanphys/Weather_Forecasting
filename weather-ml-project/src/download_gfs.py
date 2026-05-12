from pathlib import Path
import requests

# Date of NOAA forecast
DATE = "20250310"

# Forecast cycle
CYCLE = "00"

# Forecast hours
FORECAST_HOURS = ["000", "006", "012", "018", "024"]

# Create raw data folder
project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Download files
for forecast_hour in FORECAST_HOURS:

    # NOAA filename
    noaa_file = f"gfs.t{CYCLE}z.pgrb2.0p25.f{forecast_hour}"

    # NOAA URL
    url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{DATE}/{CYCLE}/atmos/{noaa_file}"

    # Local file path
    local_file = raw_dir / f"gfs.{DATE}.t{CYCLE}z.pgrb2.0p25.f{forecast_hour}"

    # Skip if file already exists
    if local_file.exists():
        print("Already downloaded:")
        print(local_file)
        continue

    print("Downloading:")
    print(url)

    # Download file
    response = requests.get(url)

    # Stop if request failed
    response.raise_for_status()

    # Save file
    with open(local_file, "wb") as file:
        file.write(response.content)

    print("Saved:")
    print(local_file)


#_____________________________________
#download_gfs.py
#
#Goal:
#Download raw NOAA GFS forecast files.
#
#Main steps:
#1. Choose date and forecast hours
#2. Build NOAA URL
#3. Download files using requests
#4. Save files in data/raw
#5. Skip download if file already exists
