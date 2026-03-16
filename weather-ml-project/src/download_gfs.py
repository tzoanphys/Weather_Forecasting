from pathlib import Path
import requests

# -----------------------------------------
# NOAA GFS public AWS file download
# -----------------------------------------

DATE = "20250316"
CYCLE = "00"
FORECAST = "000"   # analysis / f000

filename = f"gfs.t{CYCLE}z.pgrb2.0p25.f{FORECAST}"
url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{DATE}/{CYCLE}/atmos/{filename}"

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / filename

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Downloading:")
print(url)
print("Saving to:")
print(output_path)

with requests.get(url, headers=headers, stream=True, timeout=60) as response:
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

print("Download complete.")
print(f"Saved file size: {output_path.stat().st_size / (1024**2):.2f} MB")