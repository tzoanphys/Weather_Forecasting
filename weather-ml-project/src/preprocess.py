from pathlib import Path
import xarray as xr

# --------------------------------------------
# Paths
# --------------------------------------------
project_root = Path(__file__).resolve().parent.parent

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

# Find all downloaded GFS files
raw_files = sorted(raw_dir.glob("gfs.????????.t00z.pgrb2.0p25.f*"))

print("Number of raw files found:", len(raw_files))

if not raw_files:
    raise FileNotFoundError("No raw GFS files found in data/raw")


# --------------------------------------------
# Process each raw file
# --------------------------------------------
for raw_file in raw_files:

    print("Processing file:")
    print(raw_file)

    # Open only 10-meter wind data from the GRIB2 file
    wind_ds = xr.open_dataset(
        raw_file,
        engine="cfgrib", #GRIB2 is a meteorological format
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "heightAboveGround", #select variables measured above the ground, 10metters
                "level": 10,
            },
            "indexpath": ""
        }
    )

    print("Opened 10-meter wind data")

    # Check that u10 and v10 exist
    if "u10" not in wind_ds.data_vars or "v10" not in wind_ds.data_vars:
        print("u10 or v10 not found, skipping this file")
        continue

    # Select Belgium region
    belgium_ds = wind_ds.sel(
        longitude=slice(2, 7),
        latitude=slice(52, 49)
    )

    print("Selected Belgium region")

    # Remove metadata that can cause saving problems
    belgium_ds.attrs = {}

    for variable in belgium_ds.data_vars:
        belgium_ds[variable].attrs = {} #Remove metadata from each variable.

    # Save processed file
    output_file = processed_dir / f"{raw_file.name}_belgium.nc"

    belgium_ds.to_netcdf(output_file, engine="netcdf4")

    print("Saved processed file:")
    print(output_file)

print("All preprocessing finished")




#_________________________________________________________________________
#preprocess.py takes the raw downloaded NOAA files and makes them easier for machine learning.
#
# GOALS:
#
#1. Reads raw GFS GRIB2 files from data/raw
#2. Extracts 10-meter wind data
#3. Selects only the Belgium region
#4. Saves clean NetCDF files in data/processed

#   raw GRIB2 files
#         ↓
#   extract u10 and v10
#		↓
#	select Belgium
#		↓
#	save as NetCDF
