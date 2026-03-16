from pathlib import Path
import cfgrib
import xarray as xr
import numpy as np

# --------------------------------------------
# Paths
# --------------------------------------------
raw_file = Path("data/raw/gfs.t00z.pgrb2.0p25.f000")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {raw_file}")

# --------------------------------------------
# Open all GRIB datasets
# --------------------------------------------
datasets = cfgrib.open_datasets(raw_file)

print(f"Number of datasets found: {len(datasets)}")

# --------------------------------------------
# Find the dataset that contains u10 and v10
# --------------------------------------------
wind_ds = None

for i, ds in enumerate(datasets):
    vars_in_ds = list(ds.data_vars)
    if "u10" in vars_in_ds and "v10" in vars_in_ds:
        wind_ds = ds
        print(f"\nFound 10 m wind dataset at DATASET {i}")
        print(wind_ds)
        break

if wind_ds is None:
    raise ValueError("Could not find a dataset containing both u10 and v10.")

# --------------------------------------------
# Crop Belgium region
# --------------------------------------------
belgium_ds = wind_ds.sel(
    longitude=slice(2, 7),
    latitude=slice(52, 49)
)

print("\nBelgium subset:")
print(belgium_ds)

# --------------------------------------------
# Make time coordinates easier to save
# --------------------------------------------
if "time" in belgium_ds.coords:
    belgium_ds = belgium_ds.assign_coords(
        time=belgium_ds.time.astype("datetime64[s]")
    )

if "valid_time" in belgium_ds.coords:
    belgium_ds = belgium_ds.assign_coords(
        valid_time=belgium_ds.valid_time.astype("datetime64[s]")
    )

if "step" in belgium_ds.coords:
    belgium_ds = belgium_ds.assign_coords(
        step=belgium_ds.step.astype("timedelta64[s]")
    )

# --------------------------------------------
# Remove problematic attributes if needed
# --------------------------------------------
for var_name in belgium_ds.data_vars:
    belgium_ds[var_name].attrs = {}

belgium_ds.attrs = {}

# --------------------------------------------
# Save processed dataset
# --------------------------------------------
output_file = processed_dir / "belgium_wind_subset.nc"
belgium_ds.to_netcdf(output_file, engine="netcdf4")

print(f"\nSaved processed dataset to: {output_file}")
print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")