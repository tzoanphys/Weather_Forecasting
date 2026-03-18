from pathlib import Path
import xarray as xr

# --------------------------------------------
# Paths
# --------------------------------------------
project_root = Path(__file__).resolve().parent.parent

raw_file = project_root / "data" / "raw" / "gfs.t00z.pgrb2.0p25.f000"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {raw_file}")

# --------------------------------------------
# Open only the 10 m wind dataset directly
# --------------------------------------------
wind_ds = xr.open_dataset(
    raw_file,
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {
            "typeOfLevel": "heightAboveGround",
            "level": 10,
        },
        "indexpath": ""
    }
)

print("\nOpened 10 m wind dataset:")
print(wind_ds)

# --------------------------------------------
# Check variables
# --------------------------------------------
if "u10" not in wind_ds.data_vars or "v10" not in wind_ds.data_vars:
    raise ValueError("Could not find u10 and v10 in the opened dataset.")

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