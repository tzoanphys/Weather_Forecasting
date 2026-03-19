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
raw_files = sorted(raw_dir.glob("gfs.t00z.pgrb2.0p25.f*"))

print(f"Found {len(raw_files)} raw files.")

if not raw_files:
    raise FileNotFoundError(f"No raw GFS files found in {raw_dir}")

# --------------------------------------------
# Process each file
# --------------------------------------------
for raw_file in raw_files:
    print("\n" + "-" * 60)
    print(f"Loading data from: {raw_file}")

    # Open only the 10 m wind dataset directly
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

    print("Opened 10 m wind dataset.")

    # Check variables
    if "u10" not in wind_ds.data_vars or "v10" not in wind_ds.data_vars:
        print(f"Skipping {raw_file.name}: u10/v10 not found.")
        continue

    # Crop Belgium region
    belgium_ds = wind_ds.sel(
        longitude=slice(2, 7),
        latitude=slice(52, 49)
    )

    print("Belgium subset created.")

    # Make time coordinates easier to save
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

    # Remove problematic attributes
    for var_name in belgium_ds.data_vars:
        belgium_ds[var_name].attrs = {}

    belgium_ds.attrs = {}

    # Save one processed file per raw file
    output_file = processed_dir / f"{raw_file.stem}_belgium.nc"
    belgium_ds.to_netcdf(output_file, engine="netcdf4")

    print(f"Saved processed dataset to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")

print("\nAll preprocessing finished.")