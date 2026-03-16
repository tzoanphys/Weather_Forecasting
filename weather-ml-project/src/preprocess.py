from pathlib import Path
import xarray as xr

#Path to the raw data of NOAA
file_path = Path("data/raw/gfs.t00z.pgrb2.0p25.f000")

print(f"Loading data from: {file_path}")

try:
    ds=xr.open_dataset(file_path, engine="cfgrib")

    print("\nData loaded successfully!")
    print(ds)

    print("\nDataset variables:")
    print(list(ds.data_vars))

    print("\nDataset coordinates:")
    print(list(ds.coords))

except Exception as e:
    print("\nCould not open the GRIB file as a single dataset.")
    print("This is normal for GRIB files.")
    print("\nError message:")
    print(e)