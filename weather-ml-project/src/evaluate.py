from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

# -----------------------------------------
# Paths
# -----------------------------------------
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "processed" / "belgium_wind_subset.nc"
figures_dir = project_root / "outputs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading processed dataset from: {input_file}")

# -----------------------------------------
# Load dataset without decoding time metadata
# -----------------------------------------
ds = xr.open_dataset(input_file, decode_times=False)

print("\nDataset:")
print(ds)

# -----------------------------------------
# Plot u10
# -----------------------------------------
plt.figure(figsize=(8, 5))
ds["u10"].plot()
plt.title("Belgium 10 m East-West Wind Component (u10)")
u10_path = figures_dir / "u10_belgium.png"
plt.savefig(u10_path, bbox_inches="tight")
plt.close()

print(f"Saved u10 plot to: {u10_path}")

# -----------------------------------------
# Plot v10
# -----------------------------------------
plt.figure(figsize=(8, 5))
ds["v10"].plot()
plt.title("Belgium 10 m North-South Wind Component (v10)")
v10_path = figures_dir / "v10_belgium.png"
plt.savefig(v10_path, bbox_inches="tight")
plt.close()
plt.show()

print(f"Saved v10 plot to: {v10_path}")