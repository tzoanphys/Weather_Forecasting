from pathlib import Path
import numpy as np
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
# Compute wind speed
# -----------------------------------------
wind_speed = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
wind_speed.name = "wind_speed"

# -----------------------------------------
# Common plotting function
# -----------------------------------------
def save_field_plot(data_array, title, output_path, cbar_label):
    plt.figure(figsize=(10, 6))
    #data_array.plot(cmap="viridis",interpolation="bilinear",cbar_kwargs={"label": cbar_label})
    
    plot = data_array.plot(cbar_kwargs={"label": cbar_label})
    plt.title(title, fontsize=14)
    plt.xlabel("longitude [degrees_east]")
    plt.ylabel("latitude [degrees_north]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")

# -----------------------------------------
# Plot wind speed
# -----------------------------------------
speed_path = figures_dir / "wind_speed_belgium.png"
save_field_plot(
    wind_speed,
    "Belgium 10 m Wind Speed",
    speed_path,
    "wind speed [m/s]"
)

# -----------------------------------------
# Plot u10
# -----------------------------------------
u10_path = figures_dir / "u10_belgium.png"
save_field_plot(
    ds["u10"],
    "Belgium 10 m East-West Wind Component (u10)",
    u10_path,
    "u10 [m/s]"
)

# -----------------------------------------
# Plot v10
# -----------------------------------------
v10_path = figures_dir / "v10_belgium.png"
save_field_plot(
    ds["v10"],
    "Belgium 10 m North-South Wind Component (v10)",
    v10_path,
    "v10 [m/s]"
)