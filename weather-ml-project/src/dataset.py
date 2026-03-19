from pathlib import Path
import re
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

# --------------------------------------------
# Helper: extract forecast hour from filename
# Example:
# gfs.t00z.pgrb2.0p25.f006_belgium.nc -> 6
# --------------------------------------------
def extract_forecast_hour(file_path):
    match = re.search(r"\.f(\d{3})_belgium\.nc$", file_path.name)
    if match is None:
        raise ValueError(f"Could not extract forecast hour from filename: {file_path.name}")
    return int(match.group(1))


# --------------------------------------------
# Load all processed files and stack them
# --------------------------------------------
def load_wind_time_series(processed_dir):
    files = sorted(
        processed_dir.glob("gfs.t00z.pgrb2.0p25.f*_belgium.nc"),
        key=extract_forecast_hour
    )

    if not files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")

    print("Processed files found in time order:")
    for f in files:
        print("  ", f.name)

    time_steps = []

    for file_path in files:
        ds = xr.open_dataset(file_path, decode_times=False)

        # Extract u10 and v10 as numpy arrays
        u10 = ds["u10"].values   # shape: [lat, lon]
        v10 = ds["v10"].values   # shape: [lat, lon]

        # Stack variables into channels: [2, lat, lon]
        state = np.stack([u10, v10], axis=0)

        time_steps.append(state)

    # Final shape: [time, channels, lat, lon]
    data = np.stack(time_steps, axis=0)

    print("\nStacked time-series shape:")
    print(data.shape)

    return data, files


# --------------------------------------------
# PyTorch Dataset for forecasting
# --------------------------------------------
class WindForecastDataset(Dataset):
    def __init__(self, data, input_steps=4, target_offset=1):
        """
        data shape: [time, channels, lat, lon]

        input_steps = number of past timesteps used as input
        target_offset = how far ahead the target is
        """
        self.data = data.astype(np.float32)
        self.input_steps = input_steps
        self.target_offset = target_offset

        self.samples = []

        max_start = len(self.data) - input_steps - target_offset + 1

        for start_idx in range(max_start):
            input_start = start_idx
            input_end = start_idx + input_steps
            target_idx = input_end + target_offset - 1

            self.samples.append((input_start, input_end, target_idx))

        print("\nNumber of dataset samples created:")
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_start, input_end, target_idx = self.samples[idx]

        # Input shape before reshape:
        # [input_steps, channels, lat, lon]
        x = self.data[input_start:input_end]

        # Target shape:
        # [channels, lat, lon]
        y = self.data[target_idx]

        # Merge time and channels for CNN input:
        # [input_steps * channels, lat, lon]
        input_steps, channels, lat, lon = x.shape
        x = x.reshape(input_steps * channels, lat, lon)

        # Convert to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


# --------------------------------------------
# Main test block
# --------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    data, files = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(data, input_steps=4, target_offset=1)

    if len(dataset) > 0:
        x, y = dataset[0]

        print("\nFirst sample shapes:")
        print("Input x shape:", x.shape)
        print("Target y shape:", y.shape)
    else:
        print("\nNo samples could be created.")
        print("You probably need more time steps.")