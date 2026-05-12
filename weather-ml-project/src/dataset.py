from pathlib import Path
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------
# Load all processed wind files
# ------------------------------------------------------------
def load_wind_time_series(processed_dir):

    # Find all processed Belgium NetCDF files
    files = sorted(processed_dir.glob("*_belgium.nc"))

    print("Number of processed files found:", len(files))

    if not files:
        raise FileNotFoundError("No processed files found in data/processed")

    time_steps = []
    latitudes = None
    longitudes = None

    # Open each file and read u10 and v10
    for file_path in files:
        #print("Reading file:")
        #print(file_path)

        ds = xr.open_dataset(file_path, decode_times=False)
        if latitudes is None:
            latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
            longitudes = np.asarray(ds["longitude"].values, dtype=np.float64)

        u10 = ds["u10"].values
        v10 = ds["v10"].values
        ds.close()

        # Put u10 and v10 together
        state = np.stack([u10, v10], axis=0)

        # Add this time step to the list
        time_steps.append(state)

    # Convert list to one NumPy array
    data = np.stack(time_steps, axis=0)

    print("Final data shape:")
    print(data.shape)

    return data, files, latitudes, longitudes


# ------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------
class WindForecastDataset(Dataset):

    def __init__(self, data, input_steps=2, target_offset=1):

        self.data = data.astype(np.float32)
        self.input_steps = input_steps
        self.target_offset = target_offset
        self.samples = []

        # Create input-target pairs
        max_start = len(self.data) - input_steps - target_offset + 1

        for start in range(max_start):

            input_start = start
            input_end = start + input_steps
            target_index = input_end + target_offset - 1

            self.samples.append((input_start, input_end, target_index))

        print("Number of samples created:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        input_start, input_end, target_index = self.samples[index]

        # x = previous wind fields
        x = self.data[input_start:input_end]

        # y = future wind field
        y = self.data[target_index]

        # Combine time steps and channels into one channel dimension
        input_steps, channels, lat, lon = x.shape
        x = x.reshape(input_steps * channels, lat, lon)

        # Convert NumPy arrays to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


# ------------------------------------------------------------
# Test this file alone
# ------------------------------------------------------------
if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    data, *_ = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(data, input_steps=2, target_offset=1)

    x, y = dataset[0]

    print("Input shape:")
    print(x.shape)

    print("Target shape:")
    print(y.shape)
    
    
#_________________________________________________________
# Notes:
# this build the Pytorch dataset. It loads the processed NETCDF eind files , extract u10 and v10
# stacks them to a time series and create supervised samplesd data.
# Each sample use previous winf fiels as inpput and next as target
#		 GOAL:
# 1.Create ML samples
# 2.  processed NetCDF files
# 3 Pythorch dataset
# 4. past wind fields → future wind field

    
        
        

	    				
			
		
	   	
	
		
