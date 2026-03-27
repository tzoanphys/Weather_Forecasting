from pathlib import Path
import requests
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import re 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader, random_split



#__________________________________________________________________________________#
# Download GFS data for a specific date and cycle, saving to the project directory.#
#__________________________________________________________________________________#

# NOAA GFS public AWS file download configuration
DATE = "20250316"
CYCLE = "00"
# Forecast hours (every 6 hours)
FORECAST_HOURS = ["000", "006", "012", "018", "024"]
#project root = Path(__file__).resolve().parent.parent
project_root = Path(__file__).resolve().parent.parent
#Save into weather-ml-project/data/raw
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)


headers = {
    "User-Agent": "Mozilla/5.0" }

for forecast in FORECAST_HOURS:
    filename= f"gfs.t{CYCLE}z.pgrb2.0p25.f{forecast}"
    url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{DATE}/{CYCLE}/atmos/{filename}"
    output_path = output_dir / filename
    print("\n----------------------------------------")
    print(f"Downloading: {filename}")
    print(url)
    if output_path.exists():
        print(f"Already exists, skipping: {output_path}")
        continue
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("Download complete.")
        print(f"Saved to: {output_path}")
        print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB") 

    except Exception as e:
        print(f"Failed to download {filename}")
        print(e)                
print("\nAll downloads finished.")

input_file = project_root / "data" / "processed" / "belgium_wind_subset.nc"

# Plot the wind speed map for Belgium
#plt.figure(figsize=(10, 6))
#ds = xr.open_dataset(input_file, decode_times=False)
#wind_speed = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
#wind_speed.name = "wind_speed"         
#wind_speed.plot(cmap="viridis", cbar_kwargs={"label": "wind speed [m/s]"})
#plt.title("Belgium 10 m Wind Speed", fontsize=14)
#plt.xlabel("longitude [degrees_east]")
#plt.ylabel("latitude [degrees_north]")
#plt.tight_layout()
#output_path = project_root / "outputs" / "figures" / "wind_speed_preview.png"
#output_path.parent.mkdir(parents=True, exist_ok=True)
#plt.show()



#__________________________________________________________________________________#
#   Preprocess the downloaded GFS data to extract 10 m wind components, subset to Belgium,#
#__________________________________________________________________________________#

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

raw_files = sorted(raw_dir.glob("gfs.t00z.pgrb2.0p25.f*"))
print("\n----------------------------------------")
print("\n-Preprocessing GFS data---")

print(f"Found {len(raw_files)} raw files.") 

if not raw_files:
    raise FileNotFoundError(f"No raw GFS files found in {raw_dir}")

for raw_file in raw_files:
    print("\n" + "-" * 40)
    print(f"Processing file: {raw_file}")

    wind_ds = xr.open_dataset(
        raw_file, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 10,
            },
            "indexpath": ""
        }
    )
    print("Opened dataset with cfgrib.")

    if "u10" not in wind_ds or "v10" not in wind_ds.data_vars:
        print(f"Warning: u10 or v10 not found. Skipping.")
        continue

    belgium_ds = wind_ds.sel(
        longitude=slice(2, 7),
        latitude=slice(49, 52)
    )
    print("Subsetted to Belgium region.")

    coords_to_drop = [c for c in belgium_ds.coords if c not in ["latitude", "longitude", "time"]]
    belgium_ds = belgium_ds.drop_vars(coords_to_drop)
    print("Dropped unnecessary coordinates.")

    output_file = processed_dir / f"{raw_file.name}_belgium.nc"
    belgium_ds.to_netcdf(output_file, engine="netcdf4")
    print(f"Saved: {output_file.name}  ({output_file.stat().st_size / 1024:.1f} KB)")

#__________________________________________________________________________________#
#   Dataset 
#__________________________________________________________________________________#

def extract_forecast_hour(file_path):
    match = re.search(r"f(\d{3})", file_path.name)
    if match is None:
        raise ValueError(f"Filename does not contain forecast hour in expected format: {file_path.name}")   
    return int(match.group(1))


def load_wind_time_series(processed_dir):
    files =sorted(
        processed_dir.glob("gfs.t00z.pgrb2.0p25.f*_belgium.nc"),
        key=extract_forecast_hour
    )

    if not files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")
    
    print(f"Found {len(files)} processed files for loading.")
    
    time_steps = []
    latitudes = None    
    longitudes = None   

    for i, file_path in enumerate(files):
        ds=xr.open_dataset(file_path, decode_times=False)

        u10 = ds["u10"].values
        v10 = ds["v10"].values      

        if i == 0:
            latitudes = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
            longitudes = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values   


        state =np.stack([u10, v10], axis=0)
        time_steps.append(state)    

    data = np.stack(time_steps, axis=0)

    return data, files, latitudes, longitudes

class WindForecastDataset(Dataset):

    def __init__(self, data, input_steps=4, target_offset=1):
        self.data          = data.astype(np.float32)
        self.input_steps   = input_steps
        self.target_offset = target_offset
        self.samples       = []

        max_start = len(self.data) - input_steps - target_offset + 1

        for start_idx in range(max_start):
            input_start = start_idx
            input_end   = start_idx + input_steps
            target_idx  = input_end + target_offset - 1
            self.samples.append((input_start, input_end, target_idx))

        print(f"Dataset ready: {len(self.samples)} sample(s) created.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_start, input_end, target_idx = self.samples[idx]

        x = self.data[input_start:input_end]
        y = self.data[target_idx]

        input_steps, channels, lat, lon = x.shape
        x = x.reshape(input_steps * channels, lat, lon)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


#__________________________________________________________________________________#
# Test the dataset                                                                  #
#__________________________________________________________________________________#

data, files, latitudes, longitudes = load_wind_time_series(processed_dir)

dataset = WindForecastDataset(data, input_steps=4, target_offset=1)

if len(dataset) > 0:
    x, y = dataset[0]
    print("\nFirst sample shapes:")
    print("  Input  x:", x.shape)
    print("  Target y:", y.shape)


#__________________________________________________________________________________#   Model architecture                                                                #
#  Model   architecture                                                                #    
#__________________________________________________________________________________#    

class SimpleWindCNN(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)



#__________________________________________________________________________________#   Test the model architecture                                                        #
#        TRAIN DATA
#__________________________________________________________________________________#


# Confugurations
INPUT_STEPS = 2 #how many past time steps
TARGET_OFFSET = 1 # how far in the future to predict
TRAIN_RATIO = 0.8 # training percentage
BATCH_SIZE = 4 # samples per step
LEARNING_RATE = 1e-3 #spead of learnig
NUM_EPOCHS = 10 #number of training iterations
MODEL_FILE = project_root / "outputs" / "models" / "best_model.pth"


def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple GPU with MPS.")
        return torch.device("mps")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using NVIDIA GPU: {gpu_name}")
        return torch.device("cuda")

    print("No GPU available. Using CPU.")
    return torch.device("cpu")

