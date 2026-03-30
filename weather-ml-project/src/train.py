from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import BetterWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# -----------------------------
# ️️⚙️ Settings
# -----------------------------
INPUT_STEPS = 2 # two past time steps as input
TARGET_OFFSET = 1 # predict one stp in the future
TRAIN_RATIO = 0.8
BATCH_SIZE = 4 # the model sees 4 batches each time
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MODEL_FILENAME = "wind_forecast_cnn.pth"


# -----------------------------
# ️️🗄️ Choose device
# -----------------------------
if torch.cuda.is_available(): # PyTorch is optimized to use NVIDIA GPUs when available. If you have an AMD GPU, PyTorch support is still experimental and may require additional setup.
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# -----------------------------
# Create paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
processed_dir = project_root / "data" / "processed"
model_dir = project_root / "saved_models"
model_dir.mkdir(parents=True, exist_ok=True)

print("Processed data folder:", processed_dir)
print("Model save folder:", model_dir)


# -----------------------------
# ️️📦 Load data
# -----------------------------
data, _, latitudes, longitudes = load_wind_time_series(processed_dir)

dataset = WindForecastDataset(
    data=data,
    input_steps=INPUT_STEPS,
    target_offset=TARGET_OFFSET
)

if len(dataset) == 0:
    raise ValueError("Dataset is empty.")

print("Number of samples in dataset:", len(dataset))


# -----------------------------
# ️️📊 Split data
# -----------------------------
train_size = int(TRAIN_RATIO * len(dataset)) #how many samples are created 
val_size = len(dataset) - train_size # the rest of validation

if train_size == 0:
    train_size = 1
    val_size = len(dataset) - 1

if val_size == 0:
    val_size = 1
    train_size = len(dataset) - 1

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# -----------------------------
#💥 Build model
# -----------------------------
sample_x, sample_y = dataset[0] # we start with the first sample first 

in_channels = sample_x.shape[0] #input 
out_channels = sample_y.shape[0] # target

print("Input channels:", in_channels)
print("Output channels:", out_channels)

#call the Convulation Neural Network model from model.py  
model = BetterWindCNN(
    in_channels=in_channels,
    out_channels=out_channels
).to(device)


# -----------------------------
# ⚖️ Loss and optimizer
# -----------------------------
criterion = nn.MSELoss() # choose the loss function, MSE (mean squared error) is common for regression problems 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # optimizer to update the weights of the model based on the computed gradients.
                                                                # Adam is a popular choice for training deep learning models


# -----------------------------
# 🚀 Training loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad() # clear old gradients and compute new ones

        predictions = model(x_batch) # forward pass : send the input to calculate predictions

        loss = criterion(predictions, y_batch) # compute gradients, it is callled backpropagation. 

        loss.backward() #backpropagation: compute gradients of the loss with respect to model parameters
        optimizer.step() # update model parameters based on computed gradients

        train_loss += loss.item() # compute the averager training loss across all batches 

    train_loss = train_loss / len(train_loader)

    model.eval() #==> put the model in evlautation mode. "Now i am not training, only testinng"
    val_loss = 0.0

    with torch.no_grad(): #valdation stage- no update gradients 
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch) # make predictions 
            loss = criterion(predictions, y_batch) # make validation error

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    ratio=val_loss/train_loss if train_loss > 0 else float('inf')
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Ratio: {ratio:.4f}")


# -----------------------------
# 🗃️ Save model
# -----------------------------
model_path = model_dir / MODEL_FILENAME
torch.save(model.state_dict(), model_path)

print("Training finished.")
print("Model saved at:", model_path)