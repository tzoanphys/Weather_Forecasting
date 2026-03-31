from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import json

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
NUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10  # stop if val loss does not improve for this many epochs
WEIGHT_DECAY = 1e-4           # L2 regularisation to reduce overfitting
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
output_dir = project_root / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print("Processed data folder:", processed_dir)
print("Model save folder:", model_dir)
print("Outputs folder:", output_dir)


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
if len(dataset) < 2:
    raise ValueError("Need at least 2 samples to create train/validation splits.")

train_size = int(TRAIN_RATIO * len(dataset))

if train_size == 0:
    train_size = 1

if train_size == len(dataset):
    train_size = len(dataset) - 1

val_size = len(dataset) - train_size

# Chronological split for forecasting: train on earlier samples, validate on later samples.
train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(dataset)))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

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
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# -----------------------------
# 🚀 Training loop
# -----------------------------
best_val_loss = float("inf")
best_epoch = 0
no_improve_count = 0
best_model_path = model_dir / MODEL_FILENAME

history = {
    "epochs": [],
    "train": {"mse": [], "mae": []},
    "val": {"mse": [], "mae": []},
    "ratio_val_to_train_mse": [],
    "best_epoch": None,
    "best_val_mse": None,
    "stopped_early": False,
}

def _batch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_mae = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad() # clear old gradients and compute new ones

        predictions = model(x_batch) # forward pass : send the input to calculate predictions

        loss = criterion(predictions, y_batch) # compute gradients, it is callled backpropagation. 

        loss.backward() #backpropagation: compute gradients of the loss with respect to model parameters
        optimizer.step() # update model parameters based on computed gradients

        train_loss += loss.item() # compute the averager training loss across all batches 
        train_mae += _batch_mae(predictions, y_batch).item()

    train_loss = train_loss / len(train_loader)
    train_mae = train_mae / len(train_loader)

    model.eval() #==> put the model in evlautation mode. "Now i am not training, only testinng"
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad(): #valdation stage- no update gradients 
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch) # make predictions 
            loss = criterion(predictions, y_batch) # make validation error

            val_loss += loss.item()
            val_mae += _batch_mae(predictions, y_batch).item()

    val_loss = val_loss / len(val_loader)
    val_mae = val_mae / len(val_loader)
    ratio = val_loss / train_loss if train_loss > 0 else float("inf")

    history["epochs"].append(epoch + 1)
    history["train"]["mse"].append(float(train_loss))
    history["train"]["mae"].append(float(train_mae))
    history["val"]["mse"].append(float(val_loss))
    history["val"]["mae"].append(float(val_mae))
    history["ratio_val_to_train_mse"].append(float(ratio))

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        no_improve_count = 0
        torch.save(model.state_dict(), best_model_path)
        marker = "  ✓ best"
    else:
        no_improve_count += 1
        marker = ""

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - "
        f"Train MSE: {train_loss:.6f} - Val MSE: {val_loss:.6f} - "
        f"Train MAE: {train_mae:.6f} - Val MAE: {val_mae:.6f} - "
        f"Ratio: {ratio:.4f}{marker}"
    )

    # Early stopping
    if no_improve_count >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}. No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
        history["stopped_early"] = True
        break


print(f"\nTraining finished. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}.")
print(f"Best model saved at: {best_model_path}")

history["best_epoch"] = int(best_epoch) if best_epoch else None
history["best_val_mse"] = float(best_val_loss) if best_epoch else None

history_path = output_dir / "training_history.json"
history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
print(f"Saved training history to: {history_path}")