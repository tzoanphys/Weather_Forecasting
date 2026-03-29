from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import BetterWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# -----------------------------
# Settings
# -----------------------------
INPUT_STEPS = 2
TARGET_OFFSET = 1
TRAIN_RATIO = 0.8
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MODEL_FILENAME = "wind_forecast_cnn.pth"


# -----------------------------
# Choose device
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
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
# Load data
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
# Split data
# -----------------------------
train_size = int(TRAIN_RATIO * len(dataset))
val_size = len(dataset) - train_size

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
# Build model
# -----------------------------
sample_x, sample_y = dataset[0]

in_channels = sample_x.shape[0]
out_channels = sample_y.shape[0]

print("Input channels:", in_channels)
print("Output channels:", out_channels)

model = BetterWindCNN(
    in_channels=in_channels,
    out_channels=out_channels
).to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(x_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    ratio=val_loss/train_loss if train_loss > 0 else float('inf')
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Ratio: {ratio:.4f}")


# -----------------------------
# Save model
# -----------------------------
model_path = model_dir / MODEL_FILENAME
torch.save(model.state_dict(), model_path)

print("Training finished.")
print("Model saved at:", model_path)