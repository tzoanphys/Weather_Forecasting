from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from simple_model import SimpleWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# -----------------------------
# Settings
# -----------------------------
INPUT_STEPS = 2
TARGET_OFFSET = 1

TRAIN_RATIO = 0.8
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

MODEL_FILENAME = "simple_wind_cnn.pth"


# -----------------------------
# Device: CUDA (Linux/Windows GPU) → MPS (Apple Silicon) → CPU
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)


# -----------------------------
# Paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent

processed_dir = project_root / "data" / "processed"
model_dir = project_root / "saved_models"

model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / MODEL_FILENAME


# -----------------------------
# Load data
# -----------------------------
data, *_ = load_wind_time_series(processed_dir)

dataset = WindForecastDataset(
    data=data,
    input_steps=INPUT_STEPS,
    target_offset=TARGET_OFFSET
)

print("Total samples:", len(dataset))


# -----------------------------
# Train / validation split
# -----------------------------
train_size = int(TRAIN_RATIO * len(dataset))

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(dataset)))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

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

model = SimpleWindCNN(
    in_channels=in_channels,
    out_channels=out_channels
).to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training loop
# -----------------------------
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    # ----- Training -----
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(x_batch)

        loss = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)


    # ----- Validation -----
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


    # ----- Print progress -----
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f}"
    )


    # ----- Save best model -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print("Saved best model")


print("Training finished")
print("Best validation loss:", best_val_loss)
print("Model saved at:", model_path)