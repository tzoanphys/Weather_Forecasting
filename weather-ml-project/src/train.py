# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from simple_model import SimpleWindCNN
from dataset import load_wind_time_series, WindForecastDataset

print("Using device:", config.DEVICE)

# --- Load Data ---
data, *_ = load_wind_time_series(config.PROCESSED_DIR)

dataset = WindForecastDataset(
    data=data,
    input_steps=config.INPUT_STEPS,
    target_offset=config.TARGET_OFFSET
)
print("Total samples:", len(dataset))

# --- Consistent Split Separation ---
train_indices, val_indices = config.get_train_val_indices(len(dataset))
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# --- Setup Architecture & Optimizers ---
sample_x, sample_y = dataset[0]
model = SimpleWindCNN(in_channels=sample_x.shape[0], out_channels=sample_y.shape[0]).to(config.DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# --- Training / Verification Loop ---
best_val_loss = float("inf")

for epoch in range(config.NUM_EPOCHS):
    # Training Stage
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
        
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation Stage
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            predictions = model(x_batch)
            val_loss += criterion(predictions, y_batch).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.MODEL_PATH)
        print("Saved best model state.")

print("Training cycle complete. Optimal Model Saved Target:", config.MODEL_PATH)