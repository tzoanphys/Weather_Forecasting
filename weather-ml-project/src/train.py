from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

#from simple_model import SimpleWindCNN
from model import BetterWindCNN
from dataset import load_wind_time_series, WindForecastDataset


# ============================================================
# Configuration
# ============================================================

INPUT_STEPS = 2
TARGET_OFFSET = 1
TRAIN_RATIO = 0.8
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
MODEL_FILENAME = "wind_forecast_cnn.pth"


# ============================================================
# Device selection
# ============================================================

def get_device() -> torch.device:
    """
    Select the best available device for training.

    Priority:
    1. Apple GPU via MPS
    2. NVIDIA GPU via CUDA
    3. CPU
    """
    if torch.backends.mps.is_available():
        print("Using Apple GPU with MPS.")
        return torch.device("mps")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using NVIDIA GPU: {gpu_name}")
        return torch.device("cuda")

    print("No GPU available. Using CPU.")
    return torch.device("cpu")


# ============================================================
# Paths
# ============================================================

def get_project_paths() -> tuple[Path, Path, Path]:
    """
    Build and return important project paths.

    Returns:
        project_root: root folder of the project
        processed_dir: folder containing processed NetCDF files
        model_dir: folder where trained models will be saved
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    model_dir = project_root / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    return project_root, processed_dir, model_dir


# ============================================================
# Data preparation
# ============================================================

def build_dataset(processed_dir: Path) -> WindForecastDataset:
    """
    Load wind data from disk and create the forecasting dataset.
    """
    data, _ = load_wind_time_series(processed_dir)

    dataset = WindForecastDataset(
        data=data,
        input_steps=INPUT_STEPS,
        target_offset=TARGET_OFFSET
    )

    if len(dataset) == 0:
        raise ValueError(
            "The dataset contains zero samples. "
            "Check the number of input files and the values of "
            f"INPUT_STEPS={INPUT_STEPS} and TARGET_OFFSET={TARGET_OFFSET}."
        )

    print(f"\nDataset successfully created with {len(dataset)} sample(s).")
    return dataset


def build_data_loaders(
    dataset: WindForecastDataset
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create training and validation DataLoaders.

    If the dataset is too small for splitting, use the full dataset
    for training only and skip validation.
    """
    dataset_size = len(dataset)

    if dataset_size < 2:
        print("\nDataset too small for train/validation split.")
        print("Using the full dataset for training only.")

        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )
        return train_loader, None

    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = dataset_size - train_size

    if train_size == 0:
        train_size = 1
        val_size = dataset_size - 1

    if val_size == 0:
        val_size = 1
        train_size = dataset_size - 1

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_batch_size = min(BATCH_SIZE, len(train_dataset))
    val_batch_size = min(BATCH_SIZE, len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False
    )

    print(f"\nTraining samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")

    return train_loader, val_loader


# ============================================================
# Model creation
# ============================================================

def build_model(dataset: WindForecastDataset, device: torch.device) -> nn.Module:
    """
    Create the CNN model using the dataset sample shapes.
    """
    sample_x, sample_y = dataset[0]

    in_channels = sample_x.shape[0]
    out_channels = sample_y.shape[0]

    print(f"\nModel input channels : {in_channels}")
    print(f"Model output channels: {out_channels}")

    model = BetterWindCNN(
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    return model


# ============================================================
# Training and validation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train the model for one epoch and return the average training loss.
    """
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluate the model for one epoch and return the average validation loss.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            running_loss += loss.item()

    return running_loss / len(data_loader)


def train_model() -> None:
    """
    Main training pipeline.
    """
    _, processed_dir, model_dir = get_project_paths()
    device = get_device()

    print(f"\nUsing data from: {processed_dir}")
    print(f"Models will be saved in: {model_dir}")

    dataset = build_dataset(processed_dir)
    train_loader, val_loader = build_data_loaders(dataset)
    model = build_model(dataset, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        if val_loader is not None:
            val_loss = validate_one_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f}"
            )

    model_path = model_dir / MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    print("\nTraining finished.")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    train_model()