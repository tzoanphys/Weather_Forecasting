# config.py
from pathlib import Path
import torch

# --- Hyperparameters ---
INPUT_STEPS = 2
TARGET_OFFSET = 1
TRAIN_RATIO = 0.8
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
MODEL_FILENAME = "simple_wind_cnn.pth"

# --- Centralized Device Engine Allocation ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Centralized Paths Management ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure runtime directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_FILENAME

def get_train_val_indices(dataset_length: int):
    """Guarantees train/validation splits remain identical across scripts."""
    train_size = int(TRAIN_RATIO * dataset_length)
    if train_size == dataset_length:  # Edge case validation safety
        train_size = dataset_length - 1
        
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_length))
    return train_indices, val_indices