from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PostprocessingConfig:
    """
    Small configuration object.

    The goal is to keep all important paths and settings in one place,
    so a student can quickly see what the pipeline needs.
    """

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    outputs_dir: Path = data_dir / "outputs"
    model_dir: Path = project_root / "saved_models" / "postprocessing"

    # Europe bounding box
    west: float = -10.0
    east: float = 30.0
    south: float = 35.0
    north: float = 60.0

    # Observation download choices
    max_stations: int = 25
    obs_days: int = 7

    # Model settings
    test_ratio: float = 0.2
    learning_rate: float = 1e-3
    hidden_size: int = 32
    batch_size: int = 32
    epochs: int = 60
    random_seed: int = 42

    def ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
