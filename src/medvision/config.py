from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    num_workers: int = Field(default=0, ge=0)
    num_threads: int = Field(default=2, ge=1)
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts")
    train_limit: int | None = Field(default=None, ge=1)
    val_limit: int | None = Field(default=None, ge=1)


def load_config(path: Path) -> TrainingConfig:
    # Relative data/output paths are resolved from the working directory.
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text()))
