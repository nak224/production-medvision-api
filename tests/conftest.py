from pathlib import Path

import numpy as np
import pytest
import torch
from medmnist import INFO

from medvision.model import build_model, save_checkpoint


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def synthetic_data(tmp_path: Path, monkeypatch) -> Path:
    """Small synthetic archive for offline pipeline tests, never benchmark results."""
    root = tmp_path / "data"
    root.mkdir()
    arrays = {}
    # MedMNIST validates split lengths; the fixture has its own declared sizes.
    monkeypatch.setitem(INFO["pathmnist"], "n_samples", {"train": 18, "val": 9, "test": 9})
    rng = np.random.default_rng(42)
    for split, repeats in [("train", 2), ("val", 1), ("test", 1)]:
        labels = np.tile(np.arange(9), repeats).astype(np.int64).reshape(-1, 1)
        arrays[f"{split}_images"] = rng.integers(0, 256, (len(labels), 28, 28, 3), dtype=np.uint8)
        arrays[f"{split}_labels"] = labels
    np.savez(root / "pathmnist.npz", **arrays)
    return root


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory) -> Path:
    """Random weights are explicitly restricted to test fixtures."""
    torch.manual_seed(42)
    path = tmp_path_factory.mktemp("model") / "fixture.pt"
    save_checkpoint(build_model(), path, model_version="test-fixture", config={"fixture": True})
    return path
