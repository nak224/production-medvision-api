from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from medvision.data import CLASS_NAMES, DATASET, PREPROCESSING

ARCHITECTURE = "resnet18-small-stem-v1"


def build_model() -> nn.Module:
    # Train from scratch; retain spatial detail for 28 x 28 images.
    model = resnet18(weights=None, num_classes=len(CLASS_NAMES))
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def save_checkpoint(model: nn.Module, path: Path, *, model_version: str, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": ARCHITECTURE,
            "dataset": DATASET,
            "class_names": list(CLASS_NAMES),
            "preprocessing": PREPROCESSING,
            "model_version": model_version,
            "config": config,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "dataset": DATASET,
        "class_names": list(CLASS_NAMES),
        "preprocessing": PREPROCESSING,
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Checkpoint metadata does not match the supported PathMNIST model")
    if not isinstance(checkpoint.get("model_version"), str) or not checkpoint["model_version"]:
        raise ValueError("Checkpoint must contain a model version")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, {key: value for key, value in checkpoint.items() if key != "state_dict"}
