import json
import os
import sys

import mlflow
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from medvision.config import TrainingConfig
from medvision.data import load_dataset
from medvision.evaluate import evaluate_model
from medvision.evaluate import main as evaluate_main
from medvision.model import load_checkpoint
from medvision.train import _configure_cuda_determinism, train


@pytest.mark.parametrize("existing", [None, ":16:8", ":4096:8"])
def test_cuda_determinism_configuration(monkeypatch, existing):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    if existing is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    _configure_cuda_determinism()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == (existing or ":4096:8")
    assert not torch.backends.cudnn.benchmark


def test_cuda_configuration_after_initialization_requires_restart(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="Restart the notebook"):
        _configure_cuda_determinism()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


@pytest.mark.parametrize("device", [None, torch.device("cpu")], ids=["inferred", "explicit"])
def test_metrics_for_known_predictions_and_missing_classes(device):
    images = torch.eye(9) * 10
    labels = torch.arange(9).reshape(-1, 1)
    metrics = evaluate_model(
        nn.Identity(), DataLoader(TensorDataset(images, labels), batch_size=4), device=device
    )
    assert metrics["samples"] == 9
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == 1
    assert metrics["auroc_ovr_macro"] == 1
    subset = evaluate_model(
        nn.Identity(), DataLoader(TensorDataset(images[:2], labels[:2])), device=device
    )
    assert subset["auroc_ovr_macro"] is None
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(
            nn.Identity(), DataLoader(TensorDataset(images[:0], labels[:0])), device=device
        )


def test_evaluation_with_buffers_but_no_parameters():
    class BufferedIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.0))

        def forward(self, images):
            return images * self.scale

    loader = DataLoader(TensorDataset(torch.eye(9) * 10, torch.arange(9)), batch_size=4)
    metrics = evaluate_model(BufferedIdentity(), loader)
    assert metrics["samples"] == 9
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == 1
    assert metrics["auroc_ovr_macro"] == 1


def test_training_tracking_export_and_reload(synthetic_data, tmp_path, monkeypatch):
    config = TrainingConfig(
        epochs=1, batch_size=9, data_dir=synthetic_data, output_dir=tmp_path / "artifacts"
    )
    checkpoint = train(config)
    model, metadata = load_checkpoint(checkpoint)
    # Validate reload on the training device; CPU and CUDA results need not be bitwise equal.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    report = json.loads((config.output_dir / "validation.json").read_text())
    assert report["epoch"] == 1
    assert report["split"] == "val"
    assert report["model_version"] == metadata["model_version"]
    metrics = evaluate_model(model, DataLoader(load_dataset(synthetic_data, "val"), batch_size=9))
    assert metrics == report["metrics"]
    assert metrics["samples"] == 9
    run = mlflow.get_run(metadata["model_version"])
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["val_accuracy"] == metrics["accuracy"]
    assert (config.output_dir / "mlflow.db").is_file()
    model = model.cpu()

    # Re-running the same configuration must reuse tracking storage and reproduce weights/metrics.
    second_checkpoint = train(config)
    second_model, second_metadata = load_checkpoint(second_checkpoint)
    assert second_metadata["model_version"] != metadata["model_version"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0, atol=0)

    output = tmp_path / "reports" / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medvision-evaluate",
            "--checkpoint",
            str(second_checkpoint),
            "--data-dir",
            str(synthetic_data),
            "--split",
            "test",
            "--output",
            str(output),
        ],
    )
    evaluate_main()
    evaluation = json.loads(output.read_text())
    assert evaluation["split"] == "test"
    assert evaluation["metrics"]["samples"] == 9
    assert evaluation["model_version"] == second_metadata["model_version"]
