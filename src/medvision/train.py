import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from medvision.config import TrainingConfig, load_config
from medvision.data import load_dataset
from medvision.distributed import TrainingRuntime, training_runtime
from medvision.evaluate import evaluate_model
from medvision.model import build_model, save_checkpoint


def _configure_cuda_determinism() -> None:
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA. "
                "Restart the notebook kernel/process, set the variable, and retry training."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Avoid selecting different convolution algorithms across repeated training runs.
    torch.backends.cudnn.benchmark = False


def train(config: TrainingConfig) -> Path:
    _configure_cuda_determinism()
    with training_runtime() as runtime:
        return _train(config, runtime)


def _train(config: TrainingConfig, runtime: TrainingRuntime) -> Path:
    if config.batch_size % runtime.world_size:
        raise ValueError("batch_size is global and must be divisible by the number of workers")
    local_batch_size = config.batch_size // runtime.world_size
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    device = runtime.device
    print(f"Worker {runtime.rank}/{runtime.world_size}: device={device}, batch={local_batch_size}")
    train_data = load_dataset(config.data_dir, "train", config.train_limit, config.seed)
    if len(train_data) < runtime.world_size:
        raise ValueError("Training data must contain at least one example per worker")
    sampler = (
        DistributedSampler(
            train_data,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            seed=config.seed,
            drop_last=True,
        )
        if runtime.distributed
        else None
    )
    train_loader = DataLoader(
        train_data,
        batch_size=local_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = None
    if runtime.primary:
        val_data = load_dataset(config.data_dir, "val", config.val_limit, config.seed)
        val_loader = DataLoader(
            val_data, batch_size=config.batch_size, num_workers=config.num_workers
        )
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_module = build_model().to(device)
    model = (
        DistributedDataParallel(
            model_module, device_ids=[device.index] if device.type == "cuda" else None
        )
        if runtime.distributed
        else model_module
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_path = output / "model.pt"
    best_score = -1.0
    tracking = nullcontext()
    if runtime.primary:
        # One writer owns SQLite, metrics, and the plain (non-DDP) checkpoint.
        mlflow.set_tracking_uri(f"sqlite:///{output / 'mlflow.db'}")
        experiment = mlflow.get_experiment_by_name("pathmnist")
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(
                "pathmnist", artifact_location=(output / "mlruns").as_uri()
            )
        )
        tracking = mlflow.start_run(experiment_id=experiment_id)
    with tracking as run:
        config_dict = config.model_dump(mode="json")
        if runtime.primary:
            mlflow.log_params(
                {
                    **config_dict,
                    "world_size": runtime.world_size,
                    "batch_size_per_worker": local_batch_size,
                }
            )
            mlflow.set_tags({"dataset": "pathmnist", "purpose": "research-and-portfolio"})
        for epoch in range(config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(images), targets.reshape(-1).long())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
                sample_count += len(images)
            totals = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
            if runtime.distributed:
                dist.reduce(totals, dst=0)
            if runtime.primary:
                # Evaluate every validation example once, without DDP forward collectives.
                metrics = evaluate_model(model_module, val_loader)
                scalar_metrics = {
                    f"val_{key}": value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
                scalar_metrics["train_loss"] = (totals[0] / totals[1]).item()
                scalar_metrics["train_samples"] = int(totals[1].item())
                mlflow.log_metrics(scalar_metrics, step=epoch + 1)
                print(json.dumps({"epoch": epoch + 1, **scalar_metrics}), flush=True)
                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    save_checkpoint(
                        model_module,
                        checkpoint_path,
                        model_version=run.info.run_id,
                        config=config_dict,
                    )
                    report = {
                        "epoch": epoch + 1,
                        "split": "val",
                        "model_version": run.info.run_id,
                        "config": config_dict,
                        "world_size": runtime.world_size,
                        "metrics": metrics,
                    }
                    (output / "validation.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False) + "\n"
                    )
            runtime.barrier()
        if runtime.primary:
            mlflow.log_artifact(str(output / "validation.json"))
            mlflow.log_artifact(str(checkpoint_path))
    runtime.barrier()
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet-18 PathMNIST baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    print(train(load_config(args.config)))


if __name__ == "__main__":
    main()
