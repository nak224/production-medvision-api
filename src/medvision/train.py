import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from medvision.config import TrainingConfig, load_config
from medvision.data import load_dataset
from medvision.evaluate import evaluate_model
from medvision.model import build_model, save_checkpoint


def train(config: TrainingConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_data = load_dataset(config.data_dir, "train", config.train_limit, config.seed)
    val_data = load_dataset(config.data_dir, "val", config.val_limit, config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers)
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Local tracking: no account, remote tracking service, or credential required.
    mlflow.set_tracking_uri(f"sqlite:///{output / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name("pathmnist")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment("pathmnist", artifact_location=(output / "mlruns").as_uri())
    )
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_path = output / "model.pt"
    best_score = -1.0
    with mlflow.start_run(experiment_id=experiment_id) as run:
        config_dict = config.model_dump(mode="json")
        mlflow.log_params(config_dict)
        mlflow.set_tags({"dataset": "pathmnist", "purpose": "research-and-portfolio"})
        for epoch in range(config.epochs):
            model.train()
            loss_sum = 0.0
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(images), targets.reshape(-1).long())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
            metrics = evaluate_model(model, val_loader)
            scalar_metrics = {
                f"val_{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            scalar_metrics["train_loss"] = loss_sum / len(train_data)
            mlflow.log_metrics(scalar_metrics, step=epoch + 1)
            print(json.dumps({"epoch": epoch + 1, **scalar_metrics}), flush=True)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                save_checkpoint(
                    model, checkpoint_path, model_version=run.info.run_id, config=config_dict
                )
                report = {
                    "epoch": epoch + 1,
                    "split": "val",
                    "model_version": run.info.run_id,
                    "config": config_dict,
                    "metrics": metrics,
                }
                (output / "validation.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
        mlflow.log_artifact(str(output / "validation.json"))
        mlflow.log_artifact(str(checkpoint_path))
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CPU ResNet-18 PathMNIST baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    print(train(load_config(args.config)))


if __name__ == "__main__":
    main()
