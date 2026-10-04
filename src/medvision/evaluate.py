import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from medvision.data import CLASS_NAMES, load_dataset
from medvision.model import load_checkpoint


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict:

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        targets = targets.reshape(-1).long()
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        labels.append(targets.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset")
    y_true, scores = np.concatenate(labels), np.concatenate(probabilities)
    y_pred = scores.argmax(axis=1)
    classes = list(range(len(CLASS_NAMES)))
    # Multiclass AUROC is undefined if the selected subset lacks any class.
    auroc = None
    if len(np.unique(y_true)) == len(CLASS_NAMES):
        auroc = float(
            roc_auc_score(y_true, scores, labels=classes, multi_class="ovr", average="macro")
        )
    return {
        "samples": len(y_true),
        "loss": loss_sum / len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "auroc_ovr_macro": auroc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on an official PathMNIST split"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, metadata = load_checkpoint(args.checkpoint)
    model = model.to(device)
    dataset = load_dataset(args.data_dir, args.split, args.limit, args.seed)
    report = {
        "dataset": "pathmnist",
        "split": args.split,
        "requested_limit": args.limit,
        "subset_seed": args.seed,
        "model_version": metadata["model_version"],
        "training_config": metadata["config"],
        "class_names": list(CLASS_NAMES),
        "metrics": evaluate_model(model, DataLoader(dataset, batch_size=128), device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
