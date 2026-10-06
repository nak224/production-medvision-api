"""Subprocess harness for the real torchrun integration test, using synthetic data only."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from medmnist import INFO
from torch.utils.data import Dataset

import medvision.train as training
from medvision.config import load_config


def main():
    INFO["pathmnist"]["n_samples"] = {"train": 18, "val": 9, "test": 9}
    config = load_config(Path(sys.argv[1]))
    seen = []
    models = []
    losses = []
    original_load = training.load_dataset
    original_build = training.build_model
    original_loss = training.nn.functional.cross_entropy

    class TrackingDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            seen.append(index)
            return self.dataset[index]

    def load_dataset(root, split, limit, seed):
        dataset = original_load(root, split, limit, seed)
        return TrackingDataset(dataset) if split == "train" else dataset

    def build_model():
        model = original_build()
        models.append(model)
        return model

    def cross_entropy(images, targets, *args, **kwargs):
        loss = original_loss(images, targets, *args, **kwargs)
        if torch.is_grad_enabled():
            losses.append([loss.item() * len(targets), len(targets)])
        return loss

    training.load_dataset = load_dataset
    training.build_model = build_model
    training.nn.functional.cross_entropy = cross_entropy
    checkpoint = training.train(config)
    digest = hashlib.sha256()
    for parameter in models[0].parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    (config.output_dir / f"worker-{os.environ['RANK']}.json").write_text(
        json.dumps(
            {
                "seen": seen,
                "weights": digest.hexdigest(),
                "checkpoint": str(checkpoint),
                "losses": losses,
            }
        )
    )


if __name__ == "__main__":
    main()
