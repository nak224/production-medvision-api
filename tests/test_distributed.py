import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
import torch
from PIL import Image

from medvision.config import TrainingConfig
from medvision.distributed import TrainingRuntime
from medvision.inference import Predictor
from medvision.train import _train


def test_global_batch_must_split_evenly():
    runtime = TrainingRuntime(torch.device("cpu"), world_size=2)
    with pytest.raises(ValueError, match="divisible"):
        _train(TrainingConfig(batch_size=3), runtime)


def test_two_worker_training(synthetic_data, tmp_path):
    output = tmp_path / "distributed"
    config = TrainingConfig(
        epochs=2,
        batch_size=6,
        num_threads=1,
        train_limit=17,
        data_dir=synthetic_data,
        output_dir=output,
    )
    config_path = tmp_path / "config.yaml"
    # JSON is also valid YAML.
    config_path.write_text(config.model_dump_json())
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(Path(__file__).with_name("ddp_worker.py")),
        str(config_path),
    ]
    with subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            logs, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            logs, _ = process.communicate()
            pytest.fail(f"Distributed training timed out:\n{logs}")
        assert process.returncode == 0, logs

    workers = [json.loads((output / f"worker-{rank}.json").read_text()) for rank in range(2)]
    assert workers[0]["weights"] == workers[1]["weights"]
    for worker in workers:
        assert len(worker["seen"]) == 16  # Eight examples per worker, over two epochs.
        assert worker["seen"][:8] != worker["seen"][8:]
        assert worker["checkpoint"] == str(output / "model.pt")
    for epoch in range(2):
        shards = [worker["seen"][epoch * 8 : (epoch + 1) * 8] for worker in workers]
        assert len(set(shards[0]) | set(shards[1])) == 16
        assert set(shards[0]).isdisjoint(shards[1])

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{output / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("pathmnist")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.params["world_size"] == "2"
    assert runs[0].data.params["batch_size_per_worker"] == "3"
    assert runs[0].data.metrics["train_samples"] == 16
    assert runs[0].data.metrics["val_samples"] == 9
    history = client.get_metric_history(runs[0].info.run_id, "train_loss")
    assert len(history) == 2
    for metric in history:
        offset = (metric.step - 1) * 3  # Three local batches per epoch, including the tail.
        batches = [batch for worker in workers for batch in worker["losses"][offset : offset + 3]]
        expected_loss = sum(loss for loss, _ in batches) / sum(count for _, count in batches)
        assert metric.value == pytest.approx(expected_loss)
    report = json.loads((output / "validation.json").read_text())
    assert report["world_size"] == 2
    assert report["metrics"]["samples"] == 9
    predictor = Predictor(output / "model.pt")
    prediction = predictor.predict(Image.new("RGB", (28, 28)))
    assert prediction["model_version"] == runs[0].info.run_id
    assert sum(prediction["probabilities"].values()) == pytest.approx(1)
