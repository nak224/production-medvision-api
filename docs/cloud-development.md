# Cloud development

In the prepared cloud environment, use the existing checkout at
`/workspace/production-medvision-api`. The environment is isolated, so this setup
does not require an additional Git worktree.

```bash
cd /workspace/production-medvision-api
export UV_CACHE_DIR=/workspace/.cache/uv
make setup-cpu check test
```

All training and evaluation paths are relative to the repository root. No secrets
are required. The package manager needs PyPI and the PyTorch CPU wheel hosts.
The dataset downloader additionally needs `zenodo.org`; an HTTP 403 from the
environment proxy means this network destination must be enabled in environment
settings. Saving a configuration draft alone does not activate runtime access.

After network access is active:

```bash
make download
make smoke
MEDVISION_CHECKPOINT=artifacts/smoke/model.pt make serve
```

Use a separate terminal to inspect `/health` with a local HTTP request and send a
real PNG/JPEG patch to `/predict`. Without a checkpoint, `make serve` starts for
API development but `/health` and `/predict` return 503. This indicates that the
model is not ready; inference requires a trained checkpoint. Synthetic checkpoints
are reserved for tests.

The virtual environment, datasets, reports and model files persist on disk in
the prepared environment snapshot. Uvicorn and optional MLflow UI processes must
be started again in new tasks. Refresh dependencies using `uv sync --frozen`.
Dependency installation and training are separate steps.

Offline tests generate synthetic archives and checkpoints exclusively in pytest
temporary directories. MedMNIST may warn that it cannot initialize `~/.medmnist`
on a read-only home directory; the application explicitly uses its writable
`data_dir`, so that default directory is not required.

## Docker behind the cloud proxy

This cloud machine's Docker build network does not resolve the injected proxy
hostname automatically. The following helper passes the existing proxy binding,
its current DNS resolution, and the injected CA bundle using a temporary BuildKit
mount. TLS verification remains enabled; no proxy credentials or certificates are
stored in the repository or image.

```bash
python scripts/build_cloud_image.py
```

The helper tags `medvision-api:dev` and stores Docker client state in a writable
directory under `/workspace/.cache/`. On a regular development machine use the
standard `docker build -t medvision-api .` command from the README.

## CUDA and notebook setup

Use `make setup-gpu` for the standard PyTorch dependencies. Training and evaluation
select CUDA when available and otherwise fall back to CPU. Check GPU availability
before a full run; CPU training can take considerably longer. Exported checkpoints
contain CPU tensors and remain loadable by the CPU API.

Training keeps deterministic algorithms enabled. Make exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python, preserving an existing
override such as `:16:8`. Direct calls to `train()` set the default before CUDA
initialization, and training disables cuDNN benchmarking for repeatable runs.
If CUDA is already initialized in a notebook without this variable, restart the
kernel and set it before using CUDA. CPU and GPU results need not match bit for bit;
the checkpoint round-trip test compares metrics on the training device.

In Kaggle, enable Internet access and a GPU accelerator, and clone `main` as shown
in the README. Check that CUDA is available before full training. For two GPUs,
select **GPU T4 ×2** and confirm both are visible with `make check-2gpu` (prefix
shell commands with `!` in a notebook). Download the data once before launching
workers. Kaggle storage is temporary: download checkpoints, reports and MLflow
artifacts or copy them to persistent storage before ending the session.

PNG report generation uses Matplotlib's non-interactive Agg renderer even when a
notebook exports an unavailable inline backend. No notebook plotting dependency
is required; the caller's `MPLBACKEND` environment setting is restored afterward.

## Two-GPU DDP details

`make smoke-2gpu` and `make train-2gpu` require two visible CUDA devices. They launch
one process per GPU with `torchrun` and PyTorch DistributedDataParallel (NCCL).
Ordinary `make train` and `make smoke` use one GPU, or CPU when CUDA is unavailable.
Evaluation and API inference continue to use one device.

`batch_size` is the **global batch size** and must divide evenly by the number of
workers: the default 128 means 64 images per GPU. Each worker trains on a different
shuffled shard; gradients are averaged across workers. For equal shard lengths
without duplicate samples, at most one training example is dropped per epoch with
two workers. The official training split is even, so the full two-GPU baseline
drops none. The shuffle changes each epoch.

Worker 0 evaluates the complete validation split once, writes one MLflow run and
exports the usual CPU-loadable checkpoint. Other workers wait during validation.
Batch normalization uses local per-GPU batches; the exported running statistics
belong to worker 0. Results can differ from a single-GPU run even with the same seed
and global batch size. MLflow records worker count, batch size per worker, processed
training sample count and global average loss.

Two GPUs do not guarantee twice the speed: this small model on 28 × 28 images may
spend much of its time loading data and synchronizing gradients. Compare full-epoch
timings before increasing batch size. Each GPU keeps its own model copy; memory is
not pooled. The distributed flow is tested with two CPU/Gloo workers; CUDA/NCCL
execution and speedups must be validated on the target hardware.

## Training artifacts and baseline preservation

Single-GPU and DDP commands write to the same paths. Avoid running independent jobs
against the same output directory simultaneously. Relative paths in configurations
are relative to the working directory.

| Artifact | Full baseline | Smoke run |
| --- | --- | --- |
| Best checkpoint | `artifacts/model.pt` | `artifacts/smoke/model.pt` |
| Validation report | `artifacts/validation.json` | `artifacts/smoke/validation.json` |
| MLflow database | `artifacts/mlflow.db` | `artifacts/smoke/mlflow.db` |
| Evaluation report | `reports/metrics.json` (test) | `reports/smoke-metrics.json` (validation) |

The smoke run trains for one epoch on 256 training examples and evaluates 128
validation examples. These deterministic subsets verify the workflow; they are
not a full benchmark. The full configuration uses five epochs on the complete
training split and selects the best checkpoint by validation macro-F1. Only the
separate evaluation command evaluates the test split.

Re-running training with the same output directory adds an MLflow run and replaces
the exported checkpoint/report with the new run's best model; previous run artifacts
remain in MLflow. Inspect full runs with:

```bash
uv run --frozen mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1
```

For the smoke experiment, use `sqlite:///artifacts/smoke/mlflow.db` instead.

Preserve the published baseline before further experiments:

```bash
uv run --frozen python -m medvision.baseline
```

This archives the matching checkpoint, published report, confusion-matrix image and
SHA-256 manifest under `artifacts/baselines/pathmnist-resnet18-v1/`. It checks model
version and training configuration and refuses to overwrite an existing archive.
If a later run has replaced `artifacts/model.pt`, pass the matching checkpoint with
`--checkpoint PATH`. See the [evaluation and reproduction notes](../reports/baseline/README.md)
for its model version. Store the archive persistently; weights are not in Git.
For inference, set `MEDVISION_CHECKPOINT=artifacts/baselines/pathmnist-resnet18-v1/model.pt`.

Use a different `output_dir` and evaluation report path for subsequent experiments.
Do not overwrite the published report or use repeated test-set feedback to select
model changes; compare candidates on validation data first.

Regenerate the published figure without training or a GPU:

```bash
uv run --frozen python -m medvision.reporting \
  --report reports/baseline/metrics.json \
  --output assets/baseline-confusion-matrix.png
```
