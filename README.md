# Production MedVision API

A reproducible PathMNIST classification pipeline: **PyTorch → MLflow → FastAPI**,
with offline tests, a Docker image definition and GitHub Actions.

For research and portfolio demonstration only. Not intended for clinical diagnosis,
treatment decisions, or medical use.

## Current milestone

The implementation covers data download and preprocessing, CPU/GPU training with
automatic device detection, evaluation, model export and an inference API. No trained
benchmark model or measured PathMNIST performance is bundled yet. Tests use synthetic
fixtures and do **not** establish medical accuracy. The API returns HTTP 503 until a
checkpoint is available.

See [the model card](docs/model-card.md) for intended use and limitations.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/) (tested with 0.12.19),
and optionally Docker. Run all commands from the repository root.

### CPU setup

```bash
make setup-cpu
make check
make test
```

### GPU setup

On a CUDA-capable environment such as Kaggle, install the standard PyTorch dependencies:

```bash
make setup-gpu
```

Verify that PyTorch can see the GPU before starting a full training run:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Training and evaluation automatically use CUDA when `torch.cuda.is_available()` is true
and otherwise fall back to CPU. Checkpoints are exported with CPU tensors, so artifacts
trained on a GPU remain loadable by the CPU inference service.

Training keeps deterministic algorithms enabled. Make exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python, preserving an existing
override such as `:16:8`. Direct calls to `train()` set the default before CUDA
initialization, and training disables cuDNN benchmarking for repeatable runs.
If CUDA is already initialized in a notebook without this variable, restart the
kernel and set it before using CUDA. CPU and GPU results need not match bit for bit;
the checkpoint round-trip test compares metrics on the training device.

The offline test suite includes a tiny synthetic training round trip. For cloud-specific
setup, see [cloud development](docs/cloud-development.md).

### Kaggle GPU training

Create a Kaggle Notebook, enable a GPU accelerator and Internet access, then run:

```bash
!git clone -b feat/medvision-baseline https://github.com/nak224/production-medvision-api.git
%cd production-medvision-api
%env CUBLAS_WORKSPACE_CONFIG=:4096:8
!pip install -q uv
!make setup-gpu
!uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!make check test
!make download
!make smoke
```

The device check should print `CUDA: True` before running the full baseline. If the smoke
run succeeds, start the complete training and test-set evaluation with:

```bash
!make train
!make evaluate
```

Kaggle notebook storage is temporary. Download any checkpoint, report or MLflow artifact
you want to keep before the session ends, or copy it to persistent storage.

### Using both Kaggle T4 GPUs

Select **GPU T4 ×2** in Kaggle, then verify both devices are visible and run:

```python
!uv run --frozen python -c "import torch; print('GPUs:', torch.cuda.device_count())"
!make download
!make smoke-2gpu
!make train-2gpu
!make evaluate
```

The two-GPU targets require two visible CUDA devices. They launch one process per GPU
with `torchrun` and PyTorch DistributedDataParallel (NCCL). The ordinary `make train`
and `make smoke` commands still use a single GPU, or CPU when CUDA is unavailable.
Download the data once before launching workers.

`batch_size` is the **global batch size** and must divide evenly by the number of workers:
the default 128 means 64 images per GPU. Each worker trains on a different shuffled shard;
gradients are averaged across workers. For equal shard lengths without duplicate samples,
at most one training example is dropped per epoch with two workers. The official training
split is even, so the full two-GPU baseline drops none. The shuffle changes each epoch.

Worker 0 evaluates the complete validation split once, writes one MLflow run and exports
the usual CPU-loadable checkpoint. Other workers wait during validation. Batch normalization
uses local per-GPU batches; the exported running statistics belong to worker 0. Results can
differ from a single-GPU run even with the same seed and global batch size. MLflow records
worker count, batch size per worker, processed training sample count and global average loss.

Artifacts use the same paths as the single-GPU commands: `artifacts/smoke/model.pt` for
the smoke run and `artifacts/model.pt` for full training. Avoid running independent jobs
against the same output directory simultaneously. Evaluation and API inference continue
to use one device.

Two GPUs do not guarantee twice the speed: this small model on 28 × 28 images may spend
much of its time loading data and synchronizing gradients. Compare full-epoch timings
before increasing batch size. Each T4 keeps its own model copy; their memory is not pooled.
The distributed flow is tested with two CPU/Gloo workers; T4/NCCL execution and speedups
must be validated in Kaggle.

### Download and run a small real-data experiment

```bash
make download
make smoke
MEDVISION_CHECKPOINT=artifacts/smoke/model.pt make serve
```

`make download` fetches the official 28 × 28 PathMNIST archive from `zenodo.org`,
checks the published MD5 over the TLS-verified download, and reuses a valid local
archive. Dataset files remain in the ignored `data/` directory.

`make smoke` trains for one epoch on 256 training examples and evaluates 128 validation
examples. These deterministic subsets verify the workflow; they are **not** a full
benchmark. The checkpoint, SQLite tracking database and validation report are written
to `artifacts/smoke/`; the evaluation report is `reports/smoke-metrics.json`.

### Train and evaluate the baseline

```bash
make train
make evaluate
make serve
```

`configs/base.yaml` defines five epochs on the complete official training split. The
best validation macro-F1 selects `artifacts/model.pt`. The test split is only evaluated
by the separate evaluation command, which writes `reports/metrics.json`. Training uses
CUDA automatically when available; CPU training can take considerably longer. Relative
paths in configuration files are relative to the working directory.

Each training run records parameters, train loss, validation metrics and artifacts
in local MLflow. Re-running with the same output directory adds a tracking run and
replaces the exported checkpoint/report with the new run's best model; previous
run artifacts remain in MLflow. To inspect baseline runs:

```bash
uv run --frozen mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1
```

For the smoke experiment use `sqlite:///artifacts/smoke/mlflow.db` instead.

### API requests

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -X POST -F 'file=@example.png' http://127.0.0.1:8000/predict
```

Provide your own public research patch as `example.png`. The service accepts PNG
or JPEG (including grayscale converted to RGB), up to 5 MiB and 4 million pixels.
It resizes to 28 × 28 and returns `class_id`, `label`, `confidence`, all nine
`probabilities`, and `model_version`. Confidence is an uncalibrated softmax score.
OpenAPI is exposed at `/openapi.json`, with interactive documentation at `/docs`.

`/health` returns 200 when a model is loaded, or 503 with `model_not_ready` when
the checkpoint is missing. Invalid uploads return 422, unsupported image formats
415, and images exceeding the limits 413. A corrupt or incompatible checkpoint
fails startup. Set `MEDVISION_CHECKPOINT` before starting the service; restart it
after replacing weights. There is no hot reload of model artifacts.

### Docker

Train a checkpoint first, then build and mount the artifact read-only:

```bash
docker build -t medvision-api .
chmod a+rx artifacts
chmod a+r artifacts/model.pt
docker run --rm -p 8000:8000 \
  --mount type=bind,src="$(pwd)/artifacts",dst=/models,readonly \
  medvision-api
```

For a smoke checkpoint, apply the permission commands to `artifacts/smoke` and
its `model.pt`, then mount `$(pwd)/artifacts/smoke` instead. These permissions let
the container's non-root user read the exported model. The container expects
`/models/model.pt`. Data and weights are excluded from the image. A missing
checkpoint leaves the container unhealthy (503).

## Data and architecture

- Dataset: [PathMNIST / MedMNIST v2](https://zenodo.org/records/10519652),
  28 × 28 RGB histology patches, nine tissue classes.
- Official splits: 89,996 training, 10,004 validation, 7,180 test images; the test
  data comes from a different clinical center.
- Dataset license: **CC BY 4.0**, as declared by MedMNIST 3.0.2. Original data
  and model licenses are separate from any future code license for this repository.
- Model: torchvision ResNet-18 trained from scratch; a 3 × 3 stride-1 stem and
  no initial max-pooling preserve detail in small images. No pretrained weights
  are downloaded.
- Preprocessing: RGB, bilinear resize, channel normalization with mean/std 0.5,
  shared between training, evaluation and inference.
- Metrics: accuracy, macro-F1 across all nine classes, macro one-vs-rest AUROC,
  cross-entropy loss, and a confusion matrix. AUROC is `null` when a subset lacks
  one or more classes.
- Reproducibility: locked/versioned dependencies, a versioned configuration, fixed
  seeds and untouched official splits. Results can differ across CPU/GPU hardware
  and library/runtime environments.

## Layout

```text
src/medvision/   data, configuration, model, training, evaluation, inference
api/             FastAPI application and response schemas
configs/         full baseline and small smoke-run settings
tests/           offline data, training, export, inference and API tests
docs/            model card and cloud development notes
.github/         CI: lint, formatting, tests and Docker build
```

Generated reports, datasets, weights and MLflow databases are ignored by Git.
GitHub Actions will run after these files are pushed; a local test pass is not a
hosted CI result. A public demo, full benchmark results, calibration, latency
measurements, a release and a code-license decision remain future work.
