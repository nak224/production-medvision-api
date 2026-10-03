# Cloud development

Use the existing checkout at `/workspace/production-medvision-api`. Each cloud
task already has an isolated environment; do not create a Git worktree unless the
user explicitly requests one.

```bash
cd /workspace/production-medvision-api
export UV_CACHE_DIR=/workspace/.cache/uv
make setup check test
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
API development but `/health` and `/predict` return 503. Do not report this state
as successful model inference or generate random serving weights to mask it.

The virtual environment, datasets, reports and model files persist on disk in
the prepared environment snapshot. Uvicorn and optional MLflow UI processes must
be started again in new tasks. Refresh dependencies using `uv sync --frozen`.
Do not run training automatically during dependency installation.

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
