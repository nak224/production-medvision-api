UV ?= uv
export CUBLAS_WORKSPACE_CONFIG ?= :4096:8

.PHONY: setup-cpu setup-gpu check test download train smoke evaluate serve
setup-cpu:
	uv sync --extra-index-url https://download.pytorch.org/whl/cpu

setup-gpu:
	uv sync
check:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest -q

download:
	$(UV) run --frozen medvision-download

train:
	$(UV) run --frozen medvision-train --config configs/base.yaml

smoke:
	$(UV) run --frozen medvision-train --config configs/smoke.yaml
	$(UV) run --frozen medvision-evaluate --checkpoint artifacts/smoke/model.pt --split val --limit 128 --output reports/smoke-metrics.json

evaluate:
	$(UV) run --frozen medvision-evaluate --checkpoint artifacts/model.pt

serve:
	$(UV) run --frozen uvicorn api.main:app --host 0.0.0.0 --port 8000
