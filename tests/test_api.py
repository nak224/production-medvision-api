import io
import pickle

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


def image_bytes(format="PNG", mode="RGB", size=(28, 28)):
    output = io.BytesIO()
    Image.new(mode, size).save(output, format=format)
    return output.getvalue()


@pytest.fixture
def client(checkpoint):
    with TestClient(create_app(checkpoint)) as client:
        yield client


def test_missing_model_reports_unavailable(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "model_not_ready",
            "model_loaded": False,
            "model_version": None,
        }
        assert (
            client.post("/predict", files={"file": ("image.png", image_bytes())}).status_code == 503
        )


def test_health_and_openapi(client):
    assert client.get("/health").json() == {
        "status": "ready",
        "model_loaded": True,
        "model_version": "test-fixture",
    }
    assert client.get("/health").status_code == 200
    assert {"/health", "/predict"} <= client.get("/openapi.json").json()["paths"].keys()


@pytest.mark.parametrize("format,mode", [("PNG", "RGB"), ("JPEG", "RGB"), ("PNG", "L")])
def test_prediction_upload(client, format, mode):
    response = client.post("/predict", files={"file": ("image", image_bytes(format, mode))})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["class_id"] <= 8
    assert 0 <= body["confidence"] <= 1
    assert len(body["probabilities"]) == 9
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert body["model_version"] == "test-fixture"


@pytest.mark.parametrize("payload", [b"", b"not an image", image_bytes()[:40]])
def test_invalid_upload(client, payload):
    response = client.post("/predict", files={"file": ("fake.png", payload, "image/png")})
    assert response.status_code == 422


def test_unsupported_image_format(client):
    response = client.post("/predict", files={"file": ("image.gif", image_bytes("GIF"))})
    assert response.status_code == 415


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", 20)
    assert client.post("/predict", files={"file": ("image", b"x" * 21)}).status_code == 413


def test_pixel_limit(client, monkeypatch):
    monkeypatch.setattr("api.main.MAX_IMAGE_PIXELS", 100)
    response = client.post("/predict", files={"file": ("large.png", image_bytes(size=(11, 10)))})
    assert response.status_code == 413


def test_missing_upload(client):
    assert client.post("/predict").status_code == 422


def test_corrupt_checkpoint_fails_startup(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(pickle.UnpicklingError), TestClient(create_app(path)):
        pass
