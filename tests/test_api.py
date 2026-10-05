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
    assert {"/health", "/model-info", "/predict", "/predict/batch"} <= client.get(
        "/openapi.json"
    ).json()["paths"].keys()


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


def test_model_info_uses_loaded_metadata(client):
    metadata = client.app.state.predictor.metadata
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        key: metadata[key]
        for key in ("architecture", "model_version", "dataset", "class_names", "preprocessing")
    }
    assert "config" not in response.json()
    assert "state_dict" not in response.json()
    metadata["model_version"] = "another-version"
    metadata["input_size"] = [28, 28]
    metadata["expected_input_format"] = "RGB"
    assert client.get("/model-info").json()["model_version"] == "another-version"
    assert client.get("/model-info").json()["input_size"] == [28, 28]
    assert client.get("/model-info").json()["expected_input_format"] == "RGB"


def batch_files(*payloads):
    return [("files", (name, payload)) for name, payload in payloads]


def test_batch_matches_single_predictions(client):
    uploads = [("first.png", image_bytes()), ("second.jpg", image_bytes("JPEG", "L"))]
    response = client.post("/predict/batch", files=batch_files(*uploads))
    assert response.status_code == 200
    assert len(response.json()) == 2
    for result, (name, payload) in zip(response.json(), uploads, strict=True):
        single = client.post("/predict", files={"file": (name, payload)}).json()
        assert result == {"filename": name, **single}


@pytest.mark.parametrize(
    "payload,status", [(b"invalid", 422), (b"", 422), (image_bytes("GIF"), 415)]
)
def test_batch_mixed_invalid_input(client, payload, status):
    response = client.post(
        "/predict/batch",
        files=batch_files(("valid.png", image_bytes()), ("bad.png", payload)),
    )
    assert response.status_code == status
    assert response.json()["detail"]["index"] == 1
    assert response.json()["detail"]["filename"] == "bad.png"
    assert "error" in response.json()["detail"]


def test_batch_oversized_file(client, monkeypatch):
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", 20)
    response = client.post("/predict/batch", files=batch_files(("large.png", b"x" * 21)))
    assert response.status_code == 413
    assert response.json()["detail"]["filename"] == "large.png"


def test_batch_total_size_limit(client, monkeypatch):
    payload = image_bytes()
    monkeypatch.setattr("api.main.MAX_BATCH_BYTES", len(payload) * 2 - 1)
    response = client.post(
        "/predict/batch", files=batch_files(("one.png", payload), ("two.png", payload))
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Batch exceeds 20 MiB"


def test_batch_file_count_limit(client):
    response = client.post(
        "/predict/batch", files=batch_files(*[("image.png", image_bytes())] * 17)
    )
    assert response.status_code == 413


def test_batch_pixel_limit(client, monkeypatch):
    monkeypatch.setattr("api.main.MAX_IMAGE_PIXELS", 100)
    response = client.post(
        "/predict/batch", files=batch_files(("large.png", image_bytes(size=(11, 10))))
    )
    assert response.status_code == 413


def test_batch_missing_files(client):
    assert client.post("/predict/batch").status_code == 422


def test_new_endpoints_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        assert client.get("/model-info").status_code == 503
        assert (
            client.post(
                "/predict/batch", files=batch_files(("image.png", image_bytes()))
            ).status_code
            == 503
        )


@pytest.mark.parametrize("case", ["valid", "invalid", "oversized", "count", "missing", "error"])
def test_batch_closes_every_file(client, monkeypatch, case):
    from starlette.datastructures import UploadFile

    closed = []
    original_close = UploadFile.close

    async def track_close(upload):
        await original_close(upload)
        closed.append(upload)

    monkeypatch.setattr(UploadFile, "close", track_close)
    uploads = [("one.png", image_bytes()), ("two.png", image_bytes())]
    expected_status = 200
    if case == "invalid":
        uploads[0] = ("bad.png", b"bad")
        expected_status = 422
    elif case == "oversized":
        monkeypatch.setattr("api.main.MAX_BATCH_BYTES", 1)
        expected_status = 413
    elif case == "count":
        monkeypatch.setattr("api.main.MAX_BATCH_FILES", 1)
        expected_status = 413
    elif case == "missing":
        client.app.state.predictor = None
        expected_status = 503
    elif case == "error":

        def fail(image):
            raise RuntimeError("inference failed")

        monkeypatch.setattr(client.app.state.predictor, "predict", fail)
    if case == "error":
        with pytest.raises(RuntimeError, match="inference failed"):
            client.post("/predict/batch", files=batch_files(*uploads))
    else:
        assert (
            client.post("/predict/batch", files=batch_files(*uploads)).status_code
            == expected_status
        )
    assert {upload.filename for upload in closed} == {name for name, _ in uploads}
    assert all(upload.file.closed for upload in closed)
