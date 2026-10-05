import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from api.artifacts import checkpoint_path
from api.main import create_app


@pytest.fixture
def s3(monkeypatch):
    # No credentials, metadata service, or AWS network requests in these tests.
    factory = MagicMock(return_value=MagicMock(spec=["download_file", "close"]))
    monkeypatch.setattr("api.artifacts.boto3.client", factory)
    return factory, factory.return_value


def test_s3_startup_loads_checkpoint_and_removes_download(checkpoint, monkeypatch, s3):
    factory, client = s3
    downloaded = []

    def download(bucket, key, destination):
        assert (bucket, key) == ("test-bucket", "models/model.pt")
        shutil.copyfile(checkpoint, destination)
        downloaded.append(Path(destination))

    client.download_file.side_effect = download
    monkeypatch.setenv("MEDVISION_MODEL_S3_URI", "s3://test-bucket/models/model.pt")
    monkeypatch.setenv("MEDVISION_CHECKPOINT", "/missing/local.pt")
    app = create_app()
    with TestClient(app) as http:
        assert http.get("/health").status_code == 200
        assert http.get("/model-info").json()["model_version"] == "test-fixture"
        assert not downloaded[0].parent.exists()
    assert app.state.predictor is None
    factory.assert_called_once_with("s3")
    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "https://bucket/model.pt",
        "s3://bucket",
        "s3:///model.pt",
        "s3://bucket/",
        "s3://user@bucket/model.pt",
        "s3://bucket:443/model.pt",
        "s3://bucket/model.pt?x=1",
        "s3://bucket/model.pt#fragment",
        "s3://bad bucket/model.pt",
    ],
)
def test_invalid_s3_uri_fails_startup(uri, monkeypatch, s3):
    monkeypatch.setenv("MEDVISION_MODEL_S3_URI", uri)
    with pytest.raises(ValueError, match="MEDVISION_MODEL_S3_URI"), TestClient(create_app()):
        pass
    s3[0].assert_not_called()


def test_failed_download_cleans_partial_file_and_does_not_fall_back(checkpoint, monkeypatch, s3):
    downloaded = []

    def fail(bucket, key, destination):
        path = Path(destination)
        path.write_bytes(b"partial download")
        downloaded.append(path)
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")

    s3[1].download_file.side_effect = fail
    monkeypatch.setenv("MEDVISION_MODEL_S3_URI", "s3://test-bucket/model.pt")
    monkeypatch.setenv("MEDVISION_CHECKPOINT", str(checkpoint))
    with pytest.raises(ClientError), TestClient(create_app()):
        pass
    assert not downloaded[0].parent.exists()


def test_s3_download_cleanup_when_loader_fails(s3, tmp_path):
    s3[1].download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(b"bad")
    with pytest.raises(ValueError, match="load failed"):
        with checkpoint_path(tmp_path / "local.pt", "s3://bucket/model.pt") as path:
            assert path.is_file()
            raise ValueError("load failed")
    assert not path.parent.exists()


def test_local_environment_fallback(checkpoint, monkeypatch, s3):
    monkeypatch.delenv("MEDVISION_MODEL_S3_URI", raising=False)
    monkeypatch.setenv("MEDVISION_CHECKPOINT", str(checkpoint))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
    assert checkpoint.is_file()
    s3[0].assert_not_called()


def test_explicit_checkpoint_overrides_environment(checkpoint, monkeypatch, s3):
    monkeypatch.setenv("MEDVISION_MODEL_S3_URI", "s3://bucket/model.pt")
    with TestClient(create_app(checkpoint)) as client:
        assert client.get("/health").status_code == 200
    s3[0].assert_not_called()
