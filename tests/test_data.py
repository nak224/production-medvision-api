import hashlib
import io
from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest
import torch
from PIL import Image

from medvision.data import INFO, HTTPSOnlyRedirect, download_data, load_dataset, preprocess


def test_training_and_inference_preprocessing_match(synthetic_data):
    with np.load(synthetic_data / "pathmnist.npz") as archive:
        original = Image.fromarray(archive["train_images"][0])
    training_image, label = load_dataset(synthetic_data, "train")[0]
    torch.testing.assert_close(training_image, preprocess(original))
    assert training_image.shape == (3, 28, 28)
    assert training_image.min() >= -1 and training_image.max() <= 1
    assert label.item() == 0


def test_rgb_conversion_and_resize():
    tensor = preprocess(Image.new("L", (50, 40), color=255))
    torch.testing.assert_close(tensor, torch.ones(3, 28, 28))


def test_split_selection_and_reproducible_subset(synthetic_data):
    assert len(load_dataset(synthetic_data, "train")) == 18
    assert len(load_dataset(synthetic_data, "val")) == 9
    assert len(load_dataset(synthetic_data, "test")) == 9
    first = load_dataset(synthetic_data, "train", limit=5, seed=42)
    second = load_dataset(synthetic_data, "train", limit=5, seed=42)
    other = load_dataset(synthetic_data, "train", limit=5, seed=43)
    assert first.indices == second.indices
    assert first.indices != other.indices
    assert len(set(first.indices)) == 5


def test_missing_data_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="medvision-download"):
        load_dataset(tmp_path, "train")


def test_download_refuses_to_replace_invalid_existing_archive(tmp_path: Path):
    archive = tmp_path / "pathmnist.npz"
    archive.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_data(tmp_path)
    assert archive.read_bytes() == b"invalid"


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_download_integrity_cleanup_and_reuse(tmp_path, monkeypatch, valid_checksum):
    payload = b"synthetic download fixture"
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(INFO["pathmnist"], "MD5", expected if valid_checksum else "0" * 32)
    calls = []

    class Opener:
        def open(self, url, timeout):
            calls.append(url)
            result = io.BytesIO(payload)
            result.url = url
            return result

    monkeypatch.setattr("medvision.data.build_opener", lambda *_: Opener())
    if valid_checksum:
        path = download_data(tmp_path)
        assert path.read_bytes() == payload
        assert download_data(tmp_path) == path
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="integrity check"):
            download_data(tmp_path)
        assert not (tmp_path / "pathmnist.npz").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_https_downgrade():
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPSOnlyRedirect().redirect_request(
            Request("https://zenodo.org/file"), None, 302, "Found", {}, "http://zenodo.org/file"
        )
