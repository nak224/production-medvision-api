import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from medvision.baseline import freeze_baseline
from medvision.reporting import load_report, plot_confusion_matrix

BASELINE_REPORT = Path(__file__).resolve().parents[1] / "reports/baseline/metrics.json"


def test_published_report_and_plot(tmp_path):
    report, counts = load_report(BASELINE_REPORT)
    assert int(counts.sum()) == 7180
    assert int(counts.trace()) == 5777
    assert counts[7, 7] == 79
    assert int(counts[7].sum()) == 421
    assert report["model_version"] == "68dbe3df440645848a5fe277be677909"
    image = plot_confusion_matrix(BASELINE_REPORT, tmp_path / "matrix.png")
    with Image.open(image) as png:
        assert png.format == "PNG"
        assert png.width >= 1000 and png.height >= 1000


@pytest.mark.parametrize(
    "error", ["missing_matrix", "negative", "total", "accuracy", "macro_f1", "order"]
)
def test_invalid_report_is_not_plotted(tmp_path, error):
    report = json.loads(BASELINE_REPORT.read_text())
    metrics = report["metrics"]
    if error == "missing_matrix":
        del metrics["confusion_matrix"]
    elif error == "negative":
        metrics["confusion_matrix"][0][0] = -1
    elif error == "total":
        metrics["samples"] += 1
    elif error == "order":
        report["class_names"].reverse()
    else:
        metrics[error] = 0.0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(report))
    output = tmp_path / "matrix.png"
    with pytest.raises(ValueError):
        plot_confusion_matrix(path, output)
    assert not output.exists()


@pytest.fixture
def fixture_report(tmp_path):
    # Run identity/configuration are for the test checkpoint, never a published artifact.
    report = json.loads(BASELINE_REPORT.read_text())
    report["model_version"] = "test-fixture"
    report["training_config"] = {"fixture": True}
    path = tmp_path / "test-fixture.json"
    path.write_text(json.dumps(report))
    return path


def test_freeze_preserves_bytes_and_refuses_overwrite(checkpoint, fixture_report, tmp_path):
    output = tmp_path / "baseline"
    assert freeze_baseline(checkpoint, fixture_report, output) == output
    manifest = json.loads((output / "manifest.json").read_text())
    with checkpoint.open("rb") as stream:
        expected = hashlib.file_digest(stream, "sha256").hexdigest()
    assert manifest["sha256"]["model.pt"] == expected
    for name, expected in manifest["sha256"].items():
        with (output / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    assert json.loads((output / "metrics.json").read_text()) == json.loads(
        fixture_report.read_text()
    )
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        freeze_baseline(checkpoint, fixture_report, output)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("field", ["model_version", "training_config", "requested_limit"])
def test_freeze_rejects_wrong_model_or_subset(checkpoint, fixture_report, tmp_path, field):
    report = json.loads(fixture_report.read_text())
    report[field] = {"model_version": "wrong-run", "training_config": {}, "requested_limit": 10}[
        field
    ]
    fixture_report.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        freeze_baseline(checkpoint, fixture_report, output)
    assert not output.exists()
    assert checkpoint.is_file()
