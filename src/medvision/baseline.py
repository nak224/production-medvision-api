"""Preserve a matching model and evaluation report before starting another experiment."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from medvision.model import load_checkpoint
from medvision.reporting import load_report, plot_confusion_matrix


def freeze_baseline(checkpoint: Path, report_path: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Baseline already exists; refusing to overwrite {output}")
    report, _ = load_report(report_path)
    if report["split"] != "test" or report.get("requested_limit") is not None:
        raise ValueError("A baseline requires evaluation on the complete test split")
    if report["metrics"]["samples"] != 7180:
        raise ValueError("A full PathMNIST test report must contain 7,180 samples")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")

    output.mkdir(parents=True, exist_ok=False)
    try:
        saved_checkpoint = output / "model.pt"
        shutil.copyfile(checkpoint, saved_checkpoint)
        # Validate the copied bytes, so the archived checkpoint is what we inspect.
        _, metadata = load_checkpoint(saved_checkpoint)
        if metadata["model_version"] != report["model_version"]:
            raise ValueError("Checkpoint model_version does not match the baseline report")
        if metadata["config"] != report["training_config"]:
            raise ValueError("Checkpoint training configuration does not match the baseline report")
        saved_report = output / "metrics.json"
        saved_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        plot_confusion_matrix(saved_report, output / "confusion-matrix.png")
        checksums = {}
        for name in ("model.pt", "metrics.json", "confusion-matrix.png"):
            with (output / name).open("rb") as stream:
                checksums[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {
            "model_version": report["model_version"],
            "training_config": report["training_config"],
            "sha256": checksums,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        # Only remove this invocation's new directory; existing baselines are never touched.
        shutil.rmtree(output)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a checkpoint and matching test report")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline/metrics.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/baselines/pathmnist-resnet18-v1")
    )
    args = parser.parse_args()
    print(freeze_baseline(args.checkpoint, args.report, args.output))


if __name__ == "__main__":
    main()
