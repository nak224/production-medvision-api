"""Render evaluation reports without rerunning inference or reconstructing missing counts."""

import argparse
import json
import os
from pathlib import Path
from textwrap import fill

import numpy as np

from medvision.data import CLASS_NAMES


def load_report(path: Path) -> tuple[dict, np.ndarray]:
    report = json.loads(path.read_text())
    if report.get("dataset") != "pathmnist" or report.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Report must identify PathMNIST and its ordered class names")
    if report.get("split") not in {"test", "val"} or not report.get("model_version"):
        raise ValueError("Report must identify the evaluation split and model version")
    metrics = report["metrics"]
    counts = np.asarray(metrics.get("confusion_matrix"))
    if counts.shape != (9, 9) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("confusion_matrix must contain 9 x 9 integer counts")
    samples = metrics["samples"]
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if np.any(counts < 0) or sum(map(int, counts.flat)) != samples:
        raise ValueError("Confusion matrix must be nonnegative and sum to samples")
    accuracy = float(np.trace(counts) / samples)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    f1 = np.divide(2 * counts.diagonal(), denominator, out=np.zeros(9), where=denominator > 0)
    if not np.isclose(accuracy, metrics["accuracy"], rtol=0, atol=1e-10):
        raise ValueError("Reported accuracy does not match the confusion matrix")
    if not np.isclose(float(f1.mean()), metrics["macro_f1"], rtol=0, atol=1e-10):
        raise ValueError("Reported macro-F1 does not match the confusion matrix")
    return report, counts


def plot_confusion_matrix(report_path: Path, output: Path) -> Path:
    if output.suffix.lower() != ".png":
        raise ValueError("Confusion-matrix output must be a PNG file")
    report, counts = load_report(report_path)
    # Kaggle can export an inline backend unavailable in this project's venv.
    # Select Agg before Matplotlib's first import, then restore the caller's setting.
    backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    finally:
        if backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = backend

    support = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(100 * counts, support, out=np.zeros((9, 9)), where=support > 0)
    # Use Figure directly: no interactive display or notebook backend is required.
    figure = Figure(figsize=(13, 11), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    names = [fill(name, width=20) for name in report["class_names"]]
    axis.set_xticks(range(9), names, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    axis.set_yticks(range(9), names, fontsize=10)
    axis.set_xlabel("Predicted tissue class", fontsize=12)
    axis.set_ylabel("True tissue class", fontsize=12)
    axis.set_title(
        f"PathMNIST — {report['split']} confusion matrix\n"
        f"{report['metrics']['samples']:,} images · cells show count and percentage of true class",
        fontsize=14,
        pad=20,
    )
    for row in range(9):
        for column in range(9):
            rate = f"{percentages[row, column]:.1f}%" if support[row, 0] else "n/a"
            axis.text(
                column,
                row,
                f"{counts[row, column]}\n{rate}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if percentages[row, column] >= 55 else "#172033",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Percentage within true class")
    figure.supxlabel(f"Model run: {report['model_version']}", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the actual counts from a PathMNIST report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot_confusion_matrix(args.report, args.output))


if __name__ == "__main__":
    main()
