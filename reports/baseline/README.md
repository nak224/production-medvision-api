# PathMNIST baseline v1

`metrics.json` is the evaluation output supplied by the project author from Kaggle
for model run **8ffb86dbbb4b47ecb42784b50521a47e**. Its values are preserved without
rounding or reconstructing any counts. The presentation in the root README rounds
scores for readability.

On 2026-10-06, the author supplied a fresh full-test evaluation of
`artifacts/model.pt` with `uv run --frozen medvision-evaluate --checkpoint artifacts/model.pt`.
That output identifies run `8ffb86dbbb4b47ecb42784b50521a47e` and replaces the
previous report association with `68dbe3df440645848a5fe277be677909`. All configuration,
class names, metrics and confusion counts are identical to the previously supplied
report; only the reported run ID changed. This update follows the supplied evaluation,
not an edit to checkpoint metadata or a claim that the two checkpoints are identical.
The previous report remains in Git history. The trained weights are not available
in this checkout for independent verification.

This is the complete official 7,180-image test split (`requested_limit: null`). The
recorded configuration uses five training epochs, seed 42, batch size 128 and Adam
learning rate 0.001. Selection uses validation macro-F1. The report does not specify
the selected epoch, hardware, or number of training workers; these are not inferred.

The confusion matrix sums to 7,180, with 5,777 correct predictions. Its accuracy and
macro-F1 match the reported values. AUROC and cross-entropy loss require the original
prediction scores and are retained as reported. The figure uses these actual counts:

```bash
uv run --frozen python -m medvision.reporting \
  --report reports/baseline/metrics.json \
  --output assets/baseline-confusion-matrix.png
```

The matching weights remain on the training machine. Before another run overwrites
the working export, run `uv run --frozen python -m medvision.baseline` there and copy
the resulting `artifacts/baselines/pathmnist-resnet18-v1/` directory to persistent
storage. The archive command checks run identity and configuration, produces file
hashes, and refuses to overwrite an existing snapshot. This is a local preservation
step, not a model release or an upload to an external host.

Keep this directory unchanged for future comparisons; give new baseline reports
separate directories and use validation data to choose improvements.
