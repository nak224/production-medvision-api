# Model card: PathMNIST ResNet-18 baseline

## Status and intended use

Research and portfolio demonstration only. Not intended for clinical diagnosis,
treatment decisions, or medical use. The first measured baseline is recorded in
the [Kaggle test report](../reports/baseline/metrics.json); trained weights are not
bundled in Git. Synthetic test fixtures must never be presented as trained
PathMNIST models or used to report model quality.

## Model

PyTorch / torchvision ResNet-18, initialized from scratch with nine output classes.
The stem uses a 3 × 3 convolution with stride 1 and no max-pooling. Training uses
Adam and cross-entropy; defaults and subset settings live in `configs/`.

Preprocessing is RGB conversion, bilinear resizing to 28 × 28 and normalization
with channel-wise mean/std 0.5. Checkpoints contain architecture, preprocessing
version, ordered class names, training configuration and the MLflow run ID.
Only checkpoints produced by a trusted source should be loaded.

## Dataset

[PathMNIST in MedMNIST v2](https://zenodo.org/records/10519652) is derived from
NCT-CRC-HE-100K / CRC-VAL-HE-7K colorectal histology patches. The dataset is listed
under CC BY 4.0 in MedMNIST's metadata. Preserve original-source attribution when
sharing derived artifacts. This repository downloads from the provider and does
not distribute the images.

The nine labels are adipose, background, debris, lymphocytes, mucus, smooth muscle,
normal colon mucosa, cancer-associated stroma, and colorectal adenocarcinoma epithelium.
Official train/validation/test splits are preserved. The test split comes from a
different clinical center. Small smoke-run subsets use a deterministic random selection.

## Evaluation

The best validation macro-F1 selects the export. The separate evaluation command
can evaluate validation or test data and records split, subset limit, seed, model
version, training configuration, sample count, loss, accuracy, macro-F1, macro
one-vs-rest AUROC and the ordered confusion matrix. AUROC is undefined (`null`) if
any class is absent.

The first baseline is model run `8ffb86dbbb4b47ecb42784b50521a47e`: five configured
epochs, seed 42, batch size 128 and learning rate 0.001, with no train/validation
subset limits. Its full test evaluation on 7,180 images reports accuracy 0.804596,
macro-F1 0.740466, macro one-vs-rest AUROC 0.960659 and cross-entropy loss 1.459959.
These are one run's results, not confidence intervals or a state-of-the-art claim.
The supplied report does not record GPU count or identify the selected checkpoint's
epoch within the five-epoch run.

Cancer-associated stroma has the lowest recall: 79/421 (18.8%). Its largest
confusions are adenocarcinoma epithelium (130), debris (106), and smooth muscle (83).
Smooth muscle and normal colon mucosa also have relatively low recall (58.8% and
58.3%). See the [confusion matrix](../assets/baseline-confusion-matrix.png) and
[class-level table](../README.md#error-analysis).

Accuracy and macro-F1 were checked against the supplied confusion counts. AUROC and
loss are retained from the author's evaluation output; prediction scores and the
trained checkpoint are not available in this checkout for independent recomputation.
Keep this report and its matching checkpoint as the baseline before model changes.

## Limitations

Low-resolution research patches differ substantially from clinical workflows.
Resizing arbitrary uploaded images does not establish that they are appropriate
inputs. The classifier has no out-of-distribution detection, calibration study,
fairness assessment or clinical validation. Softmax confidence is not a medical
probability. Full-slide processing, diagnostic recommendations and patient data
handling are outside this demonstration's scope.
