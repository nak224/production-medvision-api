# Model card: PathMNIST ResNet-18 baseline

## Status and intended use

Research and portfolio demonstration only. Not intended for clinical diagnosis,
treatment decisions, or medical use. No trained release or measured benchmark is
included in this milestone. Synthetic test fixtures must never be presented as
trained PathMNIST models or used to report model quality.

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
any class is absent. Full benchmark results remain to be measured.

## Limitations

Low-resolution research patches differ substantially from clinical workflows.
Resizing arbitrary uploaded images does not establish that they are appropriate
inputs. The classifier has no out-of-distribution detection, calibration study,
fairness assessment or clinical validation. Softmax confidence is not a medical
probability. Full-slide processing, diagnostic recommendations and patient data
handling are outside this demonstration's scope.
