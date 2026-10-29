# Chapter 5: Low-Power CNN-Based Port Selection

Original WCNPS 2025 CNN notebooks and selected artifacts, plus the later spatial-correlation diagnostic.

See the [root README](../../README.md) for the dissertation abstract, environment setup, licenses and chapter map. The [source manifest](../../docs/source-manifest.json) records exact original paths and checksums.

## Contents

- `scripts/`: study entry points and analysis code.
- `results/`: retained numerical evidence, configurations and applicable final artifacts.
- `figures/`: standalone dissertation figures and their available sources.

## Historical CNN study

The original training protocol is preserved with the CNN study artifacts. It is separate from the later 60/20/20 MobileNet studies. The seven selected CNNs and Optuna databases are under `results/models/cnn/`.

Create a separate environment using `scripts/legacy/REQUIREMENTS.txt` and consult the historical `tf-optuna.yaml` for recorded package versions. These are historical records, not a newly verified environment specification. Open the CNN notebooks with the working directory set to this chapter directory. Their helper import and dataset/result paths have been adapted. Notebook execution outputs have been removed, while numerical result artifacts are retained separately.

The spatial diagnostic is `scripts/spatial_pretest.py`. Its retained correlation matrices are derived summaries, not the channel dataset. Original CNN training and dataset-dependent outage evaluation have not been rerun as part of this extraction.

## Historical external utilities

The archived training helper imports `araras.utils.io`, `araras.ml.optuna.utils`, and `tensoroo.callbacks.nan_loss_pruner`. The statistics script also imports `araras.ml.model.stats`. These external packages are not bundled with the study artifacts, and `tf-optuna.yaml` does not pin them. [Araras](https://github.com/MatheusFS-dev/araras) is maintained separately. A complete historical training rerun requires compatible versions of these utilities in addition to the recorded environment. Their source is not silently replaced in this extraction. The selected CNN models and saved numerical results remain available independently.
