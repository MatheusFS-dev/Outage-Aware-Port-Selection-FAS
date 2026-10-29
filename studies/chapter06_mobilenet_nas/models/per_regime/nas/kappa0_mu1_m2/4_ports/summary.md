# Rank-1 Per-Regime NAS Model — kappa0_mu1_m2 — 4 Ports

## Selection

- Regime: kappa0_mu1_m2
- Observed ports: 4
- Trial: 82
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_2_add
- Head depth: 1
- Head width: 16
- Learning rate: 0.0003780428128504927
- Strict-INT8 validation outage probability: 0.23309
- Strict-INT8 test outage probability: 0.230785
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_82.keras | Keras v3 archive, FP32 weights | 172392 bytes | a37af889473d0cb1858f21be26ee20f152592bd1d81a99b02cb5c0b760fce3d8 |
| top_1_trial_82.tflite | TensorFlow Lite, strict INT8 | 118328 bytes | 71e9bad58f052d43e5aac7707e256cf038dcaf434eb5658ca4b2aa8af4c7e77b |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m2_4_ports/optuna_study/.
