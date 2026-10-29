# Rank-1 Per-Regime NAS Model — kappa0_mu1_m2 — 3 Ports

## Selection

- Regime: kappa0_mu1_m2
- Observed ports: 3
- Trial: 52
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_2_add
- Head depth: 1
- Head width: 32
- Learning rate: 0.001941126218779984
- Strict-INT8 validation outage probability: 0.270295
- Strict-INT8 test outage probability: 0.268485
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_52.keras | Keras v3 archive, FP32 weights | 173630 bytes | f246d3bc93543192ab146c15dfb19544b57a25f4903a740318aec568d628d76b |
| top_1_trial_52.tflite | TensorFlow Lite, strict INT8 | 116592 bytes | ae1eaec20338181d42521f1c186f4a3c4d8a0e4b1ad3b2f87d2e1fbc37f96813 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m2_3_ports/optuna_study/.
