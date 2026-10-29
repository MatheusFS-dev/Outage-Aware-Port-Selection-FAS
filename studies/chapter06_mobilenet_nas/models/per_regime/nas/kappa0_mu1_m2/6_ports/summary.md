# Rank-1 Per-Regime NAS Model — kappa0_mu1_m2 — 6 Ports

## Selection

- Regime: kappa0_mu1_m2
- Observed ports: 6
- Trial: 23
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 128
- Learning rate: 0.0002448331026963788
- Strict-INT8 validation outage probability: 0.215585
- Strict-INT8 test outage probability: 0.21354
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_23.keras | Keras v3 archive, FP32 weights | 1028373 bytes | 298ee1841f3c43d38146eda4fd8d930e26003473ae3277aff46aa618e967be3a |
| top_1_trial_23.tflite | TensorFlow Lite, strict INT8 | 455040 bytes | 40f0a354c9b2dfa03d184b28b49affccb69dfb5184ab5fe6ea260558ae939eb7 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m2_6_ports/optuna_study/.
