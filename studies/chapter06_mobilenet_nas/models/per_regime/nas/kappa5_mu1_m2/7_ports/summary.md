# Rank-1 Per-Regime NAS Model — kappa5_mu1_m2 — 7 Ports

## Selection

- Regime: kappa5_mu1_m2
- Observed ports: 7
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.158285
- Strict-INT8 test outage probability: 0.159265
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018849 bytes | 0f190d6c4a3c41e004134372721487aef24318a6d5f5859acd1b7589268e9718 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | 69ed1de54ffb6f35a665df23e575a34011189e237a055cb55d6887b183f3d6c3 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m2_7_ports/optuna_study/.
