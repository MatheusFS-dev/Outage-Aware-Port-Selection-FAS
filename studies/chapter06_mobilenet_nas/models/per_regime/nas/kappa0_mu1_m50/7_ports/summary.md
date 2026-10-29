# Rank-1 Per-Regime NAS Model — kappa0_mu1_m50 — 7 Ports

## Selection

- Regime: kappa0_mu1_m50
- Observed ports: 7
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.21365
- Strict-INT8 test outage probability: 0.212635
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018849 bytes | 8a96d6903b1ffc9e66e2f1f1382be7fdb754989ca885fb979b497ba969c6649a |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | 02bdce0118dd00ebcdf7571d173eea27d8720ef57ea9c79ea09be42bbf12c956 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m50_7_ports/optuna_study/.
