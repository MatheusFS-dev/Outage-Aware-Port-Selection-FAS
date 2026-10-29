# Rank-1 Per-Regime NAS Model — kappa5_mu1_m50 — 6 Ports

## Selection

- Regime: kappa5_mu1_m50
- Observed ports: 6
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.084045
- Strict-INT8 test outage probability: 0.08372
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018837 bytes | 00a50397850a3c8a23919916e9fab267e5fca4dbebea617ccc7c8a4761790dc4 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 452224 bytes | 3e24f72dda937975b75d372f3fe46285e98ffe5619490d24b8572b5ab0a74960 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m50_6_ports/optuna_study/.
