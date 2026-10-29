# Rank-1 Per-Regime NAS Model — kappa0_mu1_m50 — 6 Ports

## Selection

- Regime: kappa0_mu1_m50
- Observed ports: 6
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.21336
- Strict-INT8 test outage probability: 0.212105
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018837 bytes | de87a3d6fc2ce767593f38279d91a3282e8bf4b4eff9138b82fd4d74b9902720 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 452224 bytes | aa7d4dc5ebb7a4c3258eba9fad9b61df1568b0a37ebde9788581f9f46ecdc483 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m50_6_ports/optuna_study/.
