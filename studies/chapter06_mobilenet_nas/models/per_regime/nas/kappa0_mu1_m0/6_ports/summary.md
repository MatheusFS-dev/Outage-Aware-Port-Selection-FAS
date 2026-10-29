# Rank-1 Per-Regime NAS Model — kappa0_mu1_m0 — 6 Ports

## Selection

- Regime: kappa0_mu1_m0
- Observed ports: 6
- Trial: 45
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0006057862988914949
- Strict-INT8 validation outage probability: 0.212705
- Strict-INT8 test outage probability: 0.21273
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_45.keras | Keras v3 archive, FP32 weights | 746848 bytes | a9565cd58e519d47bff0acfe6e79f024a407a6b3bd7ab0b15244bf41aa311db4 |
| top_1_trial_45.tflite | TensorFlow Lite, strict INT8 | 367512 bytes | b4db2407a83fdfa868ab42bb35164808000cdd81db2a26137895d5d537408d67 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m0_6_ports/optuna_study/.
