# Rank-1 Per-Regime NAS Model — kappa0_mu1_m0 — 7 Ports

## Selection

- Regime: kappa0_mu1_m0
- Observed ports: 7
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.21226
- Strict-INT8 test outage probability: 0.212305
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018850 bytes | 645081e0a7984ec26e8d36469a1ad73a3a8e2b075acbbe6c8398185164b5313a |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | 5514574a15bf8995102ec5fc0cda3d0c283b77821f242ad16a64981458a9fab2 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m0_7_ports/optuna_study/.
