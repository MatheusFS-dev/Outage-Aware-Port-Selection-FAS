# Rank-1 Per-Regime NAS Model — kappa0_mu5_m50 — 6 Ports

## Selection

- Regime: kappa0_mu5_m50
- Observed ports: 6
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.04493
- Strict-INT8 test outage probability: 0.04543
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018837 bytes | 0018b0ada6eca9a6c1899f7fd0f391b7f54bc3720b1a1721e3382f6b8a0abaa4 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 452224 bytes | ae225c16e5d65f52d031eaf50e5ab943e87064a6100c4ab16c7a6518da5569ab |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu5_m50_6_ports/optuna_study/.
