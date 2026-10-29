# Rank-1 Per-Regime NAS Model — kappa5_mu2_m2 — 7 Ports

## Selection

- Regime: kappa5_mu2_m2
- Observed ports: 7
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.124815
- Strict-INT8 test outage probability: 0.125275
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018850 bytes | 52e6cafb8f3221664805351220224a00e1aae9d40b22ef3e443875e75e8caf00 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | c486423acb50f123269ee8eff8eb469a02ca1f8b5200015df8811f66a9d46414 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m2_7_ports/optuna_study/.
