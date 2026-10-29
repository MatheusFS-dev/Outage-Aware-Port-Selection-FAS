# Rank-1 Per-Regime NAS Model — kappa0_mu2_m50 — 5 Ports

## Selection

- Regime: kappa0_mu2_m50
- Observed ports: 5
- Trial: 78
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 80
- Learning rate: 0.00028760209016462596
- Strict-INT8 validation outage probability: 0.127915
- Strict-INT8 test outage probability: 0.128685
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_78.keras | Keras v3 archive, FP32 weights | 302616 bytes | 0a22bfafda430166ef88cbc4a027afb7979d6b08d1454b46a659c6b643594bca |
| top_1_trial_78.tflite | TensorFlow Lite, strict INT8 | 176928 bytes | bb7109c509566cb29c46b87f3aab1a4c0638877b31c56b0dc749fabdf18dfdf1 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu2_m50_5_ports/optuna_study/.
