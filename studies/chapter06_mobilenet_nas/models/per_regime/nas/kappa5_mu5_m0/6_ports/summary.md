# Rank-1 Per-Regime NAS Model — kappa5_mu5_m0 — 6 Ports

## Selection

- Regime: kappa5_mu5_m0
- Observed ports: 6
- Trial: 93
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_9_add
- Head depth: 2
- Head width: 96
- Learning rate: 0.0004577745858242478
- Strict-INT8 validation outage probability: 0.975685
- Strict-INT8 test outage probability: 0.975845
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_93.keras | Keras v3 archive, FP32 weights | 1709766 bytes | e367c88ba6470a651bf060e35efe522dc6eee9cecb34375a8d5c47109565d1bd |
| top_1_trial_93.tflite | TensorFlow Lite, strict INT8 | 704072 bytes | 09a58d8c10d720e7ff08b1a0f7ae8f8376262f819c6fd394d73e05bc057161f0 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m0_6_ports/optuna_study/.
