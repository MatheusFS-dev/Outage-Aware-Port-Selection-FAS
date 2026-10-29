# Rank-1 Per-Regime NAS Model — kappa5_mu5_m50 — 4 Ports

## Selection

- Regime: kappa5_mu5_m50
- Observed ports: 4
- Trial: 21
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0003663623552395579
- Strict-INT8 validation outage probability: 0.02041
- Strict-INT8 test outage probability: 0.019915
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_21.keras | Keras v3 archive, FP32 weights | 1018814 bytes | 2a2208b124cf1efd8e4e14b356c3e295bafe94d19593cbde3e5f48eb678399d2 |
| top_1_trial_21.tflite | TensorFlow Lite, strict INT8 | 449968 bytes | 299c6145e3a0d34bf7f8df8a0c91e7ad4a72c32ded45eda86032d262840c4676 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m50_4_ports/optuna_study/.
