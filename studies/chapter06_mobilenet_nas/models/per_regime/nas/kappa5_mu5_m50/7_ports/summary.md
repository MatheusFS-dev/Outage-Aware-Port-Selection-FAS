# Rank-1 Per-Regime NAS Model — kappa5_mu5_m50 — 7 Ports

## Selection

- Regime: kappa5_mu5_m50
- Observed ports: 7
- Trial: 74
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 144
- Learning rate: 0.000615374140486711
- Strict-INT8 validation outage probability: 0.01135
- Strict-INT8 test outage probability: 0.01114
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_74.keras | Keras v3 archive, FP32 weights | 352115 bytes | 31bd9757b1c474786b73bffa99f5a469d5772b61ef51d629d5be7869b010f08b |
| top_1_trial_74.tflite | TensorFlow Lite, strict INT8 | 194968 bytes | 084a82ac0b293ea3b4a2a61c8b9435d891c5e8f259a747d57f8d6f28bced3d7e |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m50_7_ports/optuna_study/.
