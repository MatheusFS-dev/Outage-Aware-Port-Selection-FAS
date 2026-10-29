# Rank-1 Per-Regime NAS Model — kappa5_mu2_m2 — 4 Ports

## Selection

- Regime: kappa5_mu2_m2
- Observed ports: 4
- Trial: 50
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 80
- Learning rate: 0.00013547677110594975
- Strict-INT8 validation outage probability: 0.142925
- Strict-INT8 test outage probability: 0.14381
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_50.keras | Keras v3 archive, FP32 weights | 999739 bytes | 13618987db0cb90f66a2e8a915035842f8a7e9a9862fcb43fa2d7d130fda344d |
| top_1_trial_50.tflite | TensorFlow Lite, strict INT8 | 444336 bytes | 0b8ecabe7d29b337f7f84007ea1b844ca5693d5873787a9d5b098263b2c87221 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m2_4_ports/optuna_study/.
