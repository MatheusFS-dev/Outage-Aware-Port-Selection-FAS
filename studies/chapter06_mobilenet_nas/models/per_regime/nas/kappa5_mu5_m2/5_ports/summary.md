# Rank-1 Per-Regime NAS Model — kappa5_mu5_m2 — 5 Ports

## Selection

- Regime: kappa5_mu5_m2
- Observed ports: 5
- Trial: 60
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 32
- Learning rate: 0.0008509880446401059
- Strict-INT8 validation outage probability: 0.10689
- Strict-INT8 test outage probability: 0.105255
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_60.keras | Keras v3 archive, FP32 weights | 297112 bytes | 8cbf0207ba049516e24747e1a2df763d3d544fe87143aede0d557c87bafb84cf |
| top_1_trial_60.tflite | TensorFlow Lite, strict INT8 | 177408 bytes | 03b3dbd2f47bc1c4c4d6c5e9e7ef70d0b670d6faed880797bcb523f07b799bdd |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m2_5_ports/optuna_study/.
