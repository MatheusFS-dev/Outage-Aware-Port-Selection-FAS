# Rank-1 Per-Regime NAS Model — kappa5_mu5_m2 — 6 Ports

## Selection

- Regime: kappa5_mu5_m2
- Observed ports: 6
- Trial: 31
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0001333935256563805
- Strict-INT8 validation outage probability: 0.1021
- Strict-INT8 test outage probability: 0.1006
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_31.keras | Keras v3 archive, FP32 weights | 1614488 bytes | 109be8f3117fa0b42d6fd977057b6504b441f6b60a2f24e42289e3d3850e0072 |
| top_1_trial_31.tflite | TensorFlow Lite, strict INT8 | 665400 bytes | b0660c8e8eb46908ff74603368f80db747893461b8789c7c1df0e09f7f4c5093 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m2_6_ports/optuna_study/.
