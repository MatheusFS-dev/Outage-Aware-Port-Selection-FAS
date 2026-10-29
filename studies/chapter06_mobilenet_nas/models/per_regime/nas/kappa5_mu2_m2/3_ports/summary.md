# Rank-1 Per-Regime NAS Model — kappa5_mu2_m2 — 3 Ports

## Selection

- Regime: kappa5_mu2_m2
- Observed ports: 3
- Trial: 57
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 208
- Learning rate: 0.0012488806354118492
- Strict-INT8 validation outage probability: 0.199205
- Strict-INT8 test outage probability: 0.200655
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_57.keras | Keras v3 archive, FP32 weights | 346531 bytes | 6ad05d9d1dc831ccf2856271a579dd1794f097f936ab97e4a256fcab7d9f851a |
| top_1_trial_57.tflite | TensorFlow Lite, strict INT8 | 185936 bytes | 0550dab599a192e97b76455bc3f63340cc851cdb239b4be237120b54c5a15591 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m2_3_ports/optuna_study/.
