# Rank-1 Per-Regime NAS Model — kappa0_mu1_m2 — 5 Ports

## Selection

- Regime: kappa0_mu1_m2
- Observed ports: 5
- Trial: 99
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0010150183760312171
- Strict-INT8 validation outage probability: 0.218005
- Strict-INT8 test outage probability: 0.215975
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_99.keras | Keras v3 archive, FP32 weights | 746835 bytes | d09ba8b81fcb56987489afe9f7adbc5e89467a6621a4b4564b6c36484815d751 |
| top_1_trial_99.tflite | TensorFlow Lite, strict INT8 | 366384 bytes | 21caeaaee60d32d7395c94505ccdc69d8924a1c95ea0780bfbaf4f3f2ee85513 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m2_5_ports/optuna_study/.
