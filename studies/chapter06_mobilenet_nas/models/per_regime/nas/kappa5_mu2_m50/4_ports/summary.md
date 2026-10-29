# Rank-1 Per-Regime NAS Model — kappa5_mu2_m50 — 4 Ports

## Selection

- Regime: kappa5_mu2_m50
- Observed ports: 4
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.049745
- Strict-INT8 test outage probability: 0.04952
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018814 bytes | b3328972a7e350365519396a80cabe5c5cf5c433648d134a456eb790ffa5a43b |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 449968 bytes | eea8246b8531c601642d2fc8c5783c6eae076b64d82bc32eda6afe13348ea953 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m50_4_ports/optuna_study/.
