# Rank-1 Per-Regime NAS Model — kappa5_mu5_m0 — 5 Ports

## Selection

- Regime: kappa5_mu5_m0
- Observed ports: 5
- Trial: 26
- Database state: COMPLETE
- MobileNetV3 alpha: 0.3
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 256
- Learning rate: 0.0020787845288244137
- Strict-INT8 validation outage probability: 0.975675
- Strict-INT8 test outage probability: 0.975855
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_26.keras | Keras v3 archive, FP32 weights | 690460 bytes | 9cc5620663f7b2b87ee52e07acb4fdcfd6f566478688dc4f7041350652b4125d |
| top_1_trial_26.tflite | TensorFlow Lite, strict INT8 | 357200 bytes | 71c527f051c4411e5d84975caba253d458145b9716a62fb94e029ea2fa2fda12 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m0_5_ports/optuna_study/.
