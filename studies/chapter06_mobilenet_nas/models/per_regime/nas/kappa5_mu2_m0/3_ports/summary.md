# Rank-1 Per-Regime NAS Model — kappa5_mu2_m0 — 3 Ports

## Selection

- Regime: kappa5_mu2_m0
- Observed ports: 3
- Trial: 94
- Database state: COMPLETE
- MobileNetV3 alpha: 0.2
- Truncation endpoint: v3_block_1_project_gn
- Head depth: 3
- Head width: 112
- Learning rate: 0.00017164099025284143
- Strict-INT8 validation outage probability: 0.9825
- Strict-INT8 test outage probability: 0.98241
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_94.keras | Keras v3 archive, FP32 weights | 272771 bytes | 4939629d948e20e97f305d6029aae1b7c2aa8a7b01a2c781c624467b0595545c |
| top_1_trial_94.tflite | TensorFlow Lite, strict INT8 | 131144 bytes | cd091600db7c83aab8a7e324bd9bdaa87aae9ce01c5c5c9c0133dd30fdfcd634 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m0_3_ports/optuna_study/.
