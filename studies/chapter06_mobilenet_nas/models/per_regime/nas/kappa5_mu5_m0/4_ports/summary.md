# Rank-1 Per-Regime NAS Model — kappa5_mu5_m0 — 4 Ports

## Selection

- Regime: kappa5_mu5_m0
- Observed ports: 4
- Trial: 21
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_4_add
- Head depth: 2
- Head width: 256
- Learning rate: 0.0024113073747707315
- Strict-INT8 validation outage probability: 0.97587
- Strict-INT8 test outage probability: 0.975955
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_21.keras | Keras v3 archive, FP32 weights | 852987 bytes | f9660e6e5a2e0059bede79207858d37e2d84c1ee18853e0a72607df1beb7c498 |
| top_1_trial_21.tflite | TensorFlow Lite, strict INT8 | 354376 bytes | 05e9bccf8d2c8ae94486769515d40ce0e5704d69f728e22d9dde350736f48e7d |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m0_4_ports/optuna_study/.
