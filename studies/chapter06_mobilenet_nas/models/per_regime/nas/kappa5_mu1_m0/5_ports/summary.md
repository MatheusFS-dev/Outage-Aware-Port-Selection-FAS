# Rank-1 Per-Regime NAS Model — kappa5_mu1_m0 — 5 Ports

## Selection

- Regime: kappa5_mu1_m0
- Observed ports: 5
- Trial: 55
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_5_add
- Head depth: 2
- Head width: 192
- Learning rate: 0.0010062387746642004
- Strict-INT8 validation outage probability: 0.981695
- Strict-INT8 test outage probability: 0.982075
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_55.keras | Keras v3 archive, FP32 weights | 811734 bytes | 3ecae0b3b34b63698de04e04ee2d7a6fa6b78e5f340258b590f1727528f87a3a |
| top_1_trial_55.tflite | TensorFlow Lite, strict INT8 | 362920 bytes | 75387c694e1986d4586fcc8b7f95289faf5c9ef14a02957053ba2cc5b38b65e9 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m0_5_ports/optuna_study/.
