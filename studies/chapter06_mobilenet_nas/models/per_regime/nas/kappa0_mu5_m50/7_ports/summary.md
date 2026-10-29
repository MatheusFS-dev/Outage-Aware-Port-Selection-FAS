# Rank-1 Per-Regime NAS Model — kappa0_mu5_m50 — 7 Ports

## Selection

- Regime: kappa0_mu5_m50
- Observed ports: 7
- Trial: 21
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 96
- Learning rate: 0.0003186989284975986
- Strict-INT8 validation outage probability: 0.044265
- Strict-INT8 test outage probability: 0.04489
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_21.keras | Keras v3 archive, FP32 weights | 1009310 bytes | c158f3398255da88aa8c4f8d95f6a57fe2d7f99d97bdab2cce7e90d5158158ad |
| top_1_trial_21.tflite | TensorFlow Lite, strict INT8 | 450536 bytes | db2f01d3de18385b450cb354d5d659e7bde529cc0e4c4b1513e6d5aa11ac64fd |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu5_m50_7_ports/optuna_study/.
