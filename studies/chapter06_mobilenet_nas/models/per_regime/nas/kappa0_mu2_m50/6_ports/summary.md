# Rank-1 Per-Regime NAS Model — kappa0_mu2_m50 — 6 Ports

## Selection

- Regime: kappa0_mu2_m50
- Observed ports: 6
- Trial: 70
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 48
- Learning rate: 0.0007776020859226853
- Strict-INT8 validation outage probability: 0.12421
- Strict-INT8 test outage probability: 0.12479
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_70.keras | Keras v3 archive, FP32 weights | 1182530 bytes | ce019afc71534d43895cf77b4b334fb68546e19716222638716a257e284cf507 |
| top_1_trial_70.tflite | TensorFlow Lite, strict INT8 | 533384 bytes | 63ca545ac18cf4604876c4b59c1be9fa4d2d2863603775f5db34acb21bccd56b |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu2_m50_6_ports/optuna_study/.
