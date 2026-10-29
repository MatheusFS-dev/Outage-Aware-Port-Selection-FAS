# Rank-1 Per-Regime NAS Model — kappa5_mu5_m50 — 5 Ports

## Selection

- Regime: kappa5_mu5_m50
- Observed ports: 5
- Trial: 33
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 144
- Learning rate: 0.00018394461078800357
- Strict-INT8 validation outage probability: 0.01598
- Strict-INT8 test outage probability: 0.01573
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_33.keras | Keras v3 archive, FP32 weights | 1037898 bytes | f5b6ff45b68ea08b5e5a5501252b68fdd8c18aa2e45c2d219b559ae9697aed8d |
| top_1_trial_33.tflite | TensorFlow Lite, strict INT8 | 456728 bytes | 5b1cc434d0031bdcec64a935ecf07661b401b43e70803040578994e10a4fece6 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m50_5_ports/optuna_study/.
