# Rank-1 Per-Regime NAS Model — kappa5_mu5_m50 — 3 Ports

## Selection

- Regime: kappa5_mu5_m50
- Observed ports: 3
- Trial: 85
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_5_add
- Head depth: 1
- Head width: 144
- Learning rate: 0.001945048157348829
- Strict-INT8 validation outage probability: 0.10347
- Strict-INT8 test outage probability: 0.103585
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_85.keras | Keras v3 archive, FP32 weights | 905100 bytes | ccaa335fa8199cac1948d78d906bc60fdafe08dc74dec8d84c7a683562c7399e |
| top_1_trial_85.tflite | TensorFlow Lite, strict INT8 | 396696 bytes | fc7ad3c665eff2df377f3b92e8b159cb6f9c5b69166eb16812f83e589b0afa2c |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m50_3_ports/optuna_study/.
