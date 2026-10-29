# Rank-1 Per-Regime NAS Model — kappa5_mu5_m2 — 4 Ports

## Selection

- Regime: kappa5_mu5_m2
- Observed ports: 4
- Trial: 59
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_2_add
- Head depth: 1
- Head width: 48
- Learning rate: 0.0002338125467599475
- Strict-INT8 validation outage probability: 0.11611
- Strict-INT8 test outage probability: 0.114625
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_59.keras | Keras v3 archive, FP32 weights | 204841 bytes | 06f4004b3e0549c7346706b532c21c21b295f14f3df73ad41e7d981b93446378 |
| top_1_trial_59.tflite | TensorFlow Lite, strict INT8 | 130288 bytes | a20e5b2356767043605d2a305f36cad0c8f133ef3eea3e5ceb7d99b88332aa61 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m2_4_ports/optuna_study/.
