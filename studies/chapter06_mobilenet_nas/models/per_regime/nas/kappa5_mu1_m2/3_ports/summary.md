# Rank-1 Per-Regime NAS Model — kappa5_mu1_m2 — 3 Ports

## Selection

- Regime: kappa5_mu1_m2
- Observed ports: 3
- Trial: 12
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_9_add
- Head depth: 2
- Head width: 32
- Learning rate: 0.000570175680978563
- Strict-INT8 validation outage probability: 0.224205
- Strict-INT8 test outage probability: 0.225025
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_12.keras | Keras v3 archive, FP32 weights | 1374693 bytes | 4c3723fd2104626fe056ea97f121cb780203e3efee1f9ed638c0b1fb1aada413 |
| top_1_trial_12.tflite | TensorFlow Lite, strict INT8 | 598760 bytes | 5655a5d364220fbc9b426ce024b2c6b69086db340a9ab5f5f2a24eb765484362 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m2_3_ports/optuna_study/.
