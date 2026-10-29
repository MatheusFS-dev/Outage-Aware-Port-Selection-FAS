# Rank-1 Per-Regime NAS Model — kappa5_mu5_m0 — 3 Ports

## Selection

- Regime: kappa5_mu5_m0
- Observed ports: 3
- Trial: 77
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 176
- Learning rate: 0.00867516929946585
- Strict-INT8 validation outage probability: 0.97587
- Strict-INT8 test outage probability: 0.97604
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_77.keras | Keras v3 archive, FP32 weights | 378305 bytes | 23309c3bf4de13f57ab206a6d7d93a61b88efcb09fa3d0ec660406c61559f07d |
| top_1_trial_77.tflite | TensorFlow Lite, strict INT8 | 199344 bytes | f4eb3485eed56c7f285cda8a0f5cbfdd69dbe1b3e1264789552fb5bdb2ffea49 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m0_3_ports/optuna_study/.
