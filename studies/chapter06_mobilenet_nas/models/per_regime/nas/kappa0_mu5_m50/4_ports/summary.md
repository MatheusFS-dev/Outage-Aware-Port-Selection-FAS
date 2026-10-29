# Rank-1 Per-Regime NAS Model — kappa0_mu5_m50 — 4 Ports

## Selection

- Regime: kappa0_mu5_m50
- Observed ports: 4
- Trial: 27
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 2
- Head width: 96
- Learning rate: 0.00010859201073622795
- Strict-INT8 validation outage probability: 0.05863
- Strict-INT8 test outage probability: 0.05907
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_27.keras | Keras v3 archive, FP32 weights | 316922 bytes | 77f8accb092a5d5c72cb2a09a3af892b148f2aec5d23f21606dadc64dce260ee |
| top_1_trial_27.tflite | TensorFlow Lite, strict INT8 | 177504 bytes | ac82ceb90dfa2a48987dfec3e7dfa253fb9da5b19ca548a979f7b3ce7eae181b |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu5_m50_4_ports/optuna_study/.
