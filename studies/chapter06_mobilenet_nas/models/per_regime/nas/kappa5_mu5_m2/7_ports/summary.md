# Rank-1 Per-Regime NAS Model — kappa5_mu5_m2 — 7 Ports

## Selection

- Regime: kappa5_mu5_m2
- Observed ports: 7
- Trial: 62
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.00018815938594389132
- Strict-INT8 validation outage probability: 0.100215
- Strict-INT8 test outage probability: 0.09846
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_62.keras | Keras v3 archive, FP32 weights | 1018850 bytes | a6e8e94e30264fdfe908b667658829bc4c43e3154cfdae76302ffebd5a12fdad |
| top_1_trial_62.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | 6e37910c3c073da0664045f2623a3aed795fd1ea93ae5c6f9e7ee62803bdbad5 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m2_7_ports/optuna_study/.
