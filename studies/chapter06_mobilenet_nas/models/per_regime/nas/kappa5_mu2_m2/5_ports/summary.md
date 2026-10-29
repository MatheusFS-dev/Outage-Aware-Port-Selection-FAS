# Rank-1 Per-Regime NAS Model — kappa5_mu2_m2 — 5 Ports

## Selection

- Regime: kappa5_mu2_m2
- Observed ports: 5
- Trial: 55
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0004122867866335944
- Strict-INT8 validation outage probability: 0.128705
- Strict-INT8 test outage probability: 0.129225
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_55.keras | Keras v3 archive, FP32 weights | 875523 bytes | 3314d8f62bc97ae5406b08c34fe622ec41ff523929a568fd99014b84727f86b8 |
| top_1_trial_55.tflite | TensorFlow Lite, strict INT8 | 431976 bytes | 30e56b4284b0b857d0983dafadf3bcdbedd0e97c67bfe4fc86f5fc4aeaf09268 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m2_5_ports/optuna_study/.
