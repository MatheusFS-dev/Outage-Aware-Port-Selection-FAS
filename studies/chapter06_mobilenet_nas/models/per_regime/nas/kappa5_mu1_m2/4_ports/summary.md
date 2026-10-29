# Rank-1 Per-Regime NAS Model — kappa5_mu1_m2 — 4 Ports

## Selection

- Regime: kappa5_mu1_m2
- Observed ports: 4
- Trial: 92
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 3
- Head width: 16
- Learning rate: 0.0013728321668526739
- Strict-INT8 validation outage probability: 0.175865
- Strict-INT8 test outage probability: 0.177095
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_92.keras | Keras v3 archive, FP32 weights | 594749 bytes | 49e76ed671fbd41562c0dc51310d2a571696384bc6fbed5ec8de1ce92af8772d |
| top_1_trial_92.tflite | TensorFlow Lite, strict INT8 | 312888 bytes | 515db123cd314776dad9caa5c919c1e1accb1ec8fc18e088baeff74f7fffb1a9 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m2_4_ports/optuna_study/.
