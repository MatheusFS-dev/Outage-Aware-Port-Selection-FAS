# Rank-1 Per-Regime NAS Model — kappa0_mu1_m0 — 3 Ports

## Selection

- Regime: kappa0_mu1_m0
- Observed ports: 3
- Trial: 64
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 2
- Head width: 16
- Learning rate: 0.00021982784159626137
- Strict-INT8 validation outage probability: 0.26579
- Strict-INT8 test outage probability: 0.265815
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_64.keras | Keras v3 archive, FP32 weights | 991405 bytes | 4bb1c591b7ecd73dbc501676e65707902afd3ca8fd49b923543ec0efc728fe91 |
| top_1_trial_64.tflite | TensorFlow Lite, strict INT8 | 468800 bytes | 888c980ef10c09fabe599ae0bbf8b142b9af43554c87638a0294231ff2310307 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m0_3_ports/optuna_study/.
