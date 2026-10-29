# Rank-1 Per-Regime NAS Model — kappa0_mu1_m50 — 5 Ports

## Selection

- Regime: kappa0_mu1_m50
- Observed ports: 5
- Trial: 79
- Database state: RUNNING
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_4_add
- Head depth: 1
- Head width: 160
- Learning rate: 0.00015487147307467007
- Strict-INT8 validation outage probability: 0.21851
- Strict-INT8 test outage probability: 0.217105
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_79.keras | Keras v3 archive, FP32 weights | 442166 bytes | 1732aaa18e746170729d9eb0de6301ad6c7eb4c07f619fea2efce981bd25a570 |
| top_1_trial_79.tflite | TensorFlow Lite, strict INT8 | 234328 bytes | ae3d4edd2ca9e0753fae3722ab781f9e057427531dabd981cf11b6748a6a1abb |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m50_5_ports/optuna_study/.
