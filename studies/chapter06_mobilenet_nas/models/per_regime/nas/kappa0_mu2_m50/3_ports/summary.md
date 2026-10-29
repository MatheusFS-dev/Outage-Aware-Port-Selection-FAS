# Rank-1 Per-Regime NAS Model — kappa0_mu2_m50 — 3 Ports

## Selection

- Regime: kappa0_mu2_m50
- Observed ports: 3
- Trial: 66
- Database state: COMPLETE
- MobileNetV3 alpha: 0.2
- Truncation endpoint: v3_block_10_add
- Head depth: 1
- Head width: 192
- Learning rate: 0.005408170544676705
- Strict-INT8 validation outage probability: 0.198905
- Strict-INT8 test outage probability: 0.200885
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_66.keras | Keras v3 archive, FP32 weights | 788026 bytes | 8af35819fbe06b7a44417b1dab5356eb9bfe955006826b6da4f1e4cadcf74024 |
| top_1_trial_66.tflite | TensorFlow Lite, strict INT8 | 410752 bytes | 1463dd5890177a74426d15a14fc8f94de4bcadb18bce6e876323c8aeaa4a2df3 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu2_m50_3_ports/optuna_study/.
