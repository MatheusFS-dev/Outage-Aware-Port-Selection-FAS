# Rank-1 Per-Regime NAS Model — kappa5_mu5_m2 — 3 Ports

## Selection

- Regime: kappa5_mu5_m2
- Observed ports: 3
- Trial: 60
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_2_add
- Head depth: 1
- Head width: 224
- Learning rate: 0.004844479055072981
- Strict-INT8 validation outage probability: 0.185575
- Strict-INT8 test outage probability: 0.182415
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_60.keras | Keras v3 archive, FP32 weights | 296480 bytes | 112ce76cd220e63f2fd4adcb8953b80336b1b0e84b232614e5e3f216c769cbf4 |
| top_1_trial_60.tflite | TensorFlow Lite, strict INT8 | 157656 bytes | 064fb9f6a0f1d790dd3fe6ef370a2e6933bb303832543036614d19d8f45ffdab |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m2_3_ports/optuna_study/.
