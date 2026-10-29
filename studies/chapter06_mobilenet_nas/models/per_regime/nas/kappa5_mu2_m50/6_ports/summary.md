# Rank-1 Per-Regime NAS Model — kappa5_mu2_m50 — 6 Ports

## Selection

- Regime: kappa5_mu2_m50
- Observed ports: 6
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.03743
- Strict-INT8 test outage probability: 0.037645
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018838 bytes | ecb5242d22deee38865081f94072b4b24db0b7193b0b691803178db236939776 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 452224 bytes | 99c197644a93c92e15806411cd848c64eaa007d29f5f828869db9f86792a2cb7 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m50_6_ports/optuna_study/.
