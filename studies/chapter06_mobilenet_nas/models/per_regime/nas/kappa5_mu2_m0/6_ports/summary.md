# Rank-1 Per-Regime NAS Model — kappa5_mu2_m0 — 6 Ports

## Selection

- Regime: kappa5_mu2_m0
- Observed ports: 6
- Trial: 4
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 2
- Head width: 240
- Learning rate: 0.0024269631383790876
- Strict-INT8 validation outage probability: 0.98024
- Strict-INT8 test outage probability: 0.97984
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_4.keras | Keras v3 archive, FP32 weights | 623160 bytes | 0206810223ad7dd9c50528ce793e14585e002b3a7263b315b67af1fa2b41aa23 |
| top_1_trial_4.tflite | TensorFlow Lite, strict INT8 | 268920 bytes | da85ef4e090fd6f8f2a1db9ececb23aaa01c65782c6ec523105f5e74aeea7cc4 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m0_6_ports/optuna_study/.
