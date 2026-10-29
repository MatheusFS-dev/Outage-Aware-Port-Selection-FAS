# Rank-1 Per-Regime NAS Model — kappa5_mu1_m2 — 6 Ports

## Selection

- Regime: kappa5_mu1_m2
- Observed ports: 6
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.15959
- Strict-INT8 test outage probability: 0.160555
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018837 bytes | f9df4892ff924181d5f76776b97b7a0b974f12a39a192b27b5f5b7ba8f68eafd |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 452224 bytes | 636e911edab61deb191036f898ecbe592169a5bf9c3acc74bf66a77e24e729d1 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m2_6_ports/optuna_study/.
