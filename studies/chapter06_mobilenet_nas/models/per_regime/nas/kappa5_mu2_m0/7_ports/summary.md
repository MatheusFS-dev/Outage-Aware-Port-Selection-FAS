# Rank-1 Per-Regime NAS Model — kappa5_mu2_m0 — 7 Ports

## Selection

- Regime: kappa5_mu2_m0
- Observed ports: 7
- Trial: 55
- Database state: COMPLETE
- MobileNetV3 alpha: 0.3
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 128
- Learning rate: 0.004064896523330553
- Strict-INT8 validation outage probability: 0.979965
- Strict-INT8 test outage probability: 0.97899
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_55.keras | Keras v3 archive, FP32 weights | 622389 bytes | 696ecfa1947d4d79a941df69dc83c2b6cc0b98bb547f5d423c5596fdaeb25fc8 |
| top_1_trial_55.tflite | TensorFlow Lite, strict INT8 | 338976 bytes | f5695aff9e3c34aed69948f3e28a4ad67127ba128f199e10aea69492f352b1f4 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m0_7_ports/optuna_study/.
