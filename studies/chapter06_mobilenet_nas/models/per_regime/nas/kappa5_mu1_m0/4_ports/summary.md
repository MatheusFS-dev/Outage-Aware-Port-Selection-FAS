# Rank-1 Per-Regime NAS Model — kappa5_mu1_m0 — 4 Ports

## Selection

- Regime: kappa5_mu1_m0
- Observed ports: 4
- Trial: 60
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 240
- Learning rate: 0.001303142346008352
- Strict-INT8 validation outage probability: 0.98163
- Strict-INT8 test outage probability: 0.981
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_60.keras | Keras v3 archive, FP32 weights | 1515872 bytes | 1f4637f777ff7c0bb3e73495653c3e05256c1d382340d6e5dc04738fbafae708 |
| top_1_trial_60.tflite | TensorFlow Lite, strict INT8 | 630264 bytes | 2a92e955e4d931a6506fc3b035016d5dc2e4701c36f0104288e1fcf3085f36ed |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m0_4_ports/optuna_study/.
