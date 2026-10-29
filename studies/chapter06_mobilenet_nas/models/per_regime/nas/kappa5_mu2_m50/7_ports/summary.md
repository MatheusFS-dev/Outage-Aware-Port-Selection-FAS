# Rank-1 Per-Regime NAS Model — kappa5_mu2_m50 — 7 Ports

## Selection

- Regime: kappa5_mu2_m50
- Observed ports: 7
- Trial: 77
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 160
- Learning rate: 0.0003196065950193867
- Strict-INT8 validation outage probability: 0.036755
- Strict-INT8 test outage probability: 0.037265
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_77.keras | Keras v3 archive, FP32 weights | 1455428 bytes | 9371ab677f968f9f17d148647f8e152a98e47a17aa513b541de0b86241d84b2a |
| top_1_trial_77.tflite | TensorFlow Lite, strict INT8 | 616368 bytes | 39b549310cc216cf0ff38f9f679a8f6eae8e915cb3335d3568a15fb484399f7f |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m50_7_ports/optuna_study/.
