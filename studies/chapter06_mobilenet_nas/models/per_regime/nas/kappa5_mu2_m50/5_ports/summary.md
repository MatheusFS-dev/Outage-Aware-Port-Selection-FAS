# Rank-1 Per-Regime NAS Model — kappa5_mu2_m50 — 5 Ports

## Selection

- Regime: kappa5_mu2_m50
- Observed ports: 5
- Trial: 73
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 2
- Head width: 16
- Learning rate: 0.00046272195178975804
- Strict-INT8 validation outage probability: 0.042375
- Strict-INT8 test outage probability: 0.0425
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_73.keras | Keras v3 archive, FP32 weights | 966742 bytes | f905e67cc786e0c2893df48b33a3a4a56befb46b20552a2b875958f847c4dfc8 |
| top_1_trial_73.tflite | TensorFlow Lite, strict INT8 | 435848 bytes | e9aa575a020eb81ecd5e6b63d018cedd66c5afb73706ec00360f6cb79085eb2f |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m50_5_ports/optuna_study/.
