# Rank-1 Per-Regime NAS Model — kappa5_mu2_m0 — 4 Ports

## Selection

- Regime: kappa5_mu2_m0
- Observed ports: 4
- Trial: 62
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 2
- Head width: 160
- Learning rate: 0.0006532698901609879
- Strict-INT8 validation outage probability: 0.982545
- Strict-INT8 test outage probability: 0.98189
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_62.keras | Keras v3 archive, FP32 weights | 944973 bytes | ab0e320993d220663ebf67edc93fb5ca36c1722830a584bc80c6705e0797e7a6 |
| top_1_trial_62.tflite | TensorFlow Lite, strict INT8 | 444240 bytes | 6c2e0e2757b144cb94997f9ca9b0482a368301673f12a5b6410ca844848ef12e |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m0_4_ports/optuna_study/.
