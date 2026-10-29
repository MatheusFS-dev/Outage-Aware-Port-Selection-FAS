# Rank-1 Per-Regime NAS Model — kappa5_mu5_m50 — 6 Ports

## Selection

- Regime: kappa5_mu5_m50
- Observed ports: 6
- Trial: 68
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 224
- Learning rate: 0.00036220145983404496
- Strict-INT8 validation outage probability: 0.01152
- Strict-INT8 test outage probability: 0.011315
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_68.keras | Keras v3 archive, FP32 weights | 1503800 bytes | 6bd1540a366c7988c3a0a52207fbd3bd6c1dfe08174b13f17cca1c2536ea49ad |
| top_1_trial_68.tflite | TensorFlow Lite, strict INT8 | 629064 bytes | c807dc9306cf0641490e763c97b212b935f28657a3309d0d6d69f4b021dc7172 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m50_6_ports/optuna_study/.
