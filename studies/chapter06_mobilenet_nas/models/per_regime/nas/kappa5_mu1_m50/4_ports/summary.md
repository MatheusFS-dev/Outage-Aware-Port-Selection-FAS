# Rank-1 Per-Regime NAS Model — kappa5_mu1_m50 — 4 Ports

## Selection

- Regime: kappa5_mu1_m50
- Observed ports: 4
- Trial: 70
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_2_add
- Head depth: 1
- Head width: 208
- Learning rate: 0.0002035407019347207
- Strict-INT8 validation outage probability: 0.10117
- Strict-INT8 test outage probability: 0.100785
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_70.keras | Keras v3 archive, FP32 weights | 256013 bytes | 596f06c9ef2c4e2f0c055dc8346364f38a01c040db5e4c8f02d28c7c4e54b287 |
| top_1_trial_70.tflite | TensorFlow Lite, strict INT8 | 143064 bytes | eab245788732526d7c14346aab358846d043a1b9f82b39bda22ed57f7b12d578 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m50_4_ports/optuna_study/.
