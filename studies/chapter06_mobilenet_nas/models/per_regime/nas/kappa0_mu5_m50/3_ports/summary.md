# Rank-1 Per-Regime NAS Model — kappa0_mu5_m50 — 3 Ports

## Selection

- Regime: kappa0_mu5_m50
- Observed ports: 3
- Trial: 66
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_7_add
- Head depth: 1
- Head width: 96
- Learning rate: 0.00033423273493959286
- Strict-INT8 validation outage probability: 0.138205
- Strict-INT8 test outage probability: 0.138305
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_66.keras | Keras v3 archive, FP32 weights | 807855 bytes | 51d186d58a989aaee7f6721eb02992a4e8f6497149f9a6e870d2418c6a828f18 |
| top_1_trial_66.tflite | TensorFlow Lite, strict INT8 | 393632 bytes | cd3ad05688755f33b5923c6b18199a99eec30c553f667f94441229541861e2f1 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu5_m50_3_ports/optuna_study/.
