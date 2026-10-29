# Rank-1 Per-Regime NAS Model — kappa0_mu1_m50 — 4 Ports

## Selection

- Regime: kappa0_mu1_m50
- Observed ports: 4
- Trial: 12
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_1_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0005628922880652142
- Strict-INT8 validation outage probability: 0.23165
- Strict-INT8 test outage probability: 0.23068
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_12.keras | Keras v3 archive, FP32 weights | 131676 bytes | 3500d644f5c188f653f77bae8953c2845493e201565fa0f1c4f9883100fbc361 |
| top_1_trial_12.tflite | TensorFlow Lite, strict INT8 | 93712 bytes | e4c0533f27724c5b9305406d4f093927bfccf5f2dfc1bbdf3342d7b1401c9431 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m50_4_ports/optuna_study/.
