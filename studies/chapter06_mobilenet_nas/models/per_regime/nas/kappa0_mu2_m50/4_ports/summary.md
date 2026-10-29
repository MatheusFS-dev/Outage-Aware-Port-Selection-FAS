# Rank-1 Per-Regime NAS Model — kappa0_mu2_m50 — 4 Ports

## Selection

- Regime: kappa0_mu2_m50
- Observed ports: 4
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.140565
- Strict-INT8 test outage probability: 0.14157
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018814 bytes | ce3d9f5f769e6b90cb3abd8d2d5c4749db042719d20e5df5796b7740e2bc7648 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 449968 bytes | c5e35b557e227efdbc894bd3da91da0bda35638eeba864b25caadbd1fbae4420 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu2_m50_4_ports/optuna_study/.
