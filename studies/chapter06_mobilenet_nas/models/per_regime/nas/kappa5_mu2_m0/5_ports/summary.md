# Rank-1 Per-Regime NAS Model — kappa5_mu2_m0 — 5 Ports

## Selection

- Regime: kappa5_mu2_m0
- Observed ports: 5
- Trial: 83
- Database state: COMPLETE
- MobileNetV3 alpha: 0.3
- Truncation endpoint: v3_block_5_add
- Head depth: 3
- Head width: 160
- Learning rate: 0.0015418839998492544
- Strict-INT8 validation outage probability: 0.981085
- Strict-INT8 test outage probability: 0.980395
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_83.keras | Keras v3 archive, FP32 weights | 637514 bytes | e9b7552f388a7d5fdb91f19a3896e2439d4a153d2f1717ef0fd0a4947262f0cd |
| top_1_trial_83.tflite | TensorFlow Lite, strict INT8 | 297816 bytes | c409b43851c61791cde27f8581fd97072cc587af86ee133ec41755b5f95a300f |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m0_5_ports/optuna_study/.
