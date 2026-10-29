# Rank-1 Per-Regime NAS Model — kappa5_mu1_m0 — 3 Ports

## Selection

- Regime: kappa5_mu1_m0
- Observed ports: 3
- Trial: 13
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 3
- Head width: 240
- Learning rate: 0.0014677047728814249
- Strict-INT8 validation outage probability: 0.981485
- Strict-INT8 test outage probability: 0.981135
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_13.keras | Keras v3 archive, FP32 weights | 834047 bytes | 30ff5a5bc117c3cd2f27cb345b05fd2133fa855171c4efeee397479fd23f0923 |
| top_1_trial_13.tflite | TensorFlow Lite, strict INT8 | 321584 bytes | e9be144ba5ca24e647d0f3180ce3db1d48fb91a0e817f31695fa1ea8dd28ca18 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m0_3_ports/optuna_study/.
