# Rank-1 Per-Regime NAS Model — kappa0_mu2_m50 — 7 Ports

## Selection

- Regime: kappa0_mu2_m50
- Observed ports: 7
- Trial: 7
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 112
- Learning rate: 0.0002628376030296237
- Strict-INT8 validation outage probability: 0.12324
- Strict-INT8 test outage probability: 0.123895
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_7.keras | Keras v3 archive, FP32 weights | 1018850 bytes | 204b4a050e289001aa4647575605c443217423452705802944eae7c91f558358 |
| top_1_trial_7.tflite | TensorFlow Lite, strict INT8 | 453352 bytes | 432be1b36dd7c8e9ade25e303d5b6d6965e0dce255f5dc8b6bc4ff3e739da8f9 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu2_m50_7_ports/optuna_study/.
