# Rank-1 Per-Regime NAS Model — kappa0_mu1_m50 — 3 Ports

## Selection

- Regime: kappa0_mu1_m50
- Observed ports: 3
- Trial: 66
- Database state: COMPLETE
- MobileNetV3 alpha: 0.6
- Truncation endpoint: v3_block_5_add
- Head depth: 1
- Head width: 48
- Learning rate: 0.0010548787421206605
- Strict-INT8 validation outage probability: 0.269175
- Strict-INT8 test outage probability: 0.26706
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_66.keras | Keras v3 archive, FP32 weights | 521064 bytes | 40d3629e3975c193d277db04731ef0dec49c5f01904e8865eedabc18eb20b1e4 |
| top_1_trial_66.tflite | TensorFlow Lite, strict INT8 | 273264 bytes | 6be0581571ae0ecb0e412f01c8a02ba05a1b6142cb6a02c980e5f0a34548d99d |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m50_3_ports/optuna_study/.
