# Rank-1 Per-Regime NAS Model — kappa5_mu1_m0 — 7 Ports

## Selection

- Regime: kappa5_mu1_m0
- Observed ports: 7
- Trial: 76
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_5_add
- Head depth: 3
- Head width: 32
- Learning rate: 0.0026041742239892676
- Strict-INT8 validation outage probability: 0.98122
- Strict-INT8 test outage probability: 0.981505
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_76.keras | Keras v3 archive, FP32 weights | 474065 bytes | 4a69c7d3996df158bd95cbaac68e6e3419afdcecfeda35cc644d5729d1947a62 |
| top_1_trial_76.tflite | TensorFlow Lite, strict INT8 | 261176 bytes | 2ba57a8da4029ff94207c710c8a569687b624bb1d58f8472c5050b7e35247bca |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m0_7_ports/optuna_study/.
