# Rank-1 Per-Regime NAS Model — kappa0_mu1_m0 — 5 Ports

## Selection

- Regime: kappa0_mu1_m0
- Observed ports: 5
- Trial: 76
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0010717422224481982
- Strict-INT8 validation outage probability: 0.21568
- Strict-INT8 test outage probability: 0.21593
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_76.keras | Keras v3 archive, FP32 weights | 986294 bytes | 88ce45bbe2d76fd6eebc16ba9cb7c979e92989477183e7514acc7882f1670aee |
| top_1_trial_76.tflite | TensorFlow Lite, strict INT8 | 469408 bytes | 854fb47ed095b09b2b9a976b7d2f00b14a511fad312b4ac00f4ef9c73313881e |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m0_5_ports/optuna_study/.
