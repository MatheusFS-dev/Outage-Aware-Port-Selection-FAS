# Rank-1 Per-Regime NAS Model — kappa0_mu1_m0 — 4 Ports

## Selection

- Regime: kappa0_mu1_m0
- Observed ports: 4
- Trial: 54
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 64
- Learning rate: 0.0004299453295865066
- Strict-INT8 validation outage probability: 0.230975
- Strict-INT8 test outage probability: 0.23104
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_54.keras | Keras v3 archive, FP32 weights | 315148 bytes | e37372cd3cac8f6a5e3ea5cb5c3e08b757a249c9e5e45e899c5b8c03b08e54a7 |
| top_1_trial_54.tflite | TensorFlow Lite, strict INT8 | 181656 bytes | 21a90832d77106546c3c9ddda4c1b561ee65860f61cd07fd8c5bef5b35cf1a43 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m0_4_ports/optuna_study/.
