# Rank-1 Per-Regime NAS Model — kappa5_mu1_m50 — 3 Ports

## Selection

- Regime: kappa5_mu1_m50
- Observed ports: 3
- Trial: 54
- Database state: COMPLETE
- MobileNetV3 alpha: 0.7
- Truncation endpoint: v3_block_5_add
- Head depth: 2
- Head width: 176
- Learning rate: 0.0022684150732580886
- Strict-INT8 validation outage probability: 0.16813
- Strict-INT8 test outage probability: 0.170275
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_54.keras | Keras v3 archive, FP32 weights | 779581 bytes | 805a5e0cdfc10c7c0cbe478896b173cc68ff8f0bbe47cd5454eea4b539898fc7 |
| top_1_trial_54.tflite | TensorFlow Lite, strict INT8 | 351768 bytes | 50f65c7f7e5dbf462be33134d228d4557119cb4b8748bf67042cd7dc0889876b |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m50_3_ports/optuna_study/.
