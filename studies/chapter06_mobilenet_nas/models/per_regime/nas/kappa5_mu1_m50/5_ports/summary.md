# Rank-1 Per-Regime NAS Model — kappa5_mu1_m50 — 5 Ports

## Selection

- Regime: kappa5_mu1_m50
- Observed ports: 5
- Trial: 67
- Database state: COMPLETE
- MobileNetV3 alpha: 0.4
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 32
- Learning rate: 0.0010561972300095327
- Strict-INT8 validation outage probability: 0.089105
- Strict-INT8 test outage probability: 0.089375
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_67.keras | Keras v3 archive, FP32 weights | 642910 bytes | 6bc8a884e85f38dde4bb72b2a0ffd64db70f73cf74a2a91761a36ac614daaec5 |
| top_1_trial_67.tflite | TensorFlow Lite, strict INT8 | 350416 bytes | c8045c07c367495247afd1ba04f100ac181488c8868c5d457e1d9f634a3c5483 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m50_5_ports/optuna_study/.
