# Rank-1 Per-Regime NAS Model — kappa5_mu1_m2 — 5 Ports

## Selection

- Regime: kappa5_mu1_m2
- Observed ports: 5
- Trial: 86
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.00019387903900303614
- Strict-INT8 validation outage probability: 0.16111
- Strict-INT8 test outage probability: 0.16208
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_86.keras | Keras v3 archive, FP32 weights | 961606 bytes | 9a8030f55cf85b4d194433f5afde7ef5d6b4d13e08af973e3decd874c9483c10 |
| top_1_trial_86.tflite | TensorFlow Lite, strict INT8 | 434200 bytes | 55bfd1cad7b820fd391afa6f76be5d4d2747e6d802e3dbf52d82d3aa47c34c46 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m2_5_ports/optuna_study/.
