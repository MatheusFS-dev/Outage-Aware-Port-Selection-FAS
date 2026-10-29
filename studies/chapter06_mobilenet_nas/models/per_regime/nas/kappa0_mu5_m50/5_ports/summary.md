# Rank-1 Per-Regime NAS Model — kappa0_mu5_m50 — 5 Ports

## Selection

- Regime: kappa0_mu5_m50
- Observed ports: 5
- Trial: 88
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_6_project_gn
- Head depth: 2
- Head width: 16
- Learning rate: 0.0005863305992644147
- Strict-INT8 validation outage probability: 0.05046
- Strict-INT8 test outage probability: 0.050755
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_88.keras | Keras v3 archive, FP32 weights | 966741 bytes | 9a25f5b5ff72be68f9f456b967dca80c55bb0386389e58e4ff9a96b5ed16700d |
| top_1_trial_88.tflite | TensorFlow Lite, strict INT8 | 435848 bytes | 7bb929fe31b7b78b581daa80ca26804ede54c62a7800a9f7c95329fe88609bb1 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu5_m50_5_ports/optuna_study/.
