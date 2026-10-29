# Rank-1 Per-Regime NAS Model — kappa5_mu2_m50 — 3 Ports

## Selection

- Regime: kappa5_mu2_m50
- Observed ports: 3
- Trial: 92
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_4_add
- Head depth: 1
- Head width: 16
- Learning rate: 0.0003880420945683874
- Strict-INT8 validation outage probability: 0.128875
- Strict-INT8 test outage probability: 0.129215
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_92.keras | Keras v3 archive, FP32 weights | 337758 bytes | 0e6ef737af220a7b512fce33a43e8c7a46f802036dd96175276f4bf6396b6dc1 |
| top_1_trial_92.tflite | TensorFlow Lite, strict INT8 | 198024 bytes | 90680e3ee441306f3b55fde8afd41e584dfaa355aaa34071a05bad3992635c4e |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m50_3_ports/optuna_study/.
