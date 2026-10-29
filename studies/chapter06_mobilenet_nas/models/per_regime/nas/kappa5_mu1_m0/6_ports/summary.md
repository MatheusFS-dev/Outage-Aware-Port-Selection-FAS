# Rank-1 Per-Regime NAS Model — kappa5_mu1_m0 — 6 Ports

## Selection

- Regime: kappa5_mu1_m0
- Observed ports: 6
- Trial: 4
- Database state: COMPLETE
- MobileNetV3 alpha: 0.8
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 2
- Head width: 240
- Learning rate: 0.0024269631383790876
- Strict-INT8 validation outage probability: 0.98026
- Strict-INT8 test outage probability: 0.979975
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_4.keras | Keras v3 archive, FP32 weights | 623161 bytes | 66ce1beb42ac7c074f8d0657fef3107bb15707798021f1d9b355482dbe8bfbde |
| top_1_trial_4.tflite | TensorFlow Lite, strict INT8 | 268920 bytes | 1aeb1b354813a1c2c53361a16169ffcabf359097ca0fd4bdfeec8d01bb17ddaf |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu1_m0_6_ports/optuna_study/.
