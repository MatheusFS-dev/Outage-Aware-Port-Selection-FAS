# Rank-1 Per-Regime NAS Model — kappa5_mu5_m0 — 7 Ports

## Selection

- Regime: kappa5_mu5_m0
- Observed ports: 7
- Trial: 2
- Database state: COMPLETE
- MobileNetV3 alpha: 0.5
- Truncation endpoint: v3_block_10_add
- Head depth: 1
- Head width: 256
- Learning rate: 0.0008655704137361819
- Strict-INT8 validation outage probability: 0.975675
- Strict-INT8 test outage probability: 0.975835
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_2.keras | Keras v3 archive, FP32 weights | 1604591 bytes | f7e1be9df7ddfd631610f6b68d57610c6a553258141cb90f36d44f2cde2fe74f |
| top_1_trial_2.tflite | TensorFlow Lite, strict INT8 | 685352 bytes | 0e40a844f92f4811be9eb7bfef1e54f6cad5c5ab8684c2fc54ff3cc8984cc0fc |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu5_m0_7_ports/optuna_study/.
