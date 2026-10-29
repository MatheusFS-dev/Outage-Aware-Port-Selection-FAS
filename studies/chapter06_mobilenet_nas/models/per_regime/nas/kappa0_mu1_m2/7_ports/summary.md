# Rank-1 Per-Regime NAS Model — kappa0_mu1_m2 — 7 Ports

## Selection

- Regime: kappa0_mu1_m2
- Observed ports: 7
- Trial: 46
- Database state: COMPLETE
- MobileNetV3 alpha: 1.0
- Truncation endpoint: v3_block_8_project_gn
- Head depth: 1
- Head width: 192
- Learning rate: 0.00027265049507699615
- Strict-INT8 validation outage probability: 0.21302
- Strict-INT8 test outage probability: 0.211
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_46.keras | Keras v3 archive, FP32 weights | 1677539 bytes | 078f3886ab00edb9a8d6a78a83ba8bd2e7facf70500d68c2002666d230047983 |
| top_1_trial_46.tflite | TensorFlow Lite, strict INT8 | 684448 bytes | 94147f7519d521f809159d8b7efdec6480590d784fc02136a9048955673fd0e3 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa0_mu1_m2_7_ports/optuna_study/.
