# Rank-1 Per-Regime NAS Model — kappa5_mu2_m2 — 6 Ports

## Selection

- Regime: kappa5_mu2_m2
- Observed ports: 6
- Trial: 74
- Database state: COMPLETE
- MobileNetV3 alpha: 0.9
- Truncation endpoint: v3_block_3_project_gn
- Head depth: 1
- Head width: 16
- Learning rate: 0.0012548309199509652
- Strict-INT8 validation outage probability: 0.12575
- Strict-INT8 test outage probability: 0.12628
- Selection source: results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv

## Artifacts

| Artifact | Format | Exact size | SHA-256 |
|---|---|---:|---|
| top_1_trial_74.keras | Keras v3 archive, FP32 weights | 279908 bytes | fbb83ca49f9b1e5a4d1ed849af16a417c05d6a53cb671d30ecbcdbda59890418 |
| top_1_trial_74.tflite | TensorFlow Lite, strict INT8 | 172336 bytes | ab7720b55c40615c035d00c0dde04984ba839b1926d8a76daa3d7b4121766347 |

The files are byte-for-byte copies of the selected rank-1 artifacts in results/nas_ptq_mobv3_per_regime/nas_ptq_mobv3_regime_kappa5_mu2_m2_6_ports/optuna_study/.
