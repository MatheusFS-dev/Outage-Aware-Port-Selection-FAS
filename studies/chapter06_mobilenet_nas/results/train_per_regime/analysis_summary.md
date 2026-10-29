# Per-Regime MobileNet Training Analysis

## Scope

This report summarizes the per-regime MobileNet safe-selector training campaign and its outage-probability (OP) validation outputs. The campaign contains:

- 5 training seeds (`0` through `4`)
- 3 backbones (`MobileNetV1`, `MobileNetV2`, and `MobileNetV3`)
- 14 channel regimes
- 5 observation budgets (`3` through `7` observed ports)

The resulting Cartesian product contains 1,050 training runs:

```text
5 seeds x 3 backbones x 14 regimes x 5 budgets = 1,050 runs
```

All 1,050 expected runs contain their training configuration, training history, selector history, model statistics, saved Keras model, and combined OP-evaluation summary. The OP evaluation used threshold `1.1885399`, evaluated 200,000 samples per run, and enabled only the validation split. Therefore, the findings below support model selection and validation analysis; they are not held-out test results.

## Experiment configuration

The runs use the `60/20/20` data split and train one specialized safe-port selector for each regime and observation budget. The configured decision policy chooses a known non-outage observed port when one is available and otherwise uses the MobileNet exploration scores. The training target is a uniform distribution over non-outage unobserved ports.

This is the standard per-regime training workflow. It is not the Optuna NAS plus post-training quantization workflow.

## Training results

The following values are aggregates across 350 runs per backbone. `Best selector OP` is the minimum model-only validation OP recorded in each run's selector history and then averaged across runs. Because this average combines different regimes and budgets, it is useful for comparing backbones on the common experiment grid but should not be interpreted as the OP of one homogeneous dataset.

| Backbone | Runs | Parameters | Mean recorded epochs | Mean selected epoch | Median selected epoch | Initial selector OP | Best selector OP | Mean OP improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV1 | 350 | 104,356 | 99.59 | 43.30 | 41.5 | 0.338022 | 0.314797 | 0.023225 |
| MobileNetV2 | 350 | 276,772 | 98.72 | 48.33 | 48.0 | 0.337388 | 0.314807 | 0.022581 |
| MobileNetV3 | 350 | 90,796 | 99.25 | 54.39 | 53.0 | 0.337205 | 0.314754 | 0.022451 |

The mean best selector OP is almost identical across the three backbones. The difference between the highest and lowest means is approximately `5.3e-5`. MobileNetV3 has the lowest mean by this metric, but the difference is too small to establish a meaningful practical advantage without a paired seed-level analysis.

MobileNetV1 generally selected its checkpoint earlier than the other backbones. Its median selected epoch was 41.5, compared with 48 for V2 and 53 for V3. This indicates that the V1 selector reached its selected validation checkpoint earlier on average, although the current analysis does not include wall-clock training time.

Mean loss values over the common grid were:

| Backbone | Final training loss | Final validation loss | Best validation loss |
|---|---:|---:|---:|
| MobileNetV1 | 3.015717 | 3.120623 | 3.116351 |
| MobileNetV2 | 3.029944 | 3.142268 | 3.131916 |
| MobileNetV3 | 3.077832 | 3.156091 | 3.148842 |

MobileNetV1 has the lowest mean losses. OP remains the primary decision metric because the deployed objective is outage avoidance rather than loss minimization alone.

### Model size interpretation

Parameter count gives the following architecture-size ordering:

1. MobileNetV3: 90,796 parameters
2. MobileNetV1: 104,356 parameters
3. MobileNetV2: 276,772 parameters

MobileNetV3 is therefore the smallest model by parameter count. It has approximately 13.0% fewer parameters than V1 and 67.2% fewer than V2. MobileNetV1 has approximately 62.3% fewer parameters than V2 but 14.9% more than V3.

The generated backbone score CSV currently reports serialized `model.keras` archive bytes instead of parameter count because `model_stats.txt` does not contain the `Model Size: ... B` text expected by the parser. The mean archive sizes are approximately 678,733 bytes for V1, 1,650,274 bytes for V2, and 853,759 bytes for V3. Archive size includes serialization overhead and must not be interpreted as parameter count. The parameter values above come directly from the consistent `decision_model_parameters` field present in all 350 reports for each backbone.

## Validation OP analysis

The backbone decision uses the normalized OP gap:

```text
(model-plus-observed OP - ideal OP) / (observed-only OP - ideal OP)
```

A smaller value is better: zero corresponds to ideal OP, while one corresponds to obtaining no improvement over the observed-only baseline.

| Rank | Backbone | Mean normalized OP gap | Maximum observed normalized gap | Mean best selector OP |
|---:|---|---:|---:|---:|
| 1 | MobileNetV1 | 0.176802 | 0.621622 | 0.314797 |
| 2 | MobileNetV2 | 0.181025 | 0.604651 | 0.314807 |
| 3 | MobileNetV3 | 0.187588 | 0.594595 | 0.314754 |

MobileNetV1 has the best average normalized OP gap. Its mean gap is approximately 2.3% lower than V2 and 5.8% lower than V3. MobileNetV3 has the lowest maximum observed gap, so the average-performance winner and the best worst-slice result are different.

The final rank remains `V1 > V2 > V3` because normalized OP gap is the primary ranking criterion. Correcting the interpretation of model size does not change this ranking.

## Effect of the observation budget

Mean normalized OP gap across regimes and seeds:

| Observed ports | MobileNetV1 | MobileNetV2 | MobileNetV3 | Best backbone |
|---:|---:|---:|---:|---|
| 3 | 0.344683 | 0.341776 | 0.344888 | MobileNetV2 |
| 4 | 0.293701 | 0.299678 | 0.293881 | MobileNetV1 |
| 5 | 0.135554 | 0.135741 | 0.143215 | MobileNetV1 |
| 6 | 0.058446 | 0.069128 | 0.082880 | MobileNetV1 |
| 7 | 0.051624 | 0.058804 | 0.073078 | MobileNetV1 |

MobileNetV1 is best for four of the five observation budgets. MobileNetV2 has a small advantage at three observed ports. The gap decreases sharply for every backbone as more ports become observed. From three to seven observed ports, the mean normalized gap decreases by approximately 85.0% for V1, 82.8% for V2, and 78.8% for V3.

The most substantial improvement occurs between four and five observed ports. Beyond six ports, V1 shows only a small additional average improvement, while V2 and V3 continue to improve but remain behind V1.

## Regime behavior

After averaging each backbone over seeds and observation budgets:

- MobileNetV1 has the lowest normalized OP gap in 7 of the 14 regimes.
- MobileNetV2 has the lowest normalized OP gap in the other 7 regimes.
- MobileNetV3 is not the lowest-gap backbone in any regime under this aggregation.
- `kappa5_mu1_m0` is the easiest regime for all three backbones.
- `kappa5_mu5_m0` is the hardest regime for all three backbones.

Mean normalized gaps for the easiest and hardest regimes are:

| Backbone | `kappa5_mu1_m0` | `kappa5_mu5_m0` |
|---|---:|---:|
| MobileNetV1 | 0.113293 | 0.249512 |
| MobileNetV2 | 0.113028 | 0.286675 |
| MobileNetV3 | 0.124680 | 0.294424 |

MobileNetV1 is notably stronger than V2 and V3 on the hardest average regime, even though V2 is marginally better on the easiest regime.

## Seed stability

For each backbone, regime, and observation budget, the normalized OP-gap standard deviation was computed across the five seeds. The following table summarizes those slice-level standard deviations:

| Backbone | Mean seed SD | Median seed SD | Maximum seed SD |
|---|---:|---:|---:|
| MobileNetV1 | 0.022109 | 0.011144 | 0.174528 |
| MobileNetV2 | 0.020689 | 0.012640 | 0.168183 |
| MobileNetV3 | 0.019497 | 0.008705 | 0.162979 |

MobileNetV3 has the lowest mean and median seed variability, despite ranking third in average normalized gap. The largest seed variation for every backbone occurs in `kappa5_mu5_m0`, confirming that this regime is both difficult and less stable across training seeds.

## Confidence intervals

The confidence analysis generated 3,150 Wilson interval rows:

- 1,050 for ideal OP
- 1,050 for model-plus-observed OP
- 1,050 for model-only OP

For model-plus-observed OP, the total 95% interval width ranges from 0.000882 to 0.003876, with a mean width of 0.002518. These intervals are narrow because every OP evaluation uses 200,000 validation samples.

The Wilson intervals quantify binomial sampling uncertainty within each OP evaluation. They do not quantify training-seed variability; the seed-level standard deviations reported above address that separate source of variation.

## Generated outputs

The completed analysis produced:

- [`backbone_decision/best_backbone_scores.csv`](backbone_decision/best_backbone_scores.csv)
- [`backbone_decision/regime_diagnostic.csv`](backbone_decision/regime_diagnostic.csv)
- [`backbone_decision/decision_summary.csv`](backbone_decision/decision_summary.csv)
- [`backbone_decision/plots/macro_avg_normalized_op_gap.png`](backbone_decision/plots/macro_avg_normalized_op_gap.png)
- [`backbone_decision/plots/worst_regime_normalized_op_gap.png`](backbone_decision/plots/worst_regime_normalized_op_gap.png)
- [`confidence_analysis/op_confidence_intervals.csv`](confidence_analysis/op_confidence_intervals.csv)

## Limitations and audit findings

1. **Validation only:** All 1,050 OP evaluations enabled `val` and disabled `test`. No held-out test conclusion can be drawn from this result tree.
2. **Parameter count versus archive size:** The backbone-decision CSV's `model_size` field is the serialized Keras archive size, not the architecture parameter count. Parameter-based comparisons must use `decision_model_parameters`.
3. **Missing deployment metrics:** RAM and inference-latency values are unavailable in the parsed score table, so the analysis cannot compare runtime resource use.
4. **Training-history provenance:** Although 1,026 runs record all 100 requested epochs, 24 histories contain fewer rows. Nine of those report a selected epoch beyond the current history length. This is consistent with resumed or truncated history artifacts, but the available files do not establish the cause. Their saved models and validation OP outputs are present.
5. **Rank heatmap:** The heatmap was not generated because the plotting function pivots duplicate seed-level keys without aggregating the five seeds first.
6. **Confidence forest plot:** The plot was not generated because 1,050 labeled validation rows produce an impractically tall figure and caused an X11 `BadAlloc` during rendering. The confidence table was written successfully.
7. **Dedicated test aggregate:** `analyze_per_regime_backbone_test.py` currently expects a stale `seed_*/trainings/.../test/...` path, while these results use `seed_*/train_.../val/...`.

## Conclusion

MobileNetV1 is the strongest overall choice for the current per-regime validation campaign. It provides the lowest mean normalized OP gap, wins four of five observation budgets, performs best in half of the regimes, and has substantially fewer parameters than V2. MobileNetV3 remains relevant when parameter count or seed stability is prioritized: it is the smallest backbone, has the lowest average seed variability, and has the best maximum-gap result. MobileNetV2 provides the best mean gap at the most constrained three-port budget and wins the other half of the regime-level comparisons, but it is by far the largest architecture.

The evidence supports selecting MobileNetV1 for average OP performance, subject to confirmation on the held-out test split. A final model claim should wait until test OP evaluation is generated and the same comparisons are repeated without using validation data for final reporting.
