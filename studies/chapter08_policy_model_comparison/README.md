# Chapter 8: Policy–Model Comparison

Budget-specific UNI/MAR/GC learned selectors, independent training seeds, CPU screening and held-out aggregation.

See the [root README](../../README.md) for the dissertation abstract, environment setup, licenses and chapter map. The [source manifest](../../docs/source-manifest.json) records exact original paths and checksums.

## Contents

- `scripts/`: study entry points and analysis code.
- `results/`: retained numerical evidence, configurations and applicable final artifacts.
- `figures/`: standalone dissertation figures and their available sources.

## Protocol

The retained learned-policy campaign uses UNI, MAR and GC, observation budgets 3–7, and seeds 0–4. Each policy/seed is independently initialized. NAS-selected structures come from Chapter 6. The CPU screening configuration explicitly records random initialization and must not be interpreted as a trained-model outage comparison.
