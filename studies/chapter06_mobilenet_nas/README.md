# Chapter 6: MobileNet-Based Neural Architecture Search

Backbone comparison, all-regime and per-regime MobileNetV3 NAS, quantization, held-out analysis and Pico deployment.

See the [root README](../../README.md) for the dissertation abstract, environment setup, licenses and chapter map. The [source manifest](../../docs/source-manifest.json) records exact original paths and checksums.

## Contents

- `scripts/`: study entry points and analysis code.
- `results/`: retained numerical evidence, configurations and applicable final artifacts.
- `figures/`: standalone dissertation figures and their available sources.

## Deployment

`models/` contains selected exported models. Final model artifacts also remain beside their configurations in `results/` to preserve analysis relationships. `firmware/` contains Pico source, the existing build instructions and selected UF2 images. Building requires the Pico SDK and pico-tflmicro paths referenced by `firmware/CMakeLists.txt`. Hardware measurements require the actual board and INA219 setup.
