![Header](https://capsule-render.vercel.app/api?type=blur&height=180&color=264653&text=Outage-Aware%20Port%20Selection%20FAS&fontSize=30&fontColor=ffffff)

<div align="center">

```text
 ██████╗ ██╗   ██╗████████╗ █████╗  ██████╗ ███████╗     █████╗ ██╗    ██╗ █████╗ ██████╗ ███████╗
██╔═══██╗██║   ██║╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔════╝
██║   ██║██║   ██║   ██║   ███████║██║  ███╗█████╗      ███████║██║ █╗ ██║███████║██████╔╝█████╗  
██║   ██║██║   ██║   ██║   ██╔══██║██║   ██║██╔══╝      ██╔══██║██║███╗██║██╔══██║██╔══██╗██╔══╝  
╚██████╔╝╚██████╔╝   ██║   ██║  ██║╚██████╔╝███████╗    ██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗
 ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝


██████╗  ██████╗ ██████╗ ████████╗    ███████╗███████╗██╗     ███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝    ██╔════╝██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
██████╔╝██║   ██║██████╔╝   ██║       ███████╗█████╗  ██║     █████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
██╔═══╝ ██║   ██║██╔══██╗   ██║       ╚════██║██╔══╝  ██║     ██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
██║     ╚██████╔╝██║  ██║   ██║       ███████║███████╗███████╗███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚══════╝╚══════╝╚══════╝╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                                             
```

</div>

<p align="center"><img src="https://readme-typing-svg.herokuapp.com?font=Georgia&color=264653&center=true&width=650&lines=From+partial+channel+observation+to+port+selection" alt="From partial channel observation to port selection"></p>

<p align="center">
<a href="LICENSE.txt"><img src="https://img.shields.io/badge/Code-Apache%202.0-blue?style=flat-square" alt="Code: Apache 2.0"></a>
<a href="LICENSE-DOCS.txt"><img src="https://img.shields.io/badge/Docs%20%26%20figures-CC%20BY%204.0-green?style=flat-square" alt="Documentation and figures: CC BY 4.0"></a>
<a href="https://github.com/MatheusFS-dev/Outage-Aware-Port-Selection-FAS/stargazers"><img src="https://img.shields.io/github/stars/MatheusFS-dev/Outage-Aware-Port-Selection-FAS?style=flat-square" alt="Stars"></a>
<a href="https://github.com/MatheusFS-dev/Outage-Aware-Port-Selection-FAS/forks"><img src="https://img.shields.io/github/forks/MatheusFS-dev/Outage-Aware-Port-Selection-FAS?style=flat-square" alt="Forks"></a>
<a href="https://visitor-badge.laobi.icu/badge?page_id=MatheusFS-dev.Outage-Aware-Port-Selection-FAS"><img src="https://visitor-badge.laobi.icu/badge?page_id=MatheusFS-dev.Outage-Aware-Port-Selection-FAS" alt="Visitors"></a>
</p>

This repository contains the code, selected models, numerical results and standalone figure material accompanying the master's dissertation **Outage-Aware Port Selection for Fluid Antenna Systems in Spatially Correlated κ–μ Shadowed Fading: Partial Channel Observation, Reconstruction, and Port-Reading Policies**, by **Matheus Ferreira Silva**, Instituto Nacional de Telecomunicações (Inatel), 2026.

The material is organized by dissertation chapter and includes the original CNN study. The dissertation manuscript and channel datasets are not distributed here.

## Contents

- [Dissertation abstract](#dissertation-abstract)
- [Studies and chapter map](#studies-and-chapter-map)
- [Installation](#installation)
- [Usage and reproducibility](#usage-and-reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Dissertation abstract

Fluid antenna systems (FAS) exploit spatial diversity through port selection, but reading every candidate port increases channel-acquisition demand. This work evaluates measurement-efficient selection from partial observations through profile reconstruction, observed-only reading policies, and policy-dependent learned inference. The simulated downlink comprises 100 uniformly spaced ports over a unit-normalized aperture. One million channel realizations are generated for each of 14 spatially correlated κ–μ shadowed fading regimes. One-dimensional convolutional neural networks (CNNs) reconstruct the full profile from uniform measurements. In the representative regimes, reconstruction reduces outage probability (OP) relative to observed-only selection and approaches the full-observation reference with approximately five or six measurements. For three to seven observations, UNI, MAR, and GC are compared through observed-only selection and as inputs to budget-specific MobileNetV3 selectors. Under the observed-only protocol, GC performs best in most regimes with three observed ports, whereas UNI performs best in most regimes with four or more observed ports. Across 75 independently initialized training runs, this ordering changes at four observations: GC leads before learned inference, whereas UNI leads after both FP32 and INT8 inference. For the evaluated configurations, the learned selectors reduce mean outage probability relative to observed-only selection in every aggregate policy–budget combination, including after quantization.

**Keywords:** fluid antenna systems, port selection, partial observation, port-reading policies, MobileNet.

## Studies and chapter map

| Chapter | Study | Material |
| --- | --- | --- |
| 4 | [System Model, Dataset, and Evaluation Protocol](studies/chapter04_system_model/README.md) | MATLAB channel generation, Python runtime-cache conversion, regime definitions and outage-threshold analysis. |
| 5 | [Low-Power CNN-Based Port Selection](studies/chapter05_low_power_cnn/README.md) | Original WCNPS 2025 CNN notebooks and selected artifacts, plus the later spatial-correlation diagnostic. |
| 6 | [MobileNet-Based Neural Architecture Search](studies/chapter06_mobilenet_nas/README.md) | Backbone comparison, all-regime and per-regime MobileNetV3 NAS, quantization, held-out analysis and Pico deployment. |
| 7 | [Measurement-Efficient Port-Reading Policies](studies/chapter07_reading_policies/README.md) | Observed-only reading policies, Monte Carlo comparisons, construction-time and threshold-sensitivity analysis. |
| 8 | [Policy–Model Comparison](studies/chapter08_policy_model_comparison/README.md) | Budget-specific UNI/MAR/GC learned selectors, independent training seeds, CPU screening and held-out aggregation. |

Shared Python utilities live in `src/`. Study entry points, results and figure assets live in `studies/`. The [source manifest](docs/source-manifest.json) records source revisions, original paths, destination paths and SHA-256 checksums. [Reproduction notes](docs/reproducibility.md) explain retained artifacts and limits.

## Installation

The current studies use Python 3.12 on Linux x86-64. From the repository root, install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test
```

The historical Chapter 5 notebooks have separate dependencies described in their [chapter guide](studies/chapter05_low_power_cnn/README.md). MATLAB is required for channel generation. Firmware builds need the Pico SDK and pico-tflmicro, as explained in the [firmware guide](studies/chapter06_mobilenet_nas/firmware/README.md).

## Usage and reproducibility

Inspect retained CSV/JSON summaries and figures without downloading a dataset. Run the automated checks from the repository root:

```bash
uv run python3 -m pytest
```

The all-regime NAS summaries can also be rebuilt from retained tables, without channel data. This writes to a separate output directory:

```bash
uv run python3 - <<'PYTHON'
from pathlib import Path
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.analyze_all_regime_nas_dissertation import analyze_all_regime_nas_results

output = Path("studies/chapter06_mobilenet_nas/results/reproduced_analysis")
for path in analyze_all_regime_nas_results(output_dir=output):
    print(path)
PYTHON
```

For the exact data placement and cache-generation requirements, read [data/README.md](data/README.md). Training and channel-dependent outage evaluation require those external files. Hardware benchmarking additionally requires the physical target. Analysis from retained numerical outputs can be performed separately from retraining.

The [chapter guides](#studies-and-chapter-map) identify the applicable scripts and protocols. Study scripts retain their configuration constants. Inspect those constants before launching expensive training or Monte Carlo campaigns. Original results and selected binaries are retained alongside their configurations, while intermediate checkpoints, TensorBoard traces and channel realizations are excluded.

## Citation

Use GitHub's **Cite this repository** action or [CITATION.cff](CITATION.cff). Cite the dissertation separately when discussing its scientific findings. No DOI or release date is claimed for this repository.

## Contributing

> [!IMPORTANT]
> Read [CONTRIBUTING.md](CONTRIBUTING.md). Preserve the reported study protocols and provenance, and keep channel datasets and intermediate training artifacts out of submissions.

1. Fork the repository.
2. Create a branch for your focused change.
3. Commit your changes with a descriptive message after running the relevant checks.
4. Push your branch to your fork.
5. Open a pull request describing the change and verification.

## License

Original software is licensed under the **[Apache License 2.0](LICENSE.txt)**. Original explanatory documentation and figures use **[Creative Commons Attribution 4.0 International](LICENSE-DOCS.txt)**. Third-party licenses and attribution remain in their respective source files, including the MIT-licensed MATLAB generator. Code examples follow the software license.

## Authors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MatheusFS-dev"><img src="https://avatars.githubusercontent.com/u/99222557?v=4" height="100" alt="Matheus Ferreira Silva"><br><sub><b>Matheus Ferreira Silva</b></sub></a>
    </td>
    <td align="center">
      <img src="assets/authors/pedro-pereira.svg" height="100" alt="Photo placeholder for Pedro Marcio Raposo Pereira"><br><sub><b>Pedro Marcio Raposo Pereira</b></sub>
    </td>
    <td align="center">
      <img src="assets/authors/felipe-figueiredo.svg" height="100" alt="Photo placeholder for Felipe Augusto Pereira de Figueiredo"><br><sub><b>Felipe Augusto Pereira de Figueiredo</b></sub>
    </td>
    <td align="center">
      <img src="assets/authors/rausley-souza.jpeg" height="100" alt="Photo of Rausley Adriano Amaral de Souza"><br><sub><b>Rausley Adriano Amaral de Souza</b></sub>
    </td>
  </tr>
</table>
