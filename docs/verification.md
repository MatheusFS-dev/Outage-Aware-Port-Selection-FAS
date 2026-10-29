# Verification of the extraction

Verified on 2026-10-29 using Python 3.12 and the companion repository's uv environment.

- `uv lock --check` passed and `uv sync --extra test` installed the independent environment.
- `uv run --extra test python3 -m pytest -q`: **38 passed, 5 subtests passed**.
- Rebuilt and compared **16** all-regime NAS, per-regime NAS and learned-policy numerical summaries against retained CSVs, with relative tolerance 1e-8 and absolute tolerance 1e-10.
- Loaded a historical seven-port CNN, a selected NAS MobileNet, and a final learned-policy model. The CNN has 605,220 parameters and the sampled NAS and learned-policy models each have 15,756 parameters. A selected TFLite model allocated successfully.
- Firmware automation discovers all **70** selected per-regime TFLite models without running a build or touching a device.
- `cffconvert --validate` accepted `CITATION.cff` under schema 1.2.0.
- Parsed all retained study/shared Python sources, verified README links and standalone figure input dependencies, checked selected-artifact ignore rules, and verified copied artifact hashes. Numerical tables, model binaries and databases retain their source checksums.
- Compiled the updated dissertation with `latexmk -pdf -interaction=nonstopmode -halt-on-error` in an isolated output directory. The build completed without undefined references. Appendix C's rendered organization and URL were inspected.

## Review and adaptations

An independent review identified obsolete firmware roots. These were fixed, and a regression test demonstrated failure before the fix and passed afterward. Firmware campaign execution now uses main guards so importing the automation does not start builds or access devices. Firmware documentation paths were updated.

Three original visual-contract tests also failed against the original source checkout because they expected superseded radar renderers/ticks or old fixed axis limits. Companion tests were updated to verify the currently retained figures and data bounds. Plotting behavior and numerical results were preserved.

## Limits

No datasets, full training campaigns, MATLAB simulations, or physical measurements were run. The Pico SDK and pico-tflmicro environment variables are not configured, so a firmware compilation was not claimed. Historical CNN training also requires the external utilities documented in its chapter guide. Chapter 7 raw Monte Carlo repetitions were unavailable in the source checkout.

The retained study tree is approximately 4.9 GB, including final baseline models and about 1.9 GB of original Optuna databases. These search histories are retained because the analyses consume them. No artifact was committed or pushed.
