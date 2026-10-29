# Reproduction and provenance

## Evidence sources

Chapter 5 includes the original CNN study code and artifacts. Current code and result artifacts support Chapters 4 and 6–8 and the later Chapter 5 spatial diagnostic. Standalone figure assets are extracted from the current dissertation worktree. The manuscript itself is excluded.

`source-manifest.json` records each original source path, source revision, source SHA-256, destination path and destination SHA-256. When source content differs from its recorded commit, the checksum identifies the working-copy content, not the commit alone. Notebook outputs are cleared and executable paths/imports are adapted. Numerical result files and selected binaries retain their source bytes.

## Retained artifacts

Retain reported numerical tables, configuration and preprocessing artifacts, selected final CNN/MobileNet models, rank-1 NAS artifacts, the search histories used in dissertation analyses, and final firmware images. Nonwinning NAS model binaries and intermediate checkpoints are excluded. The original result-directory nesting is retained inside each chapter to keep seed, budget, regime and artifact relationships explicit.

## Boundaries

Channel datasets and float32 runtime caches are absent. Regeneration requires their separately supplied inputs or MATLAB generation. Fresh random realizations need not match original sample identities. The Chapter 7 raw `runs/dataset_analysis/` tree was absent at extraction, so available figure/table artifacts do not stand in for original Monte Carlo repetitions.

Historical notebook dependencies are recorded separately. The extraction does not claim a rerun of training, a new physical power measurement, or a new embedded deployment. See `verification.md` for checks actually performed.
