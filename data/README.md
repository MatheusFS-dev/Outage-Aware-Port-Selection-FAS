# External channel data

Channel datasets are intentionally not distributed due to size. Training and outage recomputation require locally generated or separately supplied data via contact with the authors.

Place the 14 MATLAB `SNR_events_*.mat` files under `data/kappa-mu_shadowed/`. The current Python loader expects the `SNR_events` variable and the regime filenames defined in `src/utils/data.py`. Generate the optional float32 cache with Chapter 4's `generate_kms_runtime_cache.py`, which writes to `data/kms_runtime_cache_float32/`.

The MATLAB generator and its required functions are in `studies/chapter04_system_model/scripts/matlab/`. Inspect its output directory and parameters before running it. The historical Chapter 5 notebooks also read from `data/kappa-mu_shadowed/` after the documented path adaptation.

Dataset generation requires MATLAB. It can consume substantial memory, storage, and time. New generated realizations are not guaranteed to reproduce the original sample identities or numerical results. Retained result tables can be inspected without these datasets.

## MATLAB generator formats

`studies/chapter04_system_model/scripts/matlab/legacy_dataset/function_main_SINR_onlyPorts_kappaMu.m` writes sample-by-port `SNR_events` matrices with the `SNR_events_W...mat` filenames consumed by Python. Its caller, `generateDatasets.m`, is preserved with its existing example parameter values. Configure the dissertation's regime list, unit aperture, 100 ports and one million realizations per regime before generation. Its `parfor` loop requires MATLAB parallel support. Outputs are written beneath the MATLAB working directory's `runs/` directory and must be placed in the external data directory described above.

The retained v1 `gen_data.m` is a separate envelope-generation implementation. It writes `R_simulated` and `FAS_...mat` filenames, which are not direct inputs to the Python loader. Do not substitute or rename those files as `SNR_events` datasets. MATLAB generation has not been executed during this extraction.
