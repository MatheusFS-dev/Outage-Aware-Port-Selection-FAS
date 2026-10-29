"""Run per-regime Monte Carlo simulations for RND policies on the KMS dataset.

Evaluates RND 1x, 3x, and 10x policies.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import sys
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rnd_kms_per_regime_mc.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import (
    COMMON_DATASET_PATH,
    DATA_SEED,
    KMS_REGIME_LABELS,
    MC_EVAL_FRACTION,
    NUM_MC_REPS,
    RESULT_FIELDNAMES,
    RUNS_ROOT,
    SNR_LINEAR,
    TARGET_PORTS,
    THRESHOLD,
    TOTAL_PORTS,
    build_rep_rng,
    compute_wilson_interval,
    ensure_directory,
    ensure_matching_run_config,
    evaluate_ideal_ports,
    format_index_sets_zero_based,
    format_threshold_folder,
    list_completed_repetitions,
    load_kms_regime,
    log_status,
    repetition_result_path,
    simulate_rnd_policy_result,
    split_design_eval_indices,
    threshold_to_binary,
    write_csv_rows_atomic,
)


RUN_NAME = "policy_rnd_kms_per_regime_mc"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
RND_N_TIMES = (1, 3, 10)


def build_rnd_run_config(threshold: float, output_dir: Path) -> dict[str, Any]:
    return {
        "script_name": _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rnd_kms_per_regime_mc.py'.stem,
        "policy_name": "rnd",
        "common_dataset_path": str(COMMON_DATASET_PATH),
        "output_dir": str(output_dir),
        "data_seed": int(DATA_SEED),
        "threshold": float(threshold),
        "snr_linear": float(SNR_LINEAR),
        "total_ports": int(TOTAL_PORTS),
        "target_ports": [int(value) for value in TARGET_PORTS],
        "rnd_n_times": [int(value) for value in RND_N_TIMES],
        "num_mc_reps": int(NUM_MC_REPS),
        "mc_eval_fraction": float(MC_EVAL_FRACTION),
        "regime_labels": list(KMS_REGIME_LABELS),
    }


def run_configured_rnd_simulation() -> dict[str, Any]:
    script_name = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rnd_kms_per_regime_mc.py'.stem
    threshold_output_dirs: list[str] = []

    for threshold in THRESHOLD:
        threshold_output_dir = ensure_directory(OUTPUT_DIR / format_threshold_folder(float(threshold)))
        threshold_output_dirs.append(str(threshold_output_dir))
        ensure_matching_run_config(
            threshold_output_dir,
            build_rnd_run_config(threshold=float(threshold), output_dir=threshold_output_dir),
        )

        for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
            regime_output_dir = ensure_directory(threshold_output_dir / regime_label)
            log_status(
                script_name,
                "regime",
                f"Loading threshold {float(threshold):.8f}, regime {regime_index + 1}/14: {regime_label}",
            )
            regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)

            completed_reps = list_completed_repetitions(regime_output_dir)
            for rep_index in range(NUM_MC_REPS):
                if rep_index in completed_reps:
                    log_status(script_name, "resume", f"Skipping {regime_label} repetition {rep_index}: already done")
                    continue

                rng = build_rep_rng(base_seed=DATA_SEED, regime_index=regime_index, rep_index=rep_index)
                design_indices, eval_indices = split_design_eval_indices(
                    n_samples=regime_data.shape[0],
                    eval_fraction=MC_EVAL_FRACTION,
                    rng=rng,
                )

                eval_values = regime_data[eval_indices]
                eval_binary = threshold_to_binary(eval_values, threshold=float(threshold), snr_linear=SNR_LINEAR)
                ideal_op, n_ideal_outage = evaluate_ideal_ports(eval_binary)

                rep_rows: list[dict[str, Any]] = []
                for n_times in RND_N_TIMES:
                    for n_ports in TARGET_PORTS:
                        observed_op, n_eval, n_outage, selected_draws = simulate_rnd_policy_result(
                            eval_binary,
                            n_ports=int(n_ports),
                            n_times=int(n_times),
                            rng=rng,
                        )
                        wilson_low, wilson_high = compute_wilson_interval(n_outage=n_outage, n_total=n_eval)
                        sorted_draws = [np.sort(selected) for selected in selected_draws]

                        rep_rows.append({
                            "policy": f"rnd_{n_times}x",
                            "dataset": regime_label,
                            "regime_index": int(regime_index),
                            "rep": int(rep_index),
                            "base_seed": int(DATA_SEED),
                            "n_ports": int(n_ports),
                            "design_size": int(design_indices.shape[0]),
                            "eval_size": int(eval_indices.shape[0]),
                            "selection_order_0based": format_index_sets_zero_based(selected_draws),
                            "sorted_pattern_0based": format_index_sets_zero_based(sorted_draws),
                            "selection_order_1based": "N/A",
                            "sorted_pattern_1based": "N/A",
                            "observed_op": f"{observed_op:.10f}",
                            "ideal_op": f"{ideal_op:.10f}",
                            "gap_to_ideal": f"{(observed_op - ideal_op):.10f}",
                            "n_eval": int(n_eval),
                            "n_outage": int(n_outage),
                            "n_ideal_outage": int(n_ideal_outage),
                            "wilson_low": f"{wilson_low:.10f}",
                            "wilson_high": f"{wilson_high:.10f}",
                        })

                write_csv_rows_atomic(repetition_result_path(regime_output_dir, rep_index), RESULT_FIELDNAMES, rep_rows)
                log_status(script_name, "rep", f"Saved {regime_label} repetition {rep_index} results")

            del regime_data

    return {
        "output_dir": str(OUTPUT_DIR),
        "threshold_output_dirs": threshold_output_dirs,
        "num_mc_reps": int(NUM_MC_REPS),
        "target_ports": tuple(int(value) for value in TARGET_PORTS),
    }

if __name__ == "__main__":
    run_configured_rnd_simulation()
