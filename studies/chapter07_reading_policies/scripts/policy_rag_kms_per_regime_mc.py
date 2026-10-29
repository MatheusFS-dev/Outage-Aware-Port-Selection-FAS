"""Run per-regime Monte Carlo simulations for RAG on the KMS dataset.

RAG is a redundancy-aware version of greedy coverage that penalizes overlap
with already selected ports.
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


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rag_kms_per_regime_mc.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import (
    CANONICAL_FIELDNAMES,
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
    build_canonical_rows,
    build_mar_order,
    build_rag_order,
    build_rep_rng,
    build_result_rows,
    ensure_directory,
    ensure_matching_run_config,
    format_threshold_folder,
    list_completed_repetitions,
    load_kms_regime,
    log_status,
    repetition_result_path,
    split_design_eval_indices,
    threshold_to_binary,
    write_csv_rows_atomic,
)


RUN_NAME = "policy_rag_kms_per_regime_mc"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
# Reasonable lambda values to sweep later are 0.01, 0.05, 0.1, and 0.2.
RAG_REDUNDANCY_WEIGHT = 0.1


def build_rag_run_config(threshold: float, output_dir: Path) -> dict[str, Any]:
    """Build the persisted configuration for the RAG simulation run.

    Args:
        threshold (float): Linear outage threshold evaluated by this run
            directory.
        output_dir (Path): Threshold-specific directory where artifacts are
            written.

    Returns:
        dict[str, Any]: JSON-serializable configuration used to validate
            resumable reruns.

    Raises:
        ValueError: If ``TARGET_PORTS`` is empty.
    """
    if len(TARGET_PORTS) == 0:
        raise ValueError("TARGET_PORTS must contain at least one observation budget.")

    return {
        "script_name": _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rag_kms_per_regime_mc.py'.stem,
        "policy_name": "rag",
        "common_dataset_path": str(COMMON_DATASET_PATH),
        "output_dir": str(output_dir),
        "data_seed": int(DATA_SEED),
        "threshold": float(threshold),
        "snr_linear": float(SNR_LINEAR),
        "total_ports": int(TOTAL_PORTS),
        "target_ports": [int(value) for value in TARGET_PORTS],
        "rag_redundancy_weight": float(RAG_REDUNDANCY_WEIGHT),
        "num_mc_reps": int(NUM_MC_REPS),
        "mc_eval_fraction": float(MC_EVAL_FRACTION),
        "regime_labels": list(KMS_REGIME_LABELS),
    }


def run_configured_rag_simulation() -> dict[str, Any]:
    """Run the per-regime Monte Carlo experiment for RAG.

    Args:
        None: The simulation is controlled by module-level constants imported
            from ``policy_mc_kms_common``.

    Returns:
        dict[str, Any]: Run summary with output directory, threshold
            directories, repetition count, and target budgets.

    Raises:
        ValueError: If loaded data width differs from ``TOTAL_PORTS`` or an
            existing run configuration is incompatible.
        FileNotFoundError: If an expected KMS regime file is missing.
        OSError: If output artifacts cannot be written.
    """
    script_name = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_rag_kms_per_regime_mc.py'.stem
    threshold_output_dirs: list[str] = []

    for threshold in THRESHOLD:
        threshold_output_dir = ensure_directory(OUTPUT_DIR / format_threshold_folder(float(threshold)))
        threshold_output_dirs.append(str(threshold_output_dir))
        ensure_matching_run_config(
            threshold_output_dir,
            build_rag_run_config(threshold=float(threshold), output_dir=threshold_output_dir),
        )

        for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
            regime_output_dir = ensure_directory(threshold_output_dir / regime_label)
            log_status(
                script_name,
                "regime",
                f"Loading threshold {float(threshold):.8f}, regime {regime_index + 1}/14: {regime_label}",
            )
            regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)
            if regime_data.shape[1] != TOTAL_PORTS:
                raise ValueError(f"Regime {regime_label} has {regime_data.shape[1]} ports, expected {TOTAL_PORTS}.")

            full_binary = threshold_to_binary(regime_data, threshold=float(threshold), snr_linear=SNR_LINEAR)
            full_means = np.mean(regime_data, axis=0, dtype=np.float64)
            _, full_scores = build_mar_order(design_binary=full_binary, design_means=full_means)
            full_order = build_rag_order(
                design_binary=full_binary,
                design_means=full_means,
                mar_scores=full_scores,
                max_budget=TOTAL_PORTS,
                redundancy_weight=RAG_REDUNDANCY_WEIGHT,
            )
            write_csv_rows_atomic(
                regime_output_dir / "canonical_patterns.csv",
                fieldnames=CANONICAL_FIELDNAMES,
                rows=build_canonical_rows(
                    policy_name="rag",
                    regime_label=regime_label,
                    regime_index=regime_index,
                    full_order=full_order,
                    target_ports=TARGET_PORTS,
                ),
            )
            del full_binary, full_means, full_scores, full_order

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

                design_values = regime_data[design_indices]
                eval_values = regime_data[eval_indices]
                design_binary = threshold_to_binary(design_values, threshold=float(threshold), snr_linear=SNR_LINEAR)
                eval_binary = threshold_to_binary(eval_values, threshold=float(threshold), snr_linear=SNR_LINEAR)
                design_means = np.mean(design_values, axis=0, dtype=np.float64)
                _, mar_scores = build_mar_order(design_binary=design_binary, design_means=design_means)
                policy_order = build_rag_order(
                    design_binary=design_binary,
                    design_means=design_means,
                    mar_scores=mar_scores,
                    max_budget=max(TARGET_PORTS),
                    redundancy_weight=RAG_REDUNDANCY_WEIGHT,
                )

                rep_rows = build_result_rows(
                    policy_name="rag",
                    regime_label=regime_label,
                    regime_index=regime_index,
                    rep_index=rep_index,
                    base_seed=DATA_SEED,
                    design_size=int(design_indices.shape[0]),
                    eval_size=int(eval_indices.shape[0]),
                    full_order=policy_order,
                    target_ports=TARGET_PORTS,
                    binary_eval=eval_binary,
                )
                write_csv_rows_atomic(repetition_result_path(regime_output_dir, rep_index), RESULT_FIELDNAMES, rep_rows)
                log_status(script_name, "rep", f"Saved {regime_label} repetition {rep_index} results")
                del design_values, eval_values, design_binary, eval_binary, design_means, mar_scores, policy_order, rep_rows

            del regime_data

    return {
        "output_dir": str(OUTPUT_DIR),
        "threshold_output_dirs": threshold_output_dirs,
        "num_mc_reps": int(NUM_MC_REPS),
        "target_ports": tuple(int(value) for value in TARGET_PORTS),
    }


if __name__ == "__main__":
    run_configured_rag_simulation()
