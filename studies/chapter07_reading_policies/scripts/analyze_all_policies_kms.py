"""Analyze all reading-pattern Monte Carlo results.

Reads saved repetition files from each policy run, reconstructs matching
evaluation splits, and writes comparison tables across regimes.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/analyze_all_policies_kms.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import (
    aggregate_result_rows_from_run,
    compute_wilson_interval,
    ensure_directory,
    ensure_matching_run_config,
    format_threshold_folder,
    KMS_REGIME_LABELS,
    list_completed_repetitions,
    log_status,
    read_csv_rows,
    read_json_file,
    RUNS_ROOT,
    THRESHOLD,
    write_csv_rows_atomic,
    write_json_atomic,
)


UNI_RUN_DIR = RUNS_ROOT / "policy_uni_kms_per_regime_mc"
MAR_RUN_DIR = RUNS_ROOT / "policy_mar_kms_per_regime_mc"
GC_RUN_DIR = RUNS_ROOT / "policy_gc_kms_per_regime_mc"
SUNI_RUN_DIR = RUNS_ROOT / "policy_suni_kms_per_regime_mc"
SWUNI_RUN_DIR = RUNS_ROOT / "policy_swuni_kms_per_regime_mc"
RAG_RUN_DIR = RUNS_ROOT / "policy_rag_kms_per_regime_mc"
RND_RUN_DIR = RUNS_ROOT / "policy_rnd_kms_per_regime_mc"
DETERMINISTIC_POLICY_RUNS: tuple[tuple[str, Path], ...] = (
    ("uni", UNI_RUN_DIR),
    ("mar", MAR_RUN_DIR),
    ("gc", GC_RUN_DIR),
    ("suni", SUNI_RUN_DIR),
    ("swuni", SWUNI_RUN_DIR),
    ("rag", RAG_RUN_DIR),
)
ANALYSIS_RUN_NAME = "analysis_all_policies_kms"
ANALYSIS_OUTPUT_DIR = RUNS_ROOT / ANALYSIS_RUN_NAME
FORCE_REBUILD_ANALYSIS = True
ALLOW_PARTIAL_POLICY_RUNS = False

COMPARISON_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "rep",
    "policy",
    "n_ports",
    "selection_order_0based",
    "sorted_pattern_0based",
    "selection_order_1based",
    "sorted_pattern_1based",
    "observed_op",
    "ideal_op",
    "gain_vs_uni",
    "delta_vs_uni",
    "gap_to_ideal",
    "n_eval",
    "n_outage",
    "n_ideal_outage",
    "wilson_low",
    "wilson_high",
)
SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "policy",
    "n_ports",
    "n_reps",
    "mean_observed_op",
    "std_observed_op",
    "ci95_half_width",
    "pooled_observed_op",
    "pooled_wilson_low",
    "pooled_wilson_high",
    "mean_ideal_op",
    "mean_gain_vs_uni",
    "mean_delta_vs_uni",
    "delta_ci95_half_width",
    "mean_gap_to_ideal",
)
BEST_POLICY_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "regime_index",
    "n_ports",
    "best_policy",
    "best_mean_observed_op",
)


def validate_policy_run_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    if not configs:
        raise ValueError("No configs provided")
    keys_to_match = (
        "common_dataset_path",
        "data_seed",
        "threshold",
        "snr_linear",
        "total_ports",
        "target_ports",
        "num_mc_reps",
        "mc_eval_fraction",
        "regime_labels",
    )
    base_config = configs[0]
    for config in configs[1:]:
        for key in keys_to_match:
            if base_config.get(key) != config.get(key):
                raise ValueError(f"Configurations differ for key '{key}'.")

    return {
        "common_dataset_path": str(base_config["common_dataset_path"]),
        "data_seed": int(base_config["data_seed"]),
        "threshold": float(base_config["threshold"]),
        "snr_linear": float(base_config["snr_linear"]),
        "total_ports": int(base_config["total_ports"]),
        "target_ports": tuple(int(value) for value in base_config["target_ports"]),
        "num_mc_reps": int(base_config["num_mc_reps"]),
        "mc_eval_fraction": float(base_config["mc_eval_fraction"]),
        "regime_labels": tuple(str(value) for value in base_config["regime_labels"]),
    }


def build_analysis_run_config(
    shared_config: dict[str, Any],
    *,
    deterministic_run_dirs: dict[str, Path],
    rnd_run_dir: Path,
    analysis_output_dir: Path,
) -> dict[str, Any]:
    return {
        "script_name": _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/analyze_all_policies_kms.py'.stem,
        "deterministic_run_dirs": {name: str(path) for name, path in deterministic_run_dirs.items()},
        "rnd_run_dir": str(rnd_run_dir),
        "analysis_output_dir": str(analysis_output_dir),
        "force_rebuild_analysis": bool(FORCE_REBUILD_ANALYSIS),
        "allow_partial_policy_runs": bool(ALLOW_PARTIAL_POLICY_RUNS),
        **shared_config,
    }


def index_policy_rows_by_rep_and_budget(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    index: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["rep"]), int(row["n_ports"]))
        if key in index:
            raise ValueError(f"Duplicate policy row detected for key {key}.")
        index[key] = row
    return index


def normalize_policy_row(
    source_row: dict[str, str],
    *,
    gain_vs_uni: float,
    delta_vs_uni: float,
) -> dict[str, Any]:
    observed_op = float(source_row["observed_op"])
    ideal_op = float(source_row["ideal_op"])
    return {
        "dataset": str(source_row["dataset"]),
        "regime_index": int(source_row["regime_index"]),
        "rep": int(source_row["rep"]),
        "policy": str(source_row["policy"]),
        "n_ports": int(source_row["n_ports"]),
        "selection_order_0based": str(source_row["selection_order_0based"]),
        "sorted_pattern_0based": str(source_row["sorted_pattern_0based"]),
        "selection_order_1based": str(source_row["selection_order_1based"]),
        "sorted_pattern_1based": str(source_row["sorted_pattern_1based"]),
        "observed_op": observed_op,
        "ideal_op": ideal_op,
        "gain_vs_uni": float(gain_vs_uni),
        "delta_vs_uni": float(delta_vs_uni),
        "gap_to_ideal": float(observed_op - ideal_op),
        "n_eval": int(source_row["n_eval"]),
        "n_outage": int(source_row["n_outage"]),
        "n_ideal_outage": int(source_row["n_ideal_outage"]),
        "wilson_low": float(source_row["wilson_low"]),
        "wilson_high": float(source_row["wilson_high"]),
    }


def summarize_comparison_rows(
    comparison_rows: list[dict[str, Any]],
    regime_label: str,
    regime_index: int,
) -> list[dict[str, Any]]:
    if len(comparison_rows) == 0:
        raise ValueError("comparison_rows must not be empty.")

    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in comparison_rows:
        key = (str(row["policy"]), int(row["n_ports"]))
        grouped_rows.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (policy_name, n_ports), rows in sorted(grouped_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        observed_values = np.array([float(row["observed_op"]) for row in rows], dtype=np.float64)
        ideal_values = np.array([float(row["ideal_op"]) for row in rows], dtype=np.float64)
        gains = np.array([float(row["gain_vs_uni"]) for row in rows], dtype=np.float64)
        deltas = np.array([float(row["delta_vs_uni"]) for row in rows], dtype=np.float64)
        gap_values = np.array([float(row["gap_to_ideal"]) for row in rows], dtype=np.float64)
        outage_counts = np.array([int(row["n_outage"]) for row in rows], dtype=np.int64)
        eval_counts = np.array([int(row["n_eval"]) for row in rows], dtype=np.int64)
        pooled_n_outage = int(np.sum(outage_counts))
        pooled_n_eval = int(np.sum(eval_counts))
        pooled_wilson_low, pooled_wilson_high = compute_wilson_interval(
            n_outage=pooled_n_outage,
            n_total=pooled_n_eval,
        )

        std_observed = float(np.std(observed_values, ddof=1)) if len(observed_values) > 1 else 0.0
        ci95_half_width = float(1.96 * std_observed / math.sqrt(len(observed_values)))
        delta_std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        delta_ci95_half_width = float(1.96 * delta_std / math.sqrt(len(deltas)))
        summary_rows.append(
            {
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "policy": policy_name,
                "n_ports": int(n_ports),
                "n_reps": int(len(rows)),
                "mean_observed_op": float(np.mean(observed_values)),
                "std_observed_op": std_observed,
                "ci95_half_width": ci95_half_width,
                "pooled_observed_op": float(pooled_n_outage / pooled_n_eval),
                "pooled_wilson_low": float(pooled_wilson_low),
                "pooled_wilson_high": float(pooled_wilson_high),
                "mean_ideal_op": float(np.mean(ideal_values)),
                "mean_gain_vs_uni": float(np.mean(gains)),
                "mean_delta_vs_uni": float(np.mean(deltas)),
                "delta_ci95_half_width": delta_ci95_half_width,
                "mean_gap_to_ideal": float(np.mean(gap_values)),
            }
        )
    return summary_rows


def write_regime_summary_text(
    regime_output_dir: Path,
    regime_label: str,
    summary_rows: list[dict[str, Any]],
) -> Path:
    best_by_budget: dict[int, tuple[str, float]] = {}
    for row in summary_rows:
        n_ports = int(row["n_ports"])
        candidate = (str(row["policy"]), float(row["mean_observed_op"]))
        if n_ports not in best_by_budget or candidate[1] < best_by_budget[n_ports][1]:
            best_by_budget[n_ports] = candidate

    lines = [f"Regime: {regime_label}", ""]
    lines.append("Best policy by budget:")
    for n_ports in sorted(best_by_budget):
        policy_name, mean_observed_op = best_by_budget[n_ports]
        lines.append(f"- N={n_ports}: {policy_name} with mean observed OP={mean_observed_op:.6f}")
    output_path = regime_output_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_cross_regime_summary_text(
    output_dir: Path,
    cross_regime_summary_rows: list[dict[str, Any]],
    skipped_regimes: list[str],
    target_ports: tuple[int, ...],
) -> Path:
    lines = ["Cross-regime summary", ""]
    for n_ports in target_ports:
        candidate_rows = [row for row in cross_regime_summary_rows if int(row["n_ports"]) == n_ports]
        if len(candidate_rows) == 0:
            continue
        mean_by_policy: dict[str, float] = {}
        all_policy_names = sorted({str(row["policy"]) for row in candidate_rows})
        for policy_name in all_policy_names:
            policy_rows = [row for row in candidate_rows if str(row["policy"]) == policy_name]
            if len(policy_rows) == 0:
                continue
            mean_by_policy[policy_name] = float(
                np.mean([float(row["mean_observed_op"]) for row in policy_rows], dtype=np.float64)
            )
        if len(mean_by_policy) == 0:
            continue
        best_policy = min(mean_by_policy, key=mean_by_policy.get)
        lines.append(f"- N={n_ports}: best mean policy across processed regimes is {best_policy}")
        for policy_name in all_policy_names:
            if policy_name in mean_by_policy:
                lines.append(f"  {policy_name}: {mean_by_policy[policy_name]:.6f}")
        lines.append("")

    if skipped_regimes:
        lines.append("Skipped regimes:")
        for regime_label in skipped_regimes:
            lines.append(f"- {regime_label}")
    output_path = output_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_uninalysis_for_threshold(threshold: float) -> dict[str, Any]:
    """Analyze the policy result folders for one threshold value.

    Args:
        threshold (float): Outage threshold whose policy result folders should
            be compared. The value is converted to the canonical
            ``threshold_<value>`` folder name and must already exist under each
            policy run root.

    Returns:
        dict[str, Any]: Analysis summary containing the threshold-specific
            output directory, processed regimes, and skipped regimes.

    Raises:
        FileNotFoundError: If any threshold-specific policy run directory is
            missing.
        ValueError: If policy run configurations differ or output config
            validation fails.
        OSError: If analysis outputs cannot be written.
    """
    script_name = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/analyze_all_policies_kms.py'.stem
    threshold_folder = format_threshold_folder(float(threshold))
    deterministic_run_dirs = {
        policy_name: run_root / threshold_folder for policy_name, run_root in DETERMINISTIC_POLICY_RUNS
    }
    rnd_run_dir = RND_RUN_DIR / threshold_folder
    analysis_output_dir = ANALYSIS_OUTPUT_DIR / threshold_folder
    run_dirs = [*deterministic_run_dirs.values(), rnd_run_dir]
    
    for d in run_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Policy run directory not found: {d}")

    configs = [read_json_file(d / "run_config.json") for d in run_dirs]
    shared_config = validate_policy_run_configs(configs)

    analysis_output_dir = ensure_directory(analysis_output_dir)
    ensure_matching_run_config(
        analysis_output_dir,
        build_analysis_run_config(
            shared_config,
            deterministic_run_dirs=deterministic_run_dirs,
            rnd_run_dir=rnd_run_dir,
            analysis_output_dir=analysis_output_dir,
        ),
        ignore_keys=frozenset({
            "common_dataset_path",
            "analysis_output_dir",
            "deterministic_run_dirs",
            "uni_run_dir",
            "mar_run_dir",
            "gc_run_dir",
            "rnd_run_dir",
        }),
    )

    cross_regime_summary_rows: list[dict[str, Any]] = []
    best_policy_rows: list[dict[str, Any]] = []
    processed_regimes: list[str] = []
    skipped_regimes: list[str] = []

    for regime_index, regime_label in enumerate(shared_config["regime_labels"]):
        regime_output_dir = ensure_directory(analysis_output_dir / regime_label)
        combined_csv_path = regime_output_dir / "combined_policy_rows.csv"
        summary_csv_path = regime_output_dir / "summary_by_budget.csv"

        if not FORCE_REBUILD_ANALYSIS and combined_csv_path.exists() and summary_csv_path.exists():
            log_status(script_name, "resume", f"Skipping {regime_label}: analysis files already exist")
            existing_summary_rows = []
            for row in read_csv_rows(summary_csv_path):
                existing_summary_rows.append(
                    {
                        "dataset": str(row["dataset"]),
                        "regime_index": int(row["regime_index"]),
                        "policy": str(row["policy"]),
                        "n_ports": int(row["n_ports"]),
                        "n_reps": int(row["n_reps"]),
                        "mean_observed_op": float(row["mean_observed_op"]),
                        "std_observed_op": float(row["std_observed_op"]),
                        "ci95_half_width": float(row["ci95_half_width"]),
                        "pooled_observed_op": float(row["pooled_observed_op"]),
                        "pooled_wilson_low": float(row["pooled_wilson_low"]),
                        "pooled_wilson_high": float(row["pooled_wilson_high"]),
                        "mean_ideal_op": float(row["mean_ideal_op"]),
                        "mean_gain_vs_uni": float(row["mean_gain_vs_uni"]),
                        "mean_delta_vs_uni": float(row["mean_delta_vs_uni"]),
                        "delta_ci95_half_width": float(row["delta_ci95_half_width"]),
                        "mean_gap_to_ideal": float(row["mean_gap_to_ideal"]),
                    }
                )
            processed_regimes.append(regime_label)
            cross_regime_summary_rows.extend(existing_summary_rows)
            for n_ports in shared_config["target_ports"]:
                candidate_rows = [row for row in existing_summary_rows if int(row["n_ports"]) == int(n_ports)]
                if len(candidate_rows) == 0:
                    continue
                best_row = min(candidate_rows, key=lambda item: float(item["mean_observed_op"]))
                best_policy_rows.append(
                    {
                        "dataset": regime_label,
                        "regime_index": int(regime_index),
                        "n_ports": int(n_ports),
                        "best_policy": str(best_row["policy"]),
                        "best_mean_observed_op": float(best_row["mean_observed_op"]),
                    }
                )
            continue

        completed_by_policy = {
            policy_name: list_completed_repetitions(run_dir / regime_label)
            for policy_name, run_dir in deterministic_run_dirs.items()
        }
        rnd_reps = list_completed_repetitions(rnd_run_dir / regime_label)
        common_reps_set = set(rnd_reps)
        for reps in completed_by_policy.values():
            common_reps_set = common_reps_set.intersection(reps)
        common_reps = sorted(common_reps_set)
        
        if not ALLOW_PARTIAL_POLICY_RUNS and len(common_reps) != shared_config["num_mc_reps"]:
            log_status(
                script_name,
                "skip",
                (
                    f"Skipping {regime_label}: found {len(common_reps)} common repetitions, "
                    f"expected {shared_config['num_mc_reps']}"
                ),
            )
            skipped_regimes.append(regime_label)
            continue
        if len(common_reps) == 0:
            log_status(script_name, "skip", f"Skipping {regime_label}: no common repetitions found")
            skipped_regimes.append(regime_label)
            continue

        log_status(script_name, "regime", f"Analyzing regime {regime_index + 1}/14: {regime_label}")
        
        deterministic_rows = {
            policy_name: aggregate_result_rows_from_run(run_dir, regime_label)
            for policy_name, run_dir in deterministic_run_dirs.items()
        }
        rnd_rows = aggregate_result_rows_from_run(rnd_run_dir, regime_label)
        
        deterministic_indices = {
            policy_name: index_policy_rows_by_rep_and_budget(rows)
            for policy_name, rows in deterministic_rows.items()
        }

        # RND outputs rows differently (f"rnd_{n_times}x"), so group by rep, n_ports, policy
        rnd_index: dict[tuple[int, int, str], dict[str, str]] = {}
        for row in rnd_rows:
            rnd_index[(int(row["rep"]), int(row["n_ports"]), str(row["policy"]))] = row

        comparison_rows: list[dict[str, Any]] = []
        for rep_index in common_reps:
            for n_ports in shared_config["target_ports"]:
                row_uni = deterministic_indices["uni"][(rep_index, int(n_ports))]
                observed_uni = float(row_uni["observed_op"])

                for policy_name, _ in DETERMINISTIC_POLICY_RUNS:
                    policy_row = deterministic_indices[policy_name][(rep_index, int(n_ports))]
                    observed_policy = float(policy_row["observed_op"])
                    comparison_rows.append(
                        normalize_policy_row(
                            policy_row,
                            gain_vs_uni=observed_uni - observed_policy,
                            delta_vs_uni=observed_policy - observed_uni,
                        )
                    )
                
                for n_times in configs[-1]["rnd_n_times"]:
                    policy_name = f"rnd_{n_times}x"
                    row_r = rnd_index[(rep_index, int(n_ports), policy_name)]
                    observed_r = float(row_r["observed_op"])
                    comparison_rows.append(
                        normalize_policy_row(
                            row_r,
                            gain_vs_uni=observed_uni - observed_r,
                            delta_vs_uni=observed_r - observed_uni,
                        )
                    )

        summary_rows = summarize_comparison_rows(
            comparison_rows=comparison_rows,
            regime_label=regime_label,
            regime_index=regime_index,
        )
        write_csv_rows_atomic(combined_csv_path, COMPARISON_FIELDNAMES, comparison_rows)
        write_csv_rows_atomic(summary_csv_path, SUMMARY_FIELDNAMES, summary_rows)
        write_regime_summary_text(regime_output_dir=regime_output_dir, regime_label=regime_label, summary_rows=summary_rows)
        processed_regimes.append(regime_label)
        cross_regime_summary_rows.extend(summary_rows)

        for n_ports in shared_config["target_ports"]:
            candidate_rows = [row for row in summary_rows if int(row["n_ports"]) == int(n_ports)]
            best_row = min(candidate_rows, key=lambda item: float(item["mean_observed_op"]))
            best_policy_rows.append(
                {
                    "dataset": regime_label,
                    "regime_index": int(regime_index),
                    "n_ports": int(n_ports),
                    "best_policy": str(best_row["policy"]),
                    "best_mean_observed_op": float(best_row["mean_observed_op"]),
                }
            )

    write_csv_rows_atomic(analysis_output_dir / "cross_regime_summary.csv", SUMMARY_FIELDNAMES, cross_regime_summary_rows)
    write_csv_rows_atomic(analysis_output_dir / "best_mary_regime_and_budget.csv", BEST_POLICY_FIELDNAMES, best_policy_rows)
    write_cross_regime_summary_text(
        output_dir=analysis_output_dir,
        cross_regime_summary_rows=cross_regime_summary_rows,
        skipped_regimes=skipped_regimes,
        target_ports=shared_config["target_ports"],
    )
    write_json_atomic(
        analysis_output_dir / "analysis_status.json",
        {
            "processed_regimes": processed_regimes,
            "skipped_regimes": skipped_regimes,
        },
    )

    return {
        "output_dir": str(analysis_output_dir),
        "threshold": float(threshold),
        "processed_regimes": processed_regimes,
        "skipped_regimes": skipped_regimes,
    }


def run_configured_uninalysis() -> dict[str, Any]:
    """Analyze all configured threshold-specific policy result folders.

    Args:
        None: The analysis uses module-level constants, including
            ``THRESHOLD`` for the threshold list and the four policy run roots.

    Returns:
        dict[str, Any]: Summary containing one threshold-specific analysis
            result per configured threshold.

    Raises:
        FileNotFoundError: If any expected threshold-specific policy result
            folder is missing.
        ValueError: If configurations differ across policy runs.
        OSError: If output files cannot be written.
    """
    threshold_results = []
    for threshold in THRESHOLD:
        threshold_results.append(run_uninalysis_for_threshold(float(threshold)))

    return {
        "output_dir": str(ANALYSIS_OUTPUT_DIR),
        "threshold_results": threshold_results,
    }


if __name__ == "__main__":
    run_configured_uninalysis()
