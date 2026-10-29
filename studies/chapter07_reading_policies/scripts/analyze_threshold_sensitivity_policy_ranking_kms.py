"""Analyze policy-ranking sensitivity to the outage threshold on KMS data.

The script reruns the port-selection logic for multiple outage thresholds
``eta_th`` and measures how often the best policy changes. Data-driven policies
must be rebuilt from the design split instead of reusing Monte Carlo outputs
from another reference point.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/analyze_threshold_sensitivity_policy_ranking_kms.py'
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
    RUNS_ROOT,
    SNR_LINEAR,
    TARGET_PORTS,
    THRESHOLD,
    TOTAL_PORTS,
    build_uni_indices,
    build_mar_order,
    build_gc_order,
    build_rag_order,
    build_rep_rng,
    build_suni_indices,
    build_swuni_indices,
    ensure_directory,
    evaluate_selected_ports,
    format_index_sets_zero_based,
    format_indices_one_based,
    format_indices_zero_based,
    load_kms_regime,
    simulate_rnd_policy_result,
    split_design_eval_indices,
    threshold_to_binary,
    write_csv_rows_atomic,
)


# -----------------------------------------------------------------------------
# Sensitivity configuration
# -----------------------------------------------------------------------------
RUN_NAME = "threshold_sensitivity_policy_ranking"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
THRESHOLDS = tuple(THRESHOLD)
if len(THRESHOLDS) == 0:
    raise ValueError("THRESHOLD from policy_mc_kms_common.py must contain at least one value.")
REFERENCE_THRESHOLD = float(THRESHOLDS[len(THRESHOLDS) // 2])
FIGURE_FORMATS = ("png", "pdf")
RND_N_TIMES: tuple[int, ...] = (1, 3, 10)
POLICY_ORDER = (
    "uni",
    "mar",
    "gc",
    "suni",
    "swuni",
    "rag",
    "rnd_1x",
    "rnd_3x",
    "rnd_10x",
)
POLICY_LABELS = {
    "uni": "UNI",
    "mar": "MAR",
    "gc": "GC",
    "suni": "SUNI",
    "swuni": "SWUNI",
    "rag": "RAG",
    "rnd_1x": "RND (1x)",
    "rnd_3x": "RND (3x)",
    "rnd_10x": "RND (10x)",
}
POLICY_PALETTE = {
    "uni": "tab:blue",
    "mar": "tab:orange",
    "gc": "tab:green",
    "suni": "tab:purple",
    "swuni": "tab:red",
    "rag": "tab:brown",
    "rnd_1x": "#999999",
    "rnd_3x": "#666666",
    "rnd_10x": "#333333",
}

DETAIL_FIELDNAMES = (
    "threshold",
    "policy",
    "dataset",
    "regime_index",
    "rep",
    "n_ports",
    "design_size",
    "eval_size",
    "selection_order_0based",
    "sorted_pattern_0based",
    "selection_order_1based",
    "observed_op",
    "n_eval",
    "n_outage",
)
SUMMARY_FIELDNAMES = (
    "threshold",
    "policy",
    "dataset",
    "regime_index",
    "n_ports",
    "n",
    "mean_observed_op",
    "std_observed_op",
    "median_observed_op",
    "min_observed_op",
    "max_observed_op",
)
BEST_POLICY_FIELDNAMES = (
    "threshold",
    "dataset",
    "regime_index",
    "n_ports",
    "best_policy",
    "best_mean_observed_op",
    "second_policy",
    "second_mean_observed_op",
    "margin_to_second",
)
WINNER_COUNT_FIELDNAMES = (
    "threshold",
    "policy",
    "winner_count",
    "winner_share",
)
STABILITY_FIELDNAMES = (
    "threshold",
    "reference_threshold",
    "total_cells",
    "stable_cells",
    "changed_cells",
    "changed_fraction",
)


def threshold_key(threshold: float) -> str:
    """Format one threshold value for stable CSV keys and plot labels.

    Args:
        threshold (float): Outage threshold in linear units. The value is
            expected to be one of ``THRESHOLDS``. Eight decimal places are
            retained because the configured thresholds differ at that precision.

    Returns:
        str: Compact decimal string used consistently across CSV outputs.

    Raises:
        ValueError: If ``threshold`` is not finite.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return f"{threshold:.8f}"


def save_figure(fig: plt.Figure, output_base: Path) -> dict[str, Path]:
    """Save one Matplotlib figure in all configured output formats.

    Args:
        fig (plt.Figure): Figure object to save.
        output_base (Path): Output path without file suffix. The parent
            directory must be writable.

    Returns:
        dict[str, Path]: Saved paths keyed by file extension.

    Raises:
        OSError: If any figure file cannot be written.
    """
    ensure_directory(output_base.parent)
    outputs: dict[str, Path] = {}
    for suffix in FIGURE_FORMATS:
        output_path = output_base.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[suffix] = output_path
    return outputs


def evaluate_threshold_split(
    threshold: float,
    regime_label: str,
    regime_index: int,
    rep_index: int,
    design_values: np.ndarray,
    eval_values: np.ndarray,
    design_means: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Evaluate deterministic and random policies for one threshold and split.

    Args:
        threshold (float): Outage threshold ``eta_th`` used to binarize the
            design and evaluation matrices. Larger values make success harder,
            usually increasing observed outage and potentially changing
            threshold-dependent policy rankings.
        regime_label (str): KMS regime label being processed.
        regime_index (int): Zero-based regime index in ``KMS_REGIME_LABELS``.
        rep_index (int): Zero-based Monte Carlo repetition index.
        design_values (np.ndarray): Design split with shape
            ``(n_design_samples, TOTAL_PORTS)`` used to learn data-driven
            policies.
        eval_values (np.ndarray): Evaluation split with shape
            ``(n_eval_samples, TOTAL_PORTS)`` used to score all policies.
        design_means (np.ndarray): Per-port means on ``design_values``. These
            means are independent of ``threshold`` and are reused across the
            threshold grid for the same repetition.

    Returns:
        list[dict[str, Any]]: Detailed result rows for every configured policy
        and budget.

    Raises:
        ValueError: If input shapes are inconsistent or helper functions reject
        an invalid threshold, budget, or policy input.
    """
    if design_values.shape[1] != TOTAL_PORTS or eval_values.shape[1] != TOTAL_PORTS:
        raise ValueError(f"Expected {TOTAL_PORTS} ports in both design and evaluation splits.")

    design_binary = threshold_to_binary(design_values, threshold=threshold, snr_linear=SNR_LINEAR)
    eval_binary = threshold_to_binary(eval_values, threshold=threshold, snr_linear=SNR_LINEAR)
    mar_order, mar_scores = build_mar_order(
        design_binary=design_binary,
        design_means=design_means,
    )
    gc_order = build_gc_order(
        design_binary=design_binary,
        design_means=design_means,
        mar_scores=mar_scores,
        max_budget=max(TARGET_PORTS),
    )
    rag_order = build_rag_order(
        design_binary=design_binary,
        design_means=design_means,
        mar_scores=mar_scores,
        max_budget=max(TARGET_PORTS),
        redundancy_weight=0.1,
    )
    rows: list[dict[str, Any]] = []
    for n_ports in TARGET_PORTS:
        uni_indices = build_uni_indices(total_ports=TOTAL_PORTS, n_ports=n_ports)
        policy_indices = {
            "uni": uni_indices,
            "mar": mar_order[:n_ports],
            "gc": gc_order[:n_ports],
            "suni": build_suni_indices(
                design_binary=design_binary,
                design_means=design_means,
                n_ports=int(n_ports),
            ),
            "swuni": build_swuni_indices(
                design_binary=design_binary,
                design_means=design_means,
                n_ports=int(n_ports),
            ),
            "rag": rag_order[:n_ports],
        }

        for policy_name in POLICY_ORDER:
            if policy_name.startswith("rnd_"):
                # RND 3x/10x rows are averaged estimators, so record every
                # draw and aggregate outage counts over all sample-draw pairs.
                n_times = int(policy_name.split("_")[1].strip("x"))
                observed_op, n_eval, n_outage, selected_draws = simulate_rnd_policy_result(
                    binary_eval=eval_binary,
                    n_ports=int(n_ports),
                    n_times=n_times,
                    rng=rng,
                )
                selection_order_0based = format_index_sets_zero_based(selected_draws)
                sorted_pattern_0based = format_index_sets_zero_based([np.sort(selected) for selected in selected_draws])
                selection_order_1based = "N/A"
            else:
                selected_indices = policy_indices[policy_name]
                observed_op, n_eval, n_outage = evaluate_selected_ports(
                    binary_eval=eval_binary,
                    selected_indices=selected_indices,
                )
                selection_order_0based = format_indices_zero_based(selected_indices)
                sorted_pattern_0based = format_indices_zero_based(np.sort(selected_indices))
                selection_order_1based = format_indices_one_based(selected_indices)

            rows.append(
                {
                    "threshold": threshold_key(threshold),
                    "policy": policy_name,
                    "dataset": regime_label,
                    "regime_index": int(regime_index),
                    "rep": int(rep_index),
                    "n_ports": int(n_ports),
                    "design_size": int(design_values.shape[0]),
                    "eval_size": int(eval_values.shape[0]),
                    "selection_order_0based": selection_order_0based,
                    "sorted_pattern_0based": sorted_pattern_0based,
                    "selection_order_1based": selection_order_1based,
                    "observed_op": f"{observed_op:.10f}",
                    "n_eval": int(n_eval),
                    "n_outage": int(n_outage),
                }
            )

    return rows


def summarize_detail_rows(
    detail_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate detailed Monte Carlo rows into ranking and stability tables.

    Args:
        detail_rows (list[dict[str, Any]]): Per-repetition rows emitted by
            ``evaluate_threshold_split``. Rows must include all policies for
            every threshold, regime, and budget.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        ``(summary_rows, best_policy_rows, winner_count_rows, stability_rows)``.

    Raises:
        ValueError: If no detail rows are provided, if any ranking cell is
        missing a policy, or if ``REFERENCE_THRESHOLD`` is absent from the
        configured results.
    """
    if len(detail_rows) == 0:
        raise ValueError("detail_rows must contain at least one row.")

    values_by_key: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    regime_index_by_key: dict[tuple[str, str, int], int] = {}

    for row in detail_rows:
        threshold = str(row["threshold"])
        policy = str(row["policy"])
        dataset = str(row["dataset"])
        regime_index = int(row["regime_index"])
        n_ports = int(row["n_ports"])
        values_by_key[(threshold, policy, dataset, regime_index, n_ports)].append(float(row["observed_op"]))
        regime_index_by_key[(threshold, dataset, n_ports)] = regime_index

    summary_rows: list[dict[str, Any]] = []
    mean_by_ranking_key: dict[tuple[str, str, int, str], float] = {}
    for key in sorted(values_by_key):
        threshold, policy, dataset, regime_index, n_ports = key
        values = np.array(values_by_key[key], dtype=np.float64)
        std_value = float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0
        mean_value = float(np.mean(values))
        summary_rows.append(
            {
                "threshold": threshold,
                "policy": policy,
                "dataset": dataset,
                "regime_index": int(regime_index),
                "n_ports": int(n_ports),
                "n": int(values.shape[0]),
                "mean_observed_op": f"{mean_value:.10f}",
                "std_observed_op": f"{std_value:.10f}",
                "median_observed_op": f"{float(np.median(values)):.10f}",
                "min_observed_op": f"{float(np.min(values)):.10f}",
                "max_observed_op": f"{float(np.max(values)):.10f}",
            }
        )
        mean_by_ranking_key[(threshold, dataset, int(n_ports), policy)] = mean_value

    ranking_cells = sorted(regime_index_by_key)
    best_policy_rows: list[dict[str, Any]] = []
    winners_by_threshold: dict[str, list[str]] = defaultdict(list)
    winner_by_cell: dict[tuple[str, str, int], str] = {}

    for threshold, dataset, n_ports in ranking_cells:
        policy_means = []
        for policy_name in POLICY_ORDER:
            ranking_key = (threshold, dataset, n_ports, policy_name)
            if ranking_key not in mean_by_ranking_key:
                raise ValueError(f"Missing policy result for ranking cell: {ranking_key}")
            policy_means.append((policy_name, mean_by_ranking_key[ranking_key]))

        # Exact ties are unlikely with averaged Monte Carlo values. The policy
        # order keeps the classification deterministic if one does occur.
        ranked = sorted(policy_means, key=lambda item: (item[1], POLICY_ORDER.index(item[0])))
        best_policy, best_mean = ranked[0]
        second_policy, second_mean = ranked[1]
        winners_by_threshold[threshold].append(best_policy)
        winner_by_cell[(threshold, dataset, n_ports)] = best_policy
        best_policy_rows.append(
            {
                "threshold": threshold,
                "dataset": dataset,
                "regime_index": int(regime_index_by_key[(threshold, dataset, n_ports)]),
                "n_ports": int(n_ports),
                "best_policy": best_policy,
                "best_mean_observed_op": f"{best_mean:.10f}",
                "second_policy": second_policy,
                "second_mean_observed_op": f"{second_mean:.10f}",
                "margin_to_second": f"{float(second_mean - best_mean):.10f}",
            }
        )

    winner_count_rows = build_winner_count_rows(winners_by_threshold=winners_by_threshold)
    stability_rows = build_stability_rows(winner_by_cell=winner_by_cell)
    return summary_rows, best_policy_rows, winner_count_rows, stability_rows


def build_winner_count_rows(winners_by_threshold: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Count how many ranking cells each policy wins at each threshold.

    Args:
        winners_by_threshold (dict[str, list[str]]): Mapping from threshold key
            to best-policy labels across all regime/budget cells.

    Returns:
        list[dict[str, Any]]: Winner counts and shares for each threshold and
        policy.

    Raises:
        ValueError: If any threshold has no winner entries.
    """
    rows: list[dict[str, Any]] = []
    for threshold in sorted(winners_by_threshold, key=float):
        winners = winners_by_threshold[threshold]
        if len(winners) == 0:
            raise ValueError(f"Threshold {threshold} has no winner entries.")
        for policy_name in POLICY_ORDER:
            count = int(sum(1 for winner in winners if winner == policy_name))
            rows.append(
                {
                    "threshold": threshold,
                    "policy": policy_name,
                    "winner_count": count,
                    "winner_share": f"{float(count / len(winners)):.10f}",
                }
            )
    return rows


def build_stability_rows(winner_by_cell: dict[tuple[str, str, int], str]) -> list[dict[str, Any]]:
    """Measure classification changes relative to ``REFERENCE_THRESHOLD``.

    Args:
        winner_by_cell (dict[tuple[str, str, int], str]): Best policy keyed by
            ``(threshold, dataset, n_ports)``.

    Returns:
        list[dict[str, Any]]: Threshold-level stability rows with changed-cell
        counts and fractions relative to ``REFERENCE_THRESHOLD``.

    Raises:
        ValueError: If the reference threshold is absent or any non-reference
        threshold has a different set of dataset/budget cells.
    """
    reference = threshold_key(REFERENCE_THRESHOLD)
    thresholds = sorted({key[0] for key in winner_by_cell}, key=float)
    if reference not in thresholds:
        raise ValueError(f"REFERENCE_THRESHOLD={REFERENCE_THRESHOLD} is absent from the results.")

    reference_cells = {
        (dataset, n_ports): winner
        for threshold, dataset, n_ports in winner_by_cell
        if threshold == reference
        for winner in [winner_by_cell[(threshold, dataset, n_ports)]]
    }

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_cells = {
            (dataset, n_ports): winner
            for cell_threshold, dataset, n_ports in winner_by_cell
            if cell_threshold == threshold
            for winner in [winner_by_cell[(cell_threshold, dataset, n_ports)]]
        }
        if set(threshold_cells) != set(reference_cells):
            raise ValueError(f"Threshold {threshold} does not match the reference cell set.")

        changed_count = int(
            sum(1 for cell_key, reference_winner in reference_cells.items() if threshold_cells[cell_key] != reference_winner)
        )
        total_cells = len(reference_cells)
        rows.append(
            {
                "threshold": threshold,
                "reference_threshold": reference,
                "total_cells": int(total_cells),
                "stable_cells": int(total_cells - changed_count),
                "changed_cells": int(changed_count),
                "changed_fraction": f"{float(changed_count / total_cells):.10f}",
            }
        )

    return rows


def plot_winner_counts(winner_count_rows: list[dict[str, Any]], output_base: Path) -> dict[str, Path]:
    """Plot stacked winner counts by threshold.

    Args:
        winner_count_rows (list[dict[str, Any]]): Rows from
            ``build_winner_count_rows``.
        output_base (Path): Output path without suffix.

    Returns:
        dict[str, Path]: Saved figure paths keyed by suffix.

    Raises:
        ValueError: If the count table is empty.
        OSError: If the figure cannot be written.
    """
    if len(winner_count_rows) == 0:
        raise ValueError("winner_count_rows must not be empty.")

    thresholds = sorted({str(row["threshold"]) for row in winner_count_rows}, key=float)
    x_positions = np.arange(len(thresholds), dtype=np.float64)
    bottom = np.zeros(len(thresholds), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for policy_name in POLICY_ORDER:
        counts = np.array(
            [
                int(next(row["winner_count"] for row in winner_count_rows if row["threshold"] == threshold and row["policy"] == policy_name))
                for threshold in thresholds
            ],
            dtype=np.float64,
        )
        ax.bar(
            x_positions,
            counts,
            bottom=bottom,
            color=POLICY_PALETTE[policy_name],
            label=POLICY_LABELS[policy_name],
        )
        bottom += counts

    ax.set_xlabel(r"Threshold $\eta_{\mathrm{th}}$")
    ax.set_ylabel("Winning regime-budget cells")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(thresholds)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    outputs = save_figure(fig=fig, output_base=output_base)
    plt.close(fig)
    return outputs


def plot_stability(stability_rows: list[dict[str, Any]], output_base: Path) -> dict[str, Path]:
    """Plot classification-change fraction relative to the reference threshold.

    Args:
        stability_rows (list[dict[str, Any]]): Rows from
            ``build_stability_rows``.
        output_base (Path): Output path without suffix.

    Returns:
        dict[str, Path]: Saved figure paths keyed by suffix.

    Raises:
        ValueError: If ``stability_rows`` is empty.
        OSError: If the figure cannot be written.
    """
    if len(stability_rows) == 0:
        raise ValueError("stability_rows must not be empty.")

    rows = sorted(stability_rows, key=lambda row: float(row["threshold"]))
    thresholds = np.array([float(row["threshold"]) for row in rows], dtype=np.float64)
    changed_fraction = np.array([float(row["changed_fraction"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(thresholds, 100.0 * changed_fraction, color="black", marker="o", linewidth=1.4)
    ax.set_xlabel(r"Threshold $\eta_{\mathrm{th}}$")
    ax.set_ylabel(f"Changed classifications vs. {threshold_key(REFERENCE_THRESHOLD)} (%)")
    ax.set_xticks(thresholds)
    max_percent = float(100.0 * np.max(changed_fraction))
    ax.set_ylim(0.0, max(20.0, 5.0 * math.ceil(max_percent / 5.0)))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    outputs = save_figure(fig=fig, output_base=output_base)
    plt.close(fig)
    return outputs


def plot_mean_observed_op(summary_rows: list[dict[str, Any]], output_base: Path) -> dict[str, Path]:
    """Plot average observed outage by policy and threshold.

    Args:
        summary_rows (list[dict[str, Any]]): Aggregated rows returned by
            ``summarize_detail_rows``. Each row contributes one regime/budget
            mean, so the plotted values weight regime-budget cells equally.
        output_base (Path): Output path without suffix.

    Returns:
        dict[str, Path]: Saved figure paths keyed by suffix.

    Raises:
        ValueError: If ``summary_rows`` is empty.
        OSError: If the figure cannot be written.
    """
    if len(summary_rows) == 0:
        raise ValueError("summary_rows must not be empty.")

    thresholds = sorted({str(row["threshold"]) for row in summary_rows}, key=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for policy_name in POLICY_ORDER:
        means = []
        for threshold in thresholds:
            values = [
                float(row["mean_observed_op"])
                for row in summary_rows
                if row["threshold"] == threshold and row["policy"] == policy_name
            ]
            if len(values) == 0:
                raise ValueError(f"Missing summary values for {policy_name} at threshold {threshold}.")
            means.append(float(np.mean(values)))

        ax.plot(
            np.array([float(value) for value in thresholds], dtype=np.float64),
            np.array(means, dtype=np.float64),
            marker="o",
            linewidth=1.4,
            color=POLICY_PALETTE[policy_name],
            label=POLICY_LABELS[policy_name],
        )

    ax.set_xlabel(r"Threshold $\eta_{\mathrm{th}}$")
    ax.set_ylabel("Mean observed OP")
    ax.set_xticks(np.array([float(value) for value in thresholds], dtype=np.float64))
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    outputs = save_figure(fig=fig, output_base=output_base)
    plt.close(fig)
    return outputs


def run_threshold_sensitivity_analysis() -> dict[str, Any]:
    """Run threshold-sensitivity analysis and plots.

    Args:
        None: The analysis uses module-level constants for thresholds, budgets,
        repetitions, paths, and SNR.

    Returns:
        dict[str, Any]: Paths to generated CSVs and figures.

    Raises:
        FileNotFoundError: If a KMS regime file is missing.
        ValueError: If configuration or generated rows are inconsistent.
        OSError: If outputs cannot be written.
    """
    if len(THRESHOLDS) == 0:
        raise ValueError("THRESHOLDS must contain at least one value.")
    if threshold_key(REFERENCE_THRESHOLD) not in {threshold_key(value) for value in THRESHOLDS}:
        raise ValueError("REFERENCE_THRESHOLD must be included in THRESHOLDS.")
    if NUM_MC_REPS < 1:
        raise ValueError("NUM_MC_REPS must be at least 1.")

    output_dir = ensure_directory(OUTPUT_DIR)
    detail_rows: list[dict[str, Any]] = []

    for regime_index, regime_label in enumerate(KMS_REGIME_LABELS):
        print(
            f"[threshold_sensitivity] Loading regime {regime_index + 1}/{len(KMS_REGIME_LABELS)}: {regime_label}",
            flush=True,
        )
        regime_data = load_kms_regime(regime_label=regime_label, common_path=COMMON_DATASET_PATH)
        if regime_data.shape[1] != TOTAL_PORTS:
            raise ValueError(f"Regime {regime_label} has {regime_data.shape[1]} ports, expected {TOTAL_PORTS}.")

        for rep_index in range(NUM_MC_REPS):
            rng = build_rep_rng(base_seed=DATA_SEED, regime_index=regime_index, rep_index=rep_index)
            design_indices, eval_indices = split_design_eval_indices(
                n_samples=regime_data.shape[0],
                eval_fraction=MC_EVAL_FRACTION,
                rng=rng,
            )

            # The row split is shared across thresholds so changes in ranking
            # are attributable to eta_th rather than different Monte Carlo
            # samples.
            design_values = regime_data[design_indices]
            eval_values = regime_data[eval_indices]
            design_means = np.mean(design_values, axis=0, dtype=np.float64)

            for threshold in THRESHOLDS:
                detail_rows.extend(
                    evaluate_threshold_split(
                        threshold=threshold,
                        regime_label=regime_label,
                        regime_index=regime_index,
                        rep_index=rep_index,
                        design_values=design_values,
                        eval_values=eval_values,
                        design_means=design_means,
                        rng=rng,
                    )
                )

            del design_values, eval_values, design_means

        del regime_data

    summary_rows, best_policy_rows, winner_count_rows, stability_rows = summarize_detail_rows(detail_rows)

    detail_path = write_csv_rows_atomic(output_dir / "threshold_sensitivity_details.csv", DETAIL_FIELDNAMES, detail_rows)
    summary_path = write_csv_rows_atomic(output_dir / "threshold_sensitivity_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    best_policy_path = write_csv_rows_atomic(
        output_dir / "threshold_sensitivity_best_policy.csv",
        BEST_POLICY_FIELDNAMES,
        best_policy_rows,
    )
    winner_count_path = write_csv_rows_atomic(
        output_dir / "threshold_sensitivity_winner_counts.csv",
        WINNER_COUNT_FIELDNAMES,
        winner_count_rows,
    )
    stability_path = write_csv_rows_atomic(
        output_dir / "threshold_sensitivity_stability.csv",
        STABILITY_FIELDNAMES,
        stability_rows,
    )

    figure_dir = ensure_directory(output_dir / "figures")
    winner_count_figures = plot_winner_counts(
        winner_count_rows=winner_count_rows,
        output_base=figure_dir / "winner_counts_by_threshold",
    )
    stability_figures = plot_stability(
        stability_rows=stability_rows,
        output_base=figure_dir / "classification_changes_vs_reference",
    )
    mean_op_figures = plot_mean_observed_op(
        summary_rows=summary_rows,
        output_base=figure_dir / "mean_observed_op_by_threshold",
    )
    print("")
    print(f"Saved detail rows: {detail_path}")
    print(f"Saved summary rows: {summary_path}")
    print(f"Saved best-policy rows: {best_policy_path}")
    print(f"Saved winner counts: {winner_count_path}")
    print(f"Saved stability rows: {stability_path}")
    print("Saved figures:")
    for figure_group in (winner_count_figures, stability_figures, mean_op_figures):
        for figure_path in figure_group.values():
            print(f"  {figure_path}")

    return {
        "details": detail_path,
        "summary": summary_path,
        "best_policy": best_policy_path,
        "winner_counts": winner_count_path,
        "stability": stability_path,
        "winner_count_figures": winner_count_figures,
        "stability_figures": stability_figures,
        "mean_op_figures": mean_op_figures,
    }


if __name__ == "__main__":
    run_threshold_sensitivity_analysis()
