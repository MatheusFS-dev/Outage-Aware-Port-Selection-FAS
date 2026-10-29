"""Estimate linear outage cutoffs for global and per-regime KMS data.

This script answers a practical question for real systems:

"Given a target outage probability, what linear cutoff should I use?"

The analysis is based on the repo's outage rule:
``selected_value <= threshold / snr_linear``.

For a fixed ``snr_linear`` value, the script finds the empirical cutoff value
whose sample outage rate is closest to the requested target outage
probability. It reports both the raw cutoff and the corresponding threshold in
the same linear scale used by the OP evaluation code.

By default, the script sweeps the configured target outage probabilities and
observed-port counts, writing one concise output folder per configuration.
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
import pandas as pd

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter04_system_model/scripts/find_linear_outage_threshold_kms.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from utils.constants import BLUE, GREEN, RESET
from utils.data import (
    KMS_REGIME_LABELS,
    DEFAULT_KMS_RUNTIME_CACHE_PATH,
    get_observed_ports,
    load_raw_kms_cache,
)


COMMON_DATASET_PATH = DEFAULT_KMS_RUNTIME_CACHE_PATH
OUTPUT_ROOT = _COMPANION_ROOT / 'studies/chapter07_reading_policies/results' / _COMPANION_ROOT / 'studies/chapter04_system_model/scripts/find_linear_outage_threshold_kms.py'.stem
TARGET_OUTAGE_PROBABILITY = [0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
SNR_LINEAR = 1.0
TOTAL_PORTS = 100
NUM_OBSERVED_PORTS = [3, 4, 5, 6, 7]


def _validate_target_outage_probability(target_outage_probability: float) -> None:
    """Validate the requested outage target.

    Args:
        target_outage_probability (float): Desired outage rate in the open
            interval ``(0, 1]``.

    Returns:
        None.

    Raises:
        ValueError: If the target is outside the supported range.
    """
    if not 0.0 < target_outage_probability <= 1.0:
        raise ValueError("target_outage_probability must be in the interval (0, 1].")


def _format_target_outage_folder(target_outage_probability: float) -> str:
    """Return a sortable folder label for a target outage probability.

    Args:
        target_outage_probability (float): Desired outage probability in the
            interval ``(0, 1]``. The value is formatted with exactly three
            decimal places so lexicographic folder sorting follows the numeric
            sweep order for the configured values.

    Returns:
        str: Folder-safe label in the form ``op<target>``.

    Raises:
        ValueError: If ``target_outage_probability`` is outside ``(0, 1]``.

    Examples:
        >>> _format_target_outage_folder(0.10)
        'op0.100'
    """
    _validate_target_outage_probability(target_outage_probability)

    return f"op{target_outage_probability:.3f}"


def _format_observed_ports_folder(num_observed_ports: int) -> str:
    """Return a folder label for an observed-port count.

    Args:
        num_observed_ports (int): Number of observed ports used by the observed
            selection baseline. The value is written after the ``obs`` prefix
            and is not padded because the configured counts are single-digit
            values.

    Returns:
        str: Folder-safe label in the form ``obs<count>``.

    Raises:
        ValueError: If ``num_observed_ports`` is not positive.

    Examples:
        >>> _format_observed_ports_folder(3)
        'obs3'
    """
    if num_observed_ports <= 0:
        raise ValueError("num_observed_ports must be positive.")

    return f"obs{num_observed_ports}"


def _find_sample_cutoff(selected_values: np.ndarray, target_outage_probability: float) -> float:
    """Return the empirical cutoff that matches a target outage rate.

    The cutoff is chosen as the order statistic at index
    ``ceil(target * n) - 1`` so the resulting sample outage probability is at
    least the target rate and remains easy to interpret on finite datasets.

    Args:
        selected_values (np.ndarray): One-dimensional array of realized values
            used to evaluate outage.
        target_outage_probability (float): Desired outage probability in the
            interval ``(0, 1]``.

    Returns:
        float: Empirical cutoff value in the same linear scale as
        ``selected_values``.

    Raises:
        ValueError: If the input is empty or not one-dimensional.
    """
    if selected_values.ndim != 1:
        raise ValueError("selected_values must be one-dimensional.")
    if selected_values.size == 0:
        raise ValueError("selected_values must not be empty.")

    _validate_target_outage_probability(target_outage_probability)

    sorted_values = np.sort(selected_values.astype(np.float64, copy=False))
    cutoff_index = max(0, min(sorted_values.size - 1, math.ceil(target_outage_probability * sorted_values.size) - 1))
    return float(sorted_values[cutoff_index])


def _build_regime_rows(
    scope: str,
    regime_label: str,
    regime_index: int,
    regime_data: np.ndarray,
    target_outage_probability: float,
    snr_linear: float,
    num_observed_ports: int,
    total_ports: int,
) -> list[dict[str, Any]]:
    """Build linear cutoff rows for one KMS dataset scope.

    Args:
        scope (str): Output grouping for the row. Use ``"global"`` for the
            combined all-regime calculation and ``"per_regime"`` for an
            individual KMS regime. This value only changes output labeling; it
            does not change the threshold calculation.
        regime_label (str): Canonical KMS label.
        regime_index (int): Zero-based index in the canonical regime order, or
            ``-1`` for the combined all-regime calculation.
        regime_data (np.ndarray): Full dataset matrix with shape
            ``(n_samples, total_ports)``.
        target_outage_probability (float): Desired outage probability.
        snr_linear (float): SNR scaling factor in linear units. The reported
            threshold is ``cutoff * snr_linear``.
        num_observed_ports (int): Number of observed ports for the observed
            selection baseline.
        total_ports (int): Total number of ports in each sample.

    Returns:
        list[dict[str, Any]]: One row for the ideal full-observation cutoff and
            one row for the observed-port cutoff.

    Raises:
        ValueError: If ``regime_data`` is not a non-empty 2D matrix.

    Examples:
        >>> rows = _build_regime_rows(
        ...     scope="per_regime",
        ...     regime_label="example",
        ...     regime_index=0,
        ...     regime_data=np.ones((3, 100), dtype=np.float32),
        ...     target_outage_probability=0.5,
        ...     snr_linear=1.0,
        ...     num_observed_ports=4,
        ...     total_ports=100,
        ... )
        >>> len(rows)
        2
    """
    if regime_data.ndim != 2 or regime_data.shape[0] == 0 or regime_data.shape[1] != total_ports:
        raise ValueError("regime_data must have shape (n_samples, total_ports) and contain at least one sample.")

    if snr_linear <= 0.0:
        raise ValueError("snr_linear must be positive.")

    ideal_selected_values = np.max(regime_data, axis=1).astype(np.float32)
    observed_values, observed_indices = get_observed_ports(regime_data, num_observed_ports, total_ports)
    observed_selected_values = np.max(observed_values, axis=1).astype(np.float32)
    rows = _build_selected_value_rows(
        scope=scope,
        dataset_label=regime_label,
        regime_index=regime_index,
        ideal_selected_values=ideal_selected_values,
        observed_selected_values=observed_selected_values,
        observed_indices=observed_indices,
        target_outage_probability=target_outage_probability,
        snr_linear=snr_linear,
        num_observed_ports=num_observed_ports,
    )

    return rows


def _build_selected_value_rows(
    scope: str,
    dataset_label: str,
    regime_index: int,
    ideal_selected_values: np.ndarray,
    observed_selected_values: np.ndarray,
    observed_indices: np.ndarray,
    target_outage_probability: float,
    snr_linear: float,
    num_observed_ports: int,
) -> list[dict[str, Any]]:
    """Build threshold rows from precomputed ideal and observed selections.

    Args:
        scope (str): Output grouping for the row. Use ``"global"`` for
            combined all-regime selections and ``"per_regime"`` for individual
            regime selections. This value only changes output labeling.
        dataset_label (str): Label written to the output ``dataset`` column.
        regime_index (int): Zero-based regime index, or ``-1`` for combined
            all-regime selections.
        ideal_selected_values (np.ndarray): One-dimensional array containing
            one ideal full-observation selected value per sample.
        observed_selected_values (np.ndarray): One-dimensional array containing
            one observed-port selected value per sample.
        observed_indices (np.ndarray): One-dimensional integer array of the
            observed port indices used to build ``observed_selected_values``.
        target_outage_probability (float): Desired outage probability in the
            interval ``(0, 1]``.
        snr_linear (float): SNR scaling factor in linear units. The reported
            threshold is ``cutoff * snr_linear`` and this value must be positive.
        num_observed_ports (int): Number of observed ports used for the observed
            selection baseline.

    Returns:
        list[dict[str, Any]]: One ideal row and one observed-port row.

    Raises:
        ValueError: If either selected-value array is invalid, the two arrays
            have different lengths, or ``snr_linear`` is not positive.

    Examples:
        >>> rows = _build_selected_value_rows(
        ...     scope="global",
        ...     dataset_label="all_regimes",
        ...     regime_index=-1,
        ...     ideal_selected_values=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        ...     observed_selected_values=np.array([0.05, 0.15, 0.25], dtype=np.float32),
        ...     observed_indices=np.array([0, 33, 66, 99]),
        ...     target_outage_probability=0.5,
        ...     snr_linear=1.0,
        ...     num_observed_ports=4,
        ... )
        >>> len(rows)
        2
    """
    if ideal_selected_values.ndim != 1 or observed_selected_values.ndim != 1:
        raise ValueError("selected-value arrays must be one-dimensional.")
    if ideal_selected_values.size == 0:
        raise ValueError("selected-value arrays must not be empty.")
    if ideal_selected_values.shape[0] != observed_selected_values.shape[0]:
        raise ValueError("selected-value arrays must contain the same number of samples.")
    if snr_linear <= 0.0:
        raise ValueError("snr_linear must be positive.")

    rows: list[dict[str, Any]] = []
    for selection_mode, selected_values in (
        ("ideal", ideal_selected_values),
        ("observed", observed_selected_values),
    ):
        cutoff_value = _find_sample_cutoff(selected_values, target_outage_probability)
        threshold_value = cutoff_value * snr_linear
        outage_flags = selected_values <= cutoff_value
        rows.append(
            {
                "dataset": dataset_label,
                "scope": scope,
                "regime_index": int(regime_index),
                "selection_mode": selection_mode,
                "target_outage_probability": float(target_outage_probability),
                "snr_linear": float(snr_linear),
                "snr_db": float(10.0 * math.log10(snr_linear)),
                "threshold_over_snr": float(cutoff_value),
                "threshold_linear": float(threshold_value),
                "n_samples": int(selected_values.shape[0]),
                "n_outage_samples": int(np.sum(outage_flags)),
                "empirical_outage_probability": float(np.mean(outage_flags.astype(np.float64))),
                "observed_port_count": int(num_observed_ports),
                "observed_indices": ",".join(str(index) for index in observed_indices.tolist()),
            }
        )

    return rows


def _format_result_block(title: str, rows_df: pd.DataFrame) -> list[str]:
    """Format threshold rows as a compact text summary block.

    Args:
        title (str): Section title written above the formatted rows.
        rows_df (pd.DataFrame): Threshold rows to format. The frame must include
            ``dataset``, ``selection_mode``, ``threshold_over_snr``,
            ``threshold_linear``, ``empirical_outage_probability``,
            ``n_samples``, and ``observed_indices`` columns. Rows are printed in
            their current order, so callers control the grouping and sorting.

    Returns:
        list[str]: Lines ready to write to the summary text file.

    Raises:
        ValueError: If ``rows_df`` is empty.

    Examples:
        >>> frame = pd.DataFrame(
        ...     [
        ...         {
        ...             "dataset": "all_regimes",
        ...             "selection_mode": "ideal",
        ...             "threshold_over_snr": 0.2,
        ...             "threshold_linear": 0.2,
        ...             "empirical_outage_probability": 0.5,
        ...             "n_samples": 3,
        ...             "observed_indices": "0,33,66,99",
        ...         }
        ...     ]
        ... )
        >>> _format_result_block("Global:", frame)[0]
        'Global:'
    """
    if rows_df.empty:
        raise ValueError("rows_df must not be empty.")

    lines = [title]
    for _, row in rows_df.iterrows():
        lines.extend(
            [
                f"  {row['dataset']} [{row['selection_mode']}]",
                f"    threshold_over_snr = {row['threshold_over_snr']:.8f}",
                f"    threshold_linear   = {row['threshold_linear']:.8f}",
                f"    empirical_outage   = {row['empirical_outage_probability']:.8f}",
                f"    n_samples          = {int(row['n_samples'])}",
                f"    observed_indices   = {row['observed_indices']}",
            ]
        )

    return lines


def _build_global_average_rows(per_regime_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build global rows by averaging the per-regime threshold results.

    Args:
        per_regime_df (pd.DataFrame): Per-regime threshold rows. The frame must
            contain one or more rows for each ``selection_mode`` and include the
            same columns emitted by ``_build_regime_rows``. The global result is
            the arithmetic mean of per-regime thresholds for each selection
            mode, while sample and outage counts are summed.

    Returns:
        list[dict[str, Any]]: One global average row per selection mode.

    Raises:
        ValueError: If ``per_regime_df`` is empty or required columns are
            missing.

    Examples:
        >>> rows = _build_global_average_rows(
        ...     pd.DataFrame(
        ...         [
        ...             {
        ...                 "dataset": "a",
        ...                 "scope": "per_regime",
        ...                 "regime_index": 0,
        ...                 "selection_mode": "ideal",
        ...                 "target_outage_probability": 0.1,
        ...                 "snr_linear": 1.0,
        ...                 "snr_db": 0.0,
        ...                 "threshold_over_snr": 0.2,
        ...                 "threshold_linear": 0.2,
        ...                 "n_samples": 10,
        ...                 "n_outage_samples": 1,
        ...                 "empirical_outage_probability": 0.1,
        ...                 "observed_port_count": 4,
        ...                 "observed_indices": "0,33,66,99",
        ...             }
        ...         ]
        ...     )
        ... )
        >>> rows[0]["dataset"]
        'all_regimes_average'
    """
    required_columns = {
        "selection_mode",
        "target_outage_probability",
        "snr_linear",
        "snr_db",
        "threshold_over_snr",
        "threshold_linear",
        "n_samples",
        "n_outage_samples",
        "observed_port_count",
        "observed_indices",
    }
    missing_columns = required_columns.difference(per_regime_df.columns)
    if missing_columns:
        raise ValueError(f"per_regime_df is missing required columns: {sorted(missing_columns)}")
    if per_regime_df.empty:
        raise ValueError("per_regime_df must not be empty.")

    rows: list[dict[str, Any]] = []
    for selection_mode, selection_df in per_regime_df.groupby("selection_mode", sort=False):
        n_samples = int(selection_df["n_samples"].sum())
        n_outage_samples = int(selection_df["n_outage_samples"].sum())
        rows.append(
            {
                "dataset": "all_regimes_average",
                "scope": "global",
                "regime_index": -1,
                "selection_mode": selection_mode,
                "target_outage_probability": float(selection_df["target_outage_probability"].iloc[0]),
                "snr_linear": float(selection_df["snr_linear"].iloc[0]),
                "snr_db": float(selection_df["snr_db"].iloc[0]),
                "threshold_over_snr": float(selection_df["threshold_over_snr"].mean()),
                "threshold_linear": float(selection_df["threshold_linear"].mean()),
                "n_samples": n_samples,
                "n_outage_samples": n_outage_samples,
                "empirical_outage_probability": float(n_outage_samples / n_samples),
                "observed_port_count": int(selection_df["observed_port_count"].iloc[0]),
                "observed_indices": selection_df["observed_indices"].iloc[0],
            }
        )

    return rows


def _run_linear_outage_threshold_search_config(
    datasets: list[np.ndarray],
    target_outage_probability: float,
    num_observed_ports: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run and write threshold results for one sweep configuration.

    Args:
        datasets (list[np.ndarray]): KMS regime matrices loaded in canonical
            regime order. Each matrix must have shape ``(n_samples,
            TOTAL_PORTS)`` so every configured regime is processed with the
            same port count.
        target_outage_probability (float): Desired outage probability in the
            interval ``(0, 1]`` for this output configuration.
        num_observed_ports (int): Number of observed ports used by the observed
            selection baseline for this output configuration.
        output_dir (Path): Directory that will receive this configuration's
            summary, combined CSV, global CSV, and per-regime CSV.

    Returns:
        dict[str, Any]: Summary of the generated analysis artifacts.

    Raises:
        ValueError: If the target outage probability, SNR configuration, or any
            regime matrix is invalid.
        OSError: If the output files cannot be written.
    """
    _validate_target_outage_probability(target_outage_probability)
    if SNR_LINEAR <= 0.0:
        raise ValueError("SNR_LINEAR must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{BLUE}Target outage probability={target_outage_probability:.4f}, "
        f"NUM_OBSERVED_PORTS={num_observed_ports}, SNR_LINEAR={SNR_LINEAR:.4f}{RESET}"
    )

    per_regime_rows: list[dict[str, Any]] = []

    for regime_index, (regime_label, regime_data) in enumerate(zip(KMS_REGIME_LABELS, datasets, strict=True)):
        regime_rows = _build_regime_rows(
            scope="per_regime",
            regime_label=regime_label,
            regime_index=regime_index,
            regime_data=regime_data.astype(np.float32, copy=False),
            target_outage_probability=target_outage_probability,
            snr_linear=SNR_LINEAR,
            num_observed_ports=num_observed_ports,
            total_ports=TOTAL_PORTS,
        )
        per_regime_rows.extend(regime_rows)

    per_regime_df = pd.DataFrame(per_regime_rows)
    # The global result is the all-regime average, so it is derived from the
    # per-regime threshold rows rather than from a second raw-data pass.
    global_df = pd.DataFrame(_build_global_average_rows(per_regime_df))
    results_df = pd.concat([global_df, per_regime_df], ignore_index=True)

    global_dir = output_dir / "global"
    per_regime_dir = output_dir / "per_regime"
    global_dir.mkdir(parents=True, exist_ok=True)
    per_regime_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "linear_outage_thresholds_all_scopes.csv"
    global_results_path = global_dir / "linear_outage_thresholds.csv"
    per_regime_results_path = per_regime_dir / "linear_outage_thresholds.csv"
    results_df.to_csv(results_path, index=False)
    global_df.to_csv(global_results_path, index=False)
    per_regime_df.to_csv(per_regime_results_path, index=False)

    summary_path = output_dir / "summary.txt"
    summary_lines = [
        f"Target outage probability: {target_outage_probability:.6f}",
        f"NUM_OBSERVED_PORTS: {num_observed_ports}",
        f"SNR_LINEAR: {SNR_LINEAR:.6f}",
        "",
    ]
    summary_lines.extend(_format_result_block("Global all-regime average result:", global_df))
    summary_lines.append("")
    summary_lines.extend(_format_result_block("Per-regime results:", per_regime_df))
    summary_lines.extend(
        [
            "",
            f"Combined CSV: {results_path}",
            f"Global CSV: {global_results_path}",
            f"Per-regime CSV: {per_regime_results_path}",
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"{GREEN}Wrote combined linear outage threshold results to {results_path}{RESET}")
    print(f"{GREEN}Wrote global linear outage threshold results to {global_results_path}{RESET}")
    print(f"{GREEN}Wrote per-regime linear outage threshold results to {per_regime_results_path}{RESET}")
    print(f"{GREEN}Wrote summary to {summary_path}{RESET}")

    return {
        "output_dir": output_dir,
        "results_table": results_path,
        "global_results_table": global_results_path,
        "per_regime_results_table": per_regime_results_path,
        "summary": summary_path,
        "target_outage_probability": target_outage_probability,
        "num_observed_ports": num_observed_ports,
        "snr_linear": SNR_LINEAR,
    }


def run_linear_outage_threshold_search() -> dict[str, Any]:
    """Run the configured cutoff search sweep.

    Args:
        None.

    Returns:
        dict[str, Any]: Summary of the generated analysis artifacts across all
            target outage probability and observed-port count combinations.

    Raises:
        FileNotFoundError: If the runtime cache is missing.
        ValueError: If any target outage probability or SNR configuration is
            invalid, or if either configured sweep list is empty.
        OSError: If any output files cannot be written.
    """
    if SNR_LINEAR <= 0.0:
        raise ValueError("SNR_LINEAR must be positive.")
    if not TARGET_OUTAGE_PROBABILITY:
        raise ValueError("TARGET_OUTAGE_PROBABILITY must not be empty.")
    if not NUM_OBSERVED_PORTS:
        raise ValueError("NUM_OBSERVED_PORTS must not be empty.")
    for target_outage_probability in TARGET_OUTAGE_PROBABILITY:
        _validate_target_outage_probability(target_outage_probability)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{BLUE}Starting linear outage threshold search sweep...{RESET}")
    datasets = load_raw_kms_cache(cache_path=COMMON_DATASET_PATH, regime_labels=None)

    artifact_summaries: list[dict[str, Any]] = []
    for target_outage_probability in TARGET_OUTAGE_PROBABILITY:
        target_outage_folder = _format_target_outage_folder(target_outage_probability)
        for num_observed_ports in NUM_OBSERVED_PORTS:
            observed_ports_folder = _format_observed_ports_folder(num_observed_ports)
            artifact_summaries.append(
                _run_linear_outage_threshold_search_config(
                    datasets=datasets,
                    target_outage_probability=target_outage_probability,
                    num_observed_ports=num_observed_ports,
                    output_dir=OUTPUT_ROOT / target_outage_folder / observed_ports_folder,
                )
            )

    return {
        "output_root": OUTPUT_ROOT,
        "target_outage_probabilities": TARGET_OUTAGE_PROBABILITY,
        "num_observed_ports_list": NUM_OBSERVED_PORTS,
        "snr_linear": SNR_LINEAR,
        "artifacts": artifact_summaries,
    }


if __name__ == "__main__":
    run_linear_outage_threshold_search()
