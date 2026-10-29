"""Shared helpers for per-regime Monte Carlo port-reading studies on KMS data.

This file is the common foundation used by the executable scripts in the
port-reading analysis workflow. Its job is to keep the deterministic policy
simulators and comparison/plotting scripts consistent with each other. Instead
of duplicating the same Monte Carlo, dataset, and CSV logic in several places,
this module defines one shared implementation that all scripts can call.

High-level purpose:
- Load one KMS regime at a time without materializing the full 14-regime
  dataset in memory.
- Build deterministic Monte Carlo design/evaluation splits for each regime and
  repetition.
- Convert float-valued channel matrices into binary threshold-exceedance
  matrices, which are sufficient for observed-only outage estimation.
- Implement the reusable policy primitives needed by the two simulators:
  equally spaced UNI, ranking-based MAR, and greedy GC.
- Evaluate observed-only outage, ideal outage, and uncertainty intervals.
- Provide resumable file-writing helpers so interrupted runs can continue
  safely from the last completed repetition.

Step-by-step layout of this module:
1. Path bootstrap:
   The first lines resolve the repository root and add `src` to `sys.path`
   so the helpers can import `utils.data` no matter where the script is run
   from.
2. Canonical constants:
   `KMS_REGIME_FILES` and `KMS_REGIME_LABELS` define the exact 14-regime
   ordering used across all generated outputs. `RESULT_FIELDNAMES` and
   `CANONICAL_FIELDNAMES` define stable CSV schemas.
3. Resumable filesystem helpers:
   `ensure_directory`, `write_json_atomic`, `write_csv_rows_atomic`, and
   `ensure_matching_run_config` make sure a run can be resumed safely and does
   not silently mix outputs created with incompatible configurations.
4. Dataset helpers:
   `load_kms_regime`, `build_rep_rng`, `split_design_eval_indices`, and
   `threshold_to_binary` implement the regime-local Monte Carlo setup.
5. Policy helpers:
   `build_uni_indices`, `build_mar_order`, and
  `build_gc_order`, `build_suni_indices`, `build_swuni_indices`, and
  `build_rag_order` define how the sensing patterns are created.
6. Evaluation helpers:
   `evaluate_selected_ports`, `evaluate_ideal_ports`, and
   `compute_wilson_interval` convert selected port sets into outage metrics.
7. Result-row builders:
   `build_result_rows` and `build_canonical_rows` format all per-budget
   outputs so the downstream analysis script can read them without needing to
   guess field names.
8. Resume helpers:
   `list_completed_repetitions`, `repetition_result_path`, and
   `aggregate_result_rows_from_run` support restartable Monte Carlo execution
   and later aggregation.

Why the binary representation is valid:
- For observed-only outage, a row is successful if at least one selected port
  exceeds `threshold / snr_linear`.
- Because that event depends only on whether each selected port is above or
  below the threshold, the float-valued channel matrix can be reduced to a
  boolean exceedance matrix without changing the observed-only outage result.
- This makes greedy search much cheaper while staying exact for the metric
  used in the project.

Example usage:
```python
from policy_mc_kms_common import (
    build_mar_order,
    build_gc_order,
    build_rag_order,
    load_kms_regime,
    threshold_to_binary,
)

regime = load_kms_regime("kappa0_mu1_m50", common_path=None)
binary = threshold_to_binary(regime, threshold=0.65217584, snr_linear=1.0)
means = regime.mean(axis=0)
order_mar, scores_mar = build_mar_order(binary, means)
order_gc = build_gc_order(binary, means, scores_mar, max_budget=7)
order_rag = build_rag_order(binary, means, scores_mar, max_budget=7)
```

This module is intentionally non-executable on its own. It is a shared utility
layer and should be imported by the simulation and analysis scripts.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/policy_mc_kms_common.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from utils.data import DEFAULT_DATASET_ROOT_PATH


# -----------------------------------------------------------------------------
# Common simulation configuration
# -----------------------------------------------------------------------------
COMMON_DATASET_PATH = DEFAULT_DATASET_ROOT_PATH
RUNS_ROOT = _COMPANION_ROOT / 'studies/chapter07_reading_policies/results'
DATA_SEED = 0
THRESHOLD = [0.37673008,] # [0.37673008, 0.65217584, 0.85472679]
SNR_LINEAR = 1.0
TOTAL_PORTS = 100
TARGET_PORTS = (3, 4, 5, 6, 7)
NUM_MC_REPS = 100
MC_EVAL_FRACTION = 0.20


KMS_REGIME_FILES: tuple[tuple[str, str], ...] = (
    ("kappa0_mu1_m0", "SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m0.0.mat"),
    ("kappa0_mu1_m2", "SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m2.0.mat"),
    ("kappa0_mu1_m50", "SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m50.0.mat"),
    ("kappa0_mu2_m50", "SNR_events_W1.0_U1_N100_kappa1.0e-16_mu2.0_m50.0.mat"),
    ("kappa0_mu5_m50", "SNR_events_W1.0_U1_N100_kappa1.0e-16_mu5.0_m50.0.mat"),
    ("kappa5_mu1_m0", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu1.0_m0.0.mat"),
    ("kappa5_mu1_m2", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu1.0_m2.0.mat"),
    ("kappa5_mu1_m50", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu1.0_m50.0.mat"),
    ("kappa5_mu2_m0", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu2.0_m0.0.mat"),
    ("kappa5_mu2_m2", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu2.0_m2.0.mat"),
    ("kappa5_mu2_m50", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu2.0_m50.0.mat"),
    ("kappa5_mu5_m0", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu5.0_m0.0.mat"),
    ("kappa5_mu5_m2", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu5.0_m2.0.mat"),
    ("kappa5_mu5_m50", "SNR_events_W1.0_U1_N100_kappa5.0e+00_mu5.0_m50.0.mat"),
)
KMS_REGIME_LABELS: tuple[str, ...] = tuple(label for label, _ in KMS_REGIME_FILES)
RESULT_FIELDNAMES: tuple[str, ...] = (
    "policy",
    "dataset",
    "regime_index",
    "rep",
    "base_seed",
    "n_ports",
    "design_size",
    "eval_size",
    "selection_order_0based",
    "sorted_pattern_0based",
    "selection_order_1based",
    "sorted_pattern_1based",
    "observed_op",
    "ideal_op",
    "gap_to_ideal",
    "n_eval",
    "n_outage",
    "n_ideal_outage",
    "wilson_low",
    "wilson_high",
)
CANONICAL_FIELDNAMES: tuple[str, ...] = (
    "policy",
    "dataset",
    "regime_index",
    "n_ports",
    "selection_order_0based",
    "sorted_pattern_0based",
    "selection_order_1based",
    "sorted_pattern_1based",
)


def log_status(script_name: str, section: str, message: str) -> None:
    """Print a standardized runtime message for the calling script.

    Args:
        script_name (str): Short script identifier used to namespace logs. If
            this value changes, grepping historical outputs by script becomes
            harder. Keeping it stable improves resumability audits.
        section (str): Small block label that indicates which stage produced
            the message. Concise sections make long runs easier to scan.
        message (str): Human-readable description of current progress or a
            noteworthy state transition.

    Returns:
        None: The function writes one formatted line to stdout.

    Raises:
        OSError: If stdout cannot be written.
    """
    print(f"[{script_name}:{section}] {message}", flush=True)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path.

    Args:
        path (str | Path): Directory path to create.
            If the directory already exists, this function leaves it in place.
            If it does not exist, the full parent chain is created.

    Returns:
        Path: Resolved directory path.

    Raises:
        OSError: If the directory cannot be created.
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON document atomically using a temporary sibling file.

    Args:
        path (str | Path): Final JSON destination.
            If the target exists, it is replaced atomically after the temporary
            file is fully written.
            If the target does not exist, it is created through the same
            temporary-then-rename workflow to avoid partial files after crashes.
        payload (dict[str, Any]): JSON-serializable dictionary to persist.

    Returns:
        Path: Resolved final output path.

    Raises:
        OSError: If the temporary file or rename cannot be written.
        TypeError: If ``payload`` is not JSON serializable.
    """
    output_path = Path(path).resolve()
    ensure_directory(output_path.parent)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    tmp_path.replace(output_path)
    return output_path


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a Python dictionary.

    Args:
        path (str | Path): JSON file to read.

    Returns:
        dict[str, Any]: Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is malformed JSON.
        OSError: If the file cannot be read.
    """
    with open(Path(path).resolve(), "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_matching_run_config(
    run_dir: str | Path,
    config: dict[str, Any],
    ignore_keys: frozenset[str] = frozenset(),
) -> Path:
    """Persist one run configuration and reject incompatible reruns.

    Args:
        run_dir (str | Path): Root run directory that should contain exactly
            one configuration file for the experiment.
        config (dict[str, Any]): Expected configuration for the current run.
            If no configuration exists yet, this dictionary is written.
            If a configuration already exists, it must match exactly to avoid
            mixing outputs generated under different simulation parameters.
        ignore_keys (frozenset[str]): Keys to exclude from the comparison.
            Use this for machine-local paths (e.g. ``common_dataset_path``,
            ``analysis_output_dir``) that may differ across machines without
            affecting the reproducibility of simulation results. These keys
            are still written to the file on first creation but are not
            checked on subsequent runs.

    Returns:
        Path: Path to the persisted ``run_config.json`` file.

    Raises:
        ValueError: If an existing configuration differs from ``config`` on
            any key not in ``ignore_keys``.
        OSError: If the configuration file cannot be written.
    """
    run_dir_obj = ensure_directory(run_dir)
    config_path = run_dir_obj / "run_config.json"
    if config_path.exists():
        existing_config = read_json_file(config_path)
        # Normalize both dicts via a JSON round-trip so that Python tuples
        # (which serialize as JSON arrays) compare equal to lists read back
        # from disk. Without this, e.g. (3, 4, 5) != [3, 4, 5] would always
        # trigger a false mismatch on tuple-valued config entries.
        def _json_normalize(d: dict) -> dict:
            return json.loads(json.dumps(d))

        comparable_existing = _json_normalize({k: v for k, v in existing_config.items() if k not in ignore_keys})
        comparable_current = _json_normalize({k: v for k, v in config.items() if k not in ignore_keys})
        if comparable_existing != comparable_current:
            raise ValueError(
                "Existing run_config.json does not match the current configuration. "
                f"Delete {config_path} or choose a different run directory."
            )
        return config_path
    return write_json_atomic(config_path, config)


def write_csv_rows_atomic(
    path: str | Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> Path:
    """Write CSV rows atomically to reduce restart corruption risk.

    Args:
        path (str | Path): Final CSV destination.
            If the file already exists, it is fully replaced after the
            temporary file is written successfully.
            If it does not exist, it is created atomically in the same way.
        fieldnames (tuple[str, ...]): Ordered CSV header names used for both
            the header row and all emitted dictionaries.
        rows (list[dict[str, Any]]): Sequence of dictionaries whose keys match
            ``fieldnames``. Missing keys are written as empty fields by
            ``csv.DictWriter``; extra keys are ignored.

    Returns:
        Path: Resolved final CSV path.

    Raises:
        OSError: If the temporary file cannot be written or renamed.
        csv.Error: If CSV serialization fails.
    """
    output_path = Path(path).resolve()
    ensure_directory(output_path.parent)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(output_path)
    return output_path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read all rows from a CSV file.

    Args:
        path (str | Path): CSV file path.

    Returns:
        list[dict[str, str]]: Parsed rows with string values exactly as read
            from the CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        OSError: If the file cannot be read.
        csv.Error: If CSV parsing fails.
    """
    with open(Path(path).resolve(), "r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def regime_index_from_label(regime_label: str) -> int:
    """Return the canonical KMS regime index for one regime label.

    Args:
        regime_label (str): Regime name such as ``"kappa0_mu1_m0"``.

    Returns:
        int: Zero-based regime index in the fixed KMS ordering.

    Raises:
        ValueError: If ``regime_label`` is not part of the known KMS list.
    """
    if regime_label not in KMS_REGIME_LABELS:
        raise ValueError(f"Unknown KMS regime label: {regime_label}")
    return int(KMS_REGIME_LABELS.index(regime_label))


def format_threshold_folder(threshold: float) -> str:
    """Format one threshold value as the canonical result-folder name.

    Args:
        threshold (float): Outage threshold in linear units. Values are written
            with eight digits after the decimal point because the configured
            multi-threshold study uses eight-decimal threshold values.

    Returns:
        str: Folder name such as ``"threshold_0.37673008"``.

    Raises:
        ValueError: If ``threshold`` is not finite.

    Examples:
        >>> format_threshold_folder(0.37673008)
        'threshold_0.37673008'
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return f"threshold_{threshold:.8f}"


def load_kms_regime(regime_label: str, common_path: str | None) -> np.ndarray:
    """Load one KMS regime matrix without materializing the other 13 regimes.

    Args:
        regime_label (str): One of the labels listed in ``KMS_REGIME_LABELS``.
            Each label maps to exactly one MATLAB file.
        common_path (str | None): Base directory that contains the KMS files.
            If ``None``, ``DEFAULT_DATASET_ROOT_PATH`` is used.
            If a custom string is provided, files are resolved relative to it.

    Returns:
        np.ndarray: Loaded regime matrix cast to ``float32`` with shape
            ``(n_samples, n_ports)``.

    Raises:
        ValueError: If ``regime_label`` is unknown or the loaded array is not
            two-dimensional.
        FileNotFoundError: If the expected file does not exist.
        KeyError: If the MATLAB file does not contain ``SNR_events``.
        OSError: If SciPy cannot read the file.
    """
    dataset_root = DEFAULT_DATASET_ROOT_PATH if common_path is None else str(common_path)
    file_lookup = dict(KMS_REGIME_FILES)
    if regime_label not in file_lookup:
        raise ValueError(f"Unknown KMS regime label: {regime_label}")

    file_path = Path(dataset_root) / file_lookup[regime_label]
    if not file_path.exists():
        raise FileNotFoundError(f"KMS regime file not found: {file_path}")

    regime_data = scipy.io.loadmat(str(file_path))["SNR_events"].astype(np.float32, copy=False)
    if regime_data.ndim != 2:
        raise ValueError(f"Loaded KMS regime must be 2D, got shape {regime_data.shape} for {regime_label}.")
    return regime_data


def build_rep_rng(base_seed: int, regime_index: int, rep_index: int) -> np.random.Generator:
    """Build a deterministic RNG for one regime/repetition pair.

    Args:
        base_seed (int): Experiment-level seed shared by all repetitions.
        regime_index (int): Zero-based regime identifier.
        rep_index (int): Zero-based Monte Carlo repetition identifier.

    Returns:
        np.random.Generator: RNG seeded from the three input integers.

    Raises:
        ValueError: If ``regime_index`` or ``rep_index`` is negative.
    """
    if regime_index < 0 or rep_index < 0:
        raise ValueError("regime_index and rep_index must be non-negative.")
    seed_sequence = np.random.SeedSequence([int(base_seed), int(regime_index), int(rep_index)])
    return np.random.default_rng(seed_sequence)


def split_design_eval_indices(
    n_samples: int,
    eval_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one regime into design and evaluation row indices.

    Args:
        n_samples (int): Total number of rows in the regime.
        eval_fraction (float): Fraction reserved for evaluation.
            If this value is near ``0.2``, the split is approximately 80/20.
            Larger fractions reserve more evaluation rows and fewer design rows.
            Smaller fractions do the opposite. The value must be in ``(0, 1)``.
        rng (np.random.Generator): RNG used to generate a row permutation.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(design_indices, eval_indices)`` where
            both arrays are one-dimensional integer index arrays.

    Raises:
        ValueError: If ``n_samples`` is too small or ``eval_fraction`` is out
            of range.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 to form design and evaluation splits.")
    if not (0.0 < eval_fraction < 1.0):
        raise ValueError("eval_fraction must be in the open interval (0, 1).")

    eval_count = int(n_samples * eval_fraction)
    eval_count = max(1, min(n_samples - 1, eval_count))
    permutation = rng.permutation(n_samples)
    design_indices = permutation[:-eval_count]
    eval_indices = permutation[-eval_count:]
    return design_indices, eval_indices


def threshold_to_binary(
    values: np.ndarray,
    threshold: float,
    snr_linear: float,
) -> np.ndarray:
    """Convert one float-valued channel matrix into a binary exceedance matrix.

    Args:
        values (np.ndarray): Float matrix with shape ``(n_samples, n_ports)``.
        threshold (float): Outage threshold numerator in the linear domain.
        snr_linear (float): Linear SNR denominator.
            If this value is ``1.0``, the cutoff is exactly ``threshold``.
            If it is larger, the exceedance cutoff becomes lower and more rows
            are marked as successful. If it is smaller, the cutoff becomes
            stricter and fewer rows exceed it.

    Returns:
        np.ndarray: Boolean matrix of the same shape as ``values`` where
            ``True`` marks ports above ``threshold / snr_linear``.

    Raises:
        ValueError: If ``snr_linear`` is not positive.
    """
    if snr_linear <= 0.0:
        raise ValueError("snr_linear must be strictly positive.")
    return values > (threshold / snr_linear)


def build_uni_indices(total_ports: int, n_ports: int) -> np.ndarray:
    """Select evenly spaced port indices for the geometric baseline.

    Args:
        total_ports (int): Total number of available ports.
        n_ports (int): Observation budget.
            If this value is larger, the baseline samples the aperture more
            densely. If it is smaller, the baseline keeps only a coarse spatial
            coverage. The value must satisfy ``1 <= n_ports <= total_ports``.

    Returns:
        np.ndarray: Sorted zero-based port indices of length ``n_ports``.

    Raises:
        ValueError: If ``n_ports`` is outside the valid range.
    """
    if n_ports < 1 or n_ports > total_ports:
        raise ValueError("n_ports must satisfy 1 <= n_ports <= total_ports.")
    return np.linspace(0, total_ports - 1, n_ports, dtype=int)


def build_mar_order(design_binary: np.ndarray, design_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank ports by regime-specific exceedance probability with deterministic ties.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix on the design
            split with shape ``(n_design_samples, n_ports)``.
        design_means (np.ndarray): Mean port value on the same design split.
            Higher means are used only as a tie-break after exceedance
            probability.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(ranked_indices, exceedance_scores)`` where ``ranked_indices`` is
            the full zero-based order from best to worst and
            ``exceedance_scores`` contains the empirical exceedance frequency of
            every port.

    Raises:
        ValueError: If the inputs do not agree on the number of ports.
    """
    if design_binary.ndim != 2:
        raise ValueError("design_binary must be a 2D matrix.")
    if design_means.ndim != 1 or design_means.shape[0] != design_binary.shape[1]:
        raise ValueError("design_means must be a 1D vector aligned with design_binary columns.")

    exceedance_scores = np.mean(design_binary, axis=0, dtype=np.float64)

    # Rank by the task-aligned exceedance score first, then by mean value, and
    # finally by smaller index to keep reruns deterministic.
    ranked_indices = np.lexsort((np.arange(design_binary.shape[1]), -design_means, -exceedance_scores))
    return ranked_indices.astype(int, copy=False), exceedance_scores


def build_gc_order(
    design_binary: np.ndarray,
    design_means: np.ndarray,
    mar_scores: np.ndarray,
    max_budget: int,
) -> np.ndarray:
    """Construct a greedy nested port order that minimizes observed-only outage.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix with shape
            ``(n_design_samples, n_ports)``.
        design_means (np.ndarray): Per-port means on the design split used as a
            deterministic tie-break after task-specific metrics.
        mar_scores (np.ndarray): Per-port exceedance frequencies from the
            same design split. These are used as the first tie-break when
            multiple candidate additions reduce outage equally.
        max_budget (int): Number of greedy steps to run.
            If this value is larger, the function produces a longer nested
            ordering. If all design rows become covered before reaching this
            length, the remaining slots are filled using the MAR ranking
            among unselected ports.

    Returns:
        np.ndarray: Zero-based greedy order of length ``max_budget``.

    Raises:
        ValueError: If shapes are inconsistent or ``max_budget`` is invalid.
    """
    if design_binary.ndim != 2:
        raise ValueError("design_binary must be 2D.")
    if max_budget < 1 or max_budget > design_binary.shape[1]:
        raise ValueError("max_budget must satisfy 1 <= max_budget <= n_ports.")
    if design_means.shape != mar_scores.shape or design_means.shape[0] != design_binary.shape[1]:
        raise ValueError("design_means and mar_scores must match the number of ports.")

    ranked_mar, _ = build_mar_order(design_binary=design_binary, design_means=design_means)
    selected_mask = np.zeros(design_binary.shape[1], dtype=bool)
    current_covered = np.zeros(design_binary.shape[0], dtype=bool)
    greedy_order = np.empty(max_budget, dtype=int)
    next_fill_position = 0

    # Greedily cover the currently uncovered outage rows. Because the outage
    # indicator only cares whether at least one selected port exceeds the
    # threshold, maximizing newly covered rows is exactly equivalent to
    # minimizing observed-only outage at each step.
    while next_fill_position < max_budget:
        uncovered_mask = ~current_covered

        # If every row is already covered, continue with MAR's ranking so
        # the remaining order stays deterministic and nested.
        if not np.any(uncovered_mask):
            for port_index in ranked_mar:
                if selected_mask[port_index]:
                    continue
                greedy_order[next_fill_position] = int(port_index)
                selected_mask[port_index] = True
                next_fill_position += 1
                if next_fill_position == max_budget:
                    break
            break

        candidate_cover_counts = np.sum(design_binary[uncovered_mask], axis=0, dtype=np.int64)
        candidate_cover_counts[selected_mask] = -1
        best_cover_count = int(np.max(candidate_cover_counts))
        candidate_indices = np.flatnonzero(candidate_cover_counts == best_cover_count)
        sorted_candidates = np.lexsort(
            (
                candidate_indices,
                -design_means[candidate_indices],
                -mar_scores[candidate_indices],
            )
        )
        best_port = int(candidate_indices[sorted_candidates[0]])

        greedy_order[next_fill_position] = best_port
        selected_mask[best_port] = True
        current_covered |= design_binary[:, best_port]
        next_fill_position += 1

    return greedy_order


def score_spread_pattern(
    selected_indices: np.ndarray,
    design_means: np.ndarray,
) -> tuple[int, float, tuple[int, ...]]:
    """Build the deterministic spread-aware tie-break key for one pattern.

    Args:
        selected_indices (np.ndarray): One-dimensional zero-based port indices
            in the candidate pattern. The array may be ordered by selection
            history; this function sorts it before scoring because SUNI and
            SWUNI compare unordered read sets.
        design_means (np.ndarray): Per-port mean SNR values from the design
            split. Higher summed mean SNR is preferred only after the outage
            objective and aperture span tie.

    Returns:
        tuple[int, float, tuple[int, ...]]: Tie-break key containing aperture
            span, summed design mean, and negated sorted indices. Larger tuple
            values are preferred by callers.

    Raises:
        ValueError: If ``selected_indices`` is not one-dimensional, empty, or
            not aligned with ``design_means``.
    """
    if selected_indices.ndim != 1 or selected_indices.shape[0] == 0:
        raise ValueError("selected_indices must be a non-empty 1D array.")
    if design_means.ndim != 1:
        raise ValueError("design_means must be a 1D vector.")
    if np.min(selected_indices) < 0 or np.max(selected_indices) >= design_means.shape[0]:
        raise ValueError("selected_indices must be valid indices into design_means.")

    sorted_indices = np.sort(selected_indices.astype(int, copy=False))
    span = int(sorted_indices[-1] - sorted_indices[0])
    mean_sum = float(np.sum(design_means[sorted_indices], dtype=np.float64))
    lower_index_key = tuple(-int(index) for index in sorted_indices.tolist())
    return span, mean_sum, lower_index_key


def count_selected_outages(design_binary: np.ndarray, selected_indices: np.ndarray) -> int:
    """Count design-split outage rows for a selected port set.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix with shape
            ``(n_design_samples, n_ports)``.
        selected_indices (np.ndarray): One-dimensional zero-based selected
            port indices. At least one port must be selected.

    Returns:
        int: Number of design rows where none of the selected ports exceed the
            threshold.

    Raises:
        ValueError: If ``design_binary`` is empty or if ``selected_indices`` is
            not a non-empty one-dimensional array.
    """
    if design_binary.ndim != 2 or design_binary.shape[0] == 0:
        raise ValueError("design_binary must be a non-empty 2D matrix.")
    if selected_indices.ndim != 1 or selected_indices.shape[0] == 0:
        raise ValueError("selected_indices must be a non-empty 1D array.")
    if np.min(selected_indices) < 0 or np.max(selected_indices) >= design_binary.shape[1]:
        raise ValueError("selected_indices must be valid column indices for design_binary.")

    selected_success = np.any(design_binary[:, selected_indices], axis=1)
    return int(design_binary.shape[0] - np.count_nonzero(selected_success))


def build_suni_candidate_patterns(total_ports: int, n_ports: int) -> list[np.ndarray]:
    """Generate inward-shifted uniform candidate patterns.

    Args:
        total_ports (int): Total number of available ports in the aperture.
        n_ports (int): Observation budget. For ``n_ports > 1``, each candidate
            places the uniform grid between endpoints shifted inward by the
            same integer amount. For ``n_ports == 1``, each single-port
            candidate is tested from left to right.

    Returns:
        list[np.ndarray]: Candidate zero-based patterns in outward-to-inward
            generation order.

    Raises:
        ValueError: If ``n_ports`` is outside ``[1, total_ports]``.
    """
    if n_ports < 1 or n_ports > total_ports:
        raise ValueError("n_ports must satisfy 1 <= n_ports <= total_ports.")

    if n_ports == 1:
        return [np.array([port_index], dtype=int) for port_index in range(total_ports)]

    candidates: list[np.ndarray] = []
    seen_patterns: set[tuple[int, ...]] = set()
    for shift in range(total_ports):
        left = shift
        right = total_ports - 1 - shift
        if left >= right:
            break

        candidate = np.linspace(left, right, n_ports, dtype=int)
        if candidate.shape[0] != n_ports or np.any(np.diff(candidate) <= 0):
            break

        candidate_key = tuple(int(index) for index in candidate.tolist())
        if candidate_key in seen_patterns:
            continue
        seen_patterns.add(candidate_key)
        candidates.append(candidate.astype(int, copy=False))

    if len(candidates) == 0:
        raise ValueError("No valid SUNI candidates were generated.")
    return candidates


def build_suni_indices(
    design_binary: np.ndarray,
    design_means: np.ndarray,
    n_ports: int,
) -> np.ndarray:
    """Select the best inward-shifted uniform pattern on the design split.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix with shape
            ``(n_design_samples, total_ports)``.
        design_means (np.ndarray): Per-port mean SNR values from the same
            design split. These values are used only for deterministic
            tie-breaking after outage and aperture span.
        n_ports (int): Observation budget for the SUNI pattern.

    Returns:
        np.ndarray: Zero-based selected port indices of length ``n_ports``.

    Raises:
        ValueError: If inputs are misaligned or ``n_ports`` is invalid.

    Examples:
        >>> import numpy as np
        >>> binary = np.array([[True, False, False], [False, True, False]])
        >>> means = np.array([1.0, 2.0, 0.5])
        >>> build_suni_indices(binary, means, n_ports=1).tolist()
        [1]
    """
    if design_binary.ndim != 2 or design_binary.shape[0] == 0:
        raise ValueError("design_binary must be a non-empty 2D matrix.")
    if design_means.ndim != 1 or design_means.shape[0] != design_binary.shape[1]:
        raise ValueError("design_means must be a 1D vector aligned with design_binary columns.")
    if n_ports < 1 or n_ports > design_binary.shape[1]:
        raise ValueError("n_ports must satisfy 1 <= n_ports <= total_ports.")

    best_candidate: np.ndarray | None = None
    best_key: tuple[int, int, float, tuple[int, ...]] | None = None
    for candidate in build_suni_candidate_patterns(total_ports=design_binary.shape[1], n_ports=n_ports):
        n_outage = count_selected_outages(design_binary=design_binary, selected_indices=candidate)
        span, mean_sum, lower_index_key = score_spread_pattern(candidate, design_means)
        candidate_key = (-n_outage, span, mean_sum, lower_index_key)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError("SUNI requires at least one candidate pattern.")
    return best_candidate.astype(int, copy=True)


def build_swuni_indices(
    design_binary: np.ndarray,
    design_means: np.ndarray,
    n_ports: int,
) -> np.ndarray:
    """Improve the UNI pattern with strict one-port outage-reducing swaps.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix with shape
            ``(n_design_samples, total_ports)``.
        design_means (np.ndarray): Per-port mean SNR values from the same
            design split. These values are used only after outage and aperture
            span tie.
        n_ports (int): Observation budget. The policy starts from the UNI
            selection for this budget and keeps testing one-port replacements.

    Returns:
        np.ndarray: Sorted zero-based SWUNI pattern after no one-port swap can
            strictly reduce the design outage.

    Raises:
        ValueError: If inputs are misaligned or ``n_ports`` is invalid.
    """
    if design_binary.ndim != 2 or design_binary.shape[0] == 0:
        raise ValueError("design_binary must be a non-empty 2D matrix.")
    if design_means.ndim != 1 or design_means.shape[0] != design_binary.shape[1]:
        raise ValueError("design_means must be a 1D vector aligned with design_binary columns.")
    if n_ports < 1 or n_ports > design_binary.shape[1]:
        raise ValueError("n_ports must satisfy 1 <= n_ports <= total_ports.")

    selected = build_uni_indices(total_ports=design_binary.shape[1], n_ports=n_ports).astype(int, copy=True)
    total_ports = int(design_binary.shape[1])

    while True:
        current_outage = count_selected_outages(design_binary=design_binary, selected_indices=selected)
        selected_set = {int(index) for index in selected.tolist()}
        best_candidate: np.ndarray | None = None
        best_key: tuple[int, int, float, tuple[int, ...]] | None = None

        for selected_port in sorted(selected_set):
            for replacement_port in range(total_ports):
                if replacement_port in selected_set:
                    continue
                candidate_values = sorted((selected_set - {selected_port}) | {replacement_port})
                candidate = np.array(candidate_values, dtype=int)
                n_outage = count_selected_outages(design_binary=design_binary, selected_indices=candidate)
                if n_outage >= current_outage:
                    continue

                span, mean_sum, lower_index_key = score_spread_pattern(candidate, design_means)
                candidate_key = (-n_outage, span, mean_sum, lower_index_key)
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_candidate = candidate

        if best_candidate is None:
            return np.sort(selected).astype(int, copy=False)
        selected = best_candidate


def build_rag_order(
    design_binary: np.ndarray,
    design_means: np.ndarray,
    mar_scores: np.ndarray,
    max_budget: int,
    redundancy_weight: float = 0.1,
) -> np.ndarray:
    """Construct a redundancy-aware greedy port order.

    Args:
        design_binary (np.ndarray): Boolean exceedance matrix with shape
            ``(n_design_samples, n_ports)``.
        design_means (np.ndarray): Per-port design means used as a late
            deterministic tie-break.
        mar_scores (np.ndarray): Per-port exceedance frequencies from the
            design split, used after the RAG score and new coverage tie.
        max_budget (int): Number of ordered ports to return.
        redundancy_weight (float): Non-negative multiplier for the summed
            Jaccard redundancy penalty applied to coverage rates. Useful
            candidate values are ``0.01``, ``0.05``, ``0.1``, and ``0.2``;
            the configured policy starts at ``0.1``.

    Returns:
        np.ndarray: Zero-based RAG selection order of length ``max_budget``.

    Raises:
        ValueError: If shapes are inconsistent, ``max_budget`` is invalid, or
            ``redundancy_weight`` is negative.
    """
    if design_binary.ndim != 2:
        raise ValueError("design_binary must be 2D.")
    if max_budget < 1 or max_budget > design_binary.shape[1]:
        raise ValueError("max_budget must satisfy 1 <= max_budget <= n_ports.")
    if design_means.shape != mar_scores.shape or design_means.shape[0] != design_binary.shape[1]:
        raise ValueError("design_means and mar_scores must match the number of ports.")
    if redundancy_weight < 0.0:
        raise ValueError("redundancy_weight must be non-negative.")

    ranked_mar, _ = build_mar_order(design_binary=design_binary, design_means=design_means)
    selected_mask = np.zeros(design_binary.shape[1], dtype=bool)
    current_covered = np.zeros(design_binary.shape[0], dtype=bool)
    selected_ports: list[int] = []
    rag_order = np.empty(max_budget, dtype=int)

    # RAG keeps GC's coverage target, but scores new coverage as a rate before
    # subtracting Jaccard redundancy. This keeps lambda meaningful when design
    # splits contain many samples.
    for position in range(max_budget):
        if np.all(current_covered):
            for port_index in ranked_mar:
                if selected_mask[port_index]:
                    continue
                rag_order[position] = int(port_index)
                selected_mask[port_index] = True
                selected_ports.append(int(port_index))
                current_covered |= design_binary[:, port_index]
                break
            continue

        best_port: int | None = None
        best_key: tuple[float, float, int, float, float, int] | None = None
        uncovered_mask = ~current_covered
        for candidate_port in range(design_binary.shape[1]):
            if selected_mask[candidate_port]:
                continue

            new_coverage = int(np.count_nonzero(design_binary[uncovered_mask, candidate_port]))
            new_coverage_rate = float(new_coverage / design_binary.shape[0])
            redundancy_sum = 0.0
            candidate_success = design_binary[:, candidate_port]
            for selected_port in selected_ports:
                selected_success = design_binary[:, selected_port]
                intersection = int(np.count_nonzero(candidate_success & selected_success))
                union = int(np.count_nonzero(candidate_success | selected_success))
                if union > 0:
                    redundancy_sum += intersection / union

            score = float(new_coverage_rate - (redundancy_weight * redundancy_sum))
            candidate_key = (
                score,
                new_coverage_rate,
                new_coverage,
                float(mar_scores[candidate_port]),
                float(design_means[candidate_port]),
                -candidate_port,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_port = candidate_port

        if best_port is None:
            raise ValueError("RAG could not find an unselected candidate port.")
        rag_order[position] = int(best_port)
        selected_mask[best_port] = True
        selected_ports.append(int(best_port))
        current_covered |= design_binary[:, best_port]

    return rag_order


def evaluate_selected_ports(binary_eval: np.ndarray, selected_indices: np.ndarray) -> tuple[float, int, int]:
    """Estimate observed-only outage for one selected port set on evaluation rows.

    Args:
        binary_eval (np.ndarray): Boolean exceedance matrix on the evaluation
            split with shape ``(n_eval_samples, n_ports)``.
        selected_indices (np.ndarray): Zero-based selected port indices.
            The order of indices does not change the outage value because the
            metric uses an ``any`` condition over the selected set.

    Returns:
        tuple[float, int, int]:
            ``(observed_op, n_eval, n_outage)`` where ``observed_op`` is in
            ``[0, 1]``.

    Raises:
        ValueError: If ``binary_eval`` is empty or not 2D.
    """
    if binary_eval.ndim != 2 or binary_eval.shape[0] == 0:
        raise ValueError("binary_eval must be a non-empty 2D matrix.")
    selected_success = np.any(binary_eval[:, selected_indices], axis=1)
    n_eval = int(binary_eval.shape[0])
    n_outage = int(n_eval - np.count_nonzero(selected_success))
    observed_op = float(n_outage / n_eval)
    return observed_op, n_eval, n_outage


def evaluate_ideal_ports(binary_eval: np.ndarray) -> tuple[float, int]:
    """Compute the ideal outage probability assuming all ports are observable.

    Args:
        binary_eval (np.ndarray): Boolean exceedance matrix on the evaluation
            split with shape ``(n_eval_samples, n_ports)``.

    Returns:
        tuple[float, int]:
            ``(ideal_op, n_ideal_outage)`` where ``ideal_op`` is the fraction
            of rows whose full-port maximum still fails the threshold.

    Raises:
        ValueError: If ``binary_eval`` is empty or not 2D.
    """
    if binary_eval.ndim != 2 or binary_eval.shape[0] == 0:
        raise ValueError("binary_eval must be a non-empty 2D matrix.")
    ideal_success = np.any(binary_eval, axis=1)
    n_eval = int(binary_eval.shape[0])
    n_ideal_outage = int(n_eval - np.count_nonzero(ideal_success))
    ideal_op = float(n_ideal_outage / n_eval)
    return ideal_op, n_ideal_outage


def compute_wilson_interval(n_outage: int, n_total: int, z_value: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson confidence interval for a Bernoulli outage probability.

    Args:
        n_outage (int): Number of outage events.
        n_total (int): Total number of Bernoulli trials.
            If this value is larger, the interval narrows. If it is smaller,
            uncertainty widens. The value must be strictly positive.
        z_value (float): Standard-normal critical value.
            ``1.96`` corresponds approximately to a 95% confidence interval.
            Larger values widen the interval; smaller values narrow it.

    Returns:
        tuple[float, float]: Lower and upper Wilson interval bounds.

    Raises:
        ValueError: If ``n_total`` is not positive or ``n_outage`` is outside
            ``[0, n_total]``.
    """
    if n_total <= 0:
        raise ValueError("n_total must be strictly positive.")
    if n_outage < 0 or n_outage > n_total:
        raise ValueError("n_outage must satisfy 0 <= n_outage <= n_total.")

    proportion = n_outage / n_total
    denominator = 1.0 + ((z_value**2) / n_total)
    center = (proportion + ((z_value**2) / (2.0 * n_total))) / denominator
    margin = (
        z_value
        * math.sqrt((proportion * (1.0 - proportion) / n_total) + ((z_value**2) / (4.0 * (n_total**2))))
        / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def format_indices_one_based(indices: np.ndarray) -> str:
    """Format zero-based indices as a compact one-based comma-separated string.

    Args:
        indices (np.ndarray): Zero-based integer indices.

    Returns:
        str: Comma-separated one-based indices such as ``"1,5,10"``.

    Raises:
        ValueError: If ``indices`` is not one-dimensional.
    """
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    return ",".join(str(int(index) + 1) for index in indices.tolist())


def format_indices_zero_based(indices: np.ndarray) -> str:
    """Format zero-based indices as a bracketed comma-separated list.

    Args:
        indices (np.ndarray): One-dimensional zero-based integer index array.
            Values are written exactly as zero-based port numbers. The function
            does not sort the input because callers use it for both ordered
            selections and sorted audit patterns.

    Returns:
        str: Bracketed zero-based list such as ``"[0,49,99]"``.

    Raises:
        ValueError: If ``indices`` is not one-dimensional.
    """
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    return "[" + ",".join(str(int(index)) for index in indices.tolist()) + "]"


def format_index_sets_zero_based(index_sets: list[np.ndarray]) -> str:
    """Format one or more zero-based index arrays for CSV output.

    Args:
        index_sets (list[np.ndarray]): Selection arrays to serialize. Each
            array must be one-dimensional. Multiple arrays represent repeated
            RND draws for an averaged RND baseline and are separated by
            semicolons in the returned string.

    Returns:
        str: One formatted index list, or several lists separated by
            semicolons, such as ``"[0,49,99];[2,11,80]"``.

    Raises:
        ValueError: If ``index_sets`` is empty or any element is not
            one-dimensional.
    """
    if len(index_sets) == 0:
        raise ValueError("index_sets must contain at least one selection.")
    return ";".join(format_indices_zero_based(indices) for indices in index_sets)


def simulate_rnd_policy_result(
    binary_eval: np.ndarray,
    n_ports: int,
    n_times: int,
    rng: np.random.Generator,
) -> tuple[float, int, int, list[np.ndarray]]:
    """Evaluate repeated RND port draws as one averaged RND baseline.

    The RND policy picks ``n_ports`` ports uniformly at random (without
    replacement) from all available ports and evaluates the observed-only outage
    on the full evaluation split. This is repeated ``n_times`` independently.
    The returned counts treat those repeated evaluations as the trials used by
    the averaged estimator, so confidence intervals are computed from
    ``n_eval * n_times`` binary sample-draw outcomes instead of a rounded mean.

    Using ``n_times`` > 1 reduces Monte Carlo noise in the RND baseline: a
    single draw can be lucky or unlucky, so averaging across several draws gives
    a more stable estimate of the expected OP under random selection.

    Args:
        binary_eval (np.ndarray): Boolean exceedance matrix on the evaluation
            split with shape ``(n_eval_samples, total_ports)``. Each entry
            is ``True`` if the corresponding port exceeds the threshold.
        n_ports (int): Observation budget (number of ports to draw per trial).
            Must satisfy ``1 <= n_ports <= total_ports``.
        n_times (int): Number of independent random draws to average.
            If ``1``, a single draw is made. If ``3`` or ``10``, the average
            across that many independent draws is returned. Higher values
            reduce variance in the estimate at the cost of more computation.
        rng (np.random.Generator): RNG used for all random draws. Passing the
            same seeded generator on repeated calls ensures reproducibility.

    Returns:
        tuple[float, int, int, list[np.ndarray]]:
            ``(observed_op, n_eval_total, n_outage_total, selected_draws)``.
            ``observed_op`` is the averaged outage probability,
            ``n_eval_total`` is ``binary_eval.shape[0] * n_times``,
            ``n_outage_total`` is the summed outage count across all draws, and
            ``selected_draws`` records the selected zero-based port indices for
            each draw in draw order.

    Raises:
        ValueError: If ``binary_eval`` is not 2D, ``n_ports`` is out of range,
            or ``n_times`` is less than 1.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> binary = np.array([[True, False, True], [False, False, False]])
        >>> result = simulate_rnd_policy_result(binary, n_ports=1, n_times=3, rng=rng)
        >>> result[1]
        6
    """
    if binary_eval.ndim != 2 or binary_eval.shape[0] == 0:
        raise ValueError("binary_eval must be a non-empty 2D matrix.")
    total_ports = binary_eval.shape[1]
    if n_ports < 1 or n_ports > total_ports:
        raise ValueError(f"n_ports must satisfy 1 <= n_ports <= {total_ports}.")
    if n_times < 1:
        raise ValueError("n_times must be at least 1.")

    n_eval_single = int(binary_eval.shape[0])
    n_outage_total = 0
    selected_draws: list[np.ndarray] = []
    for _ in range(n_times):
        # Draw n_ports distinct port indices uniformly at random. Using
        # choice without replacement preserves the constraint that each port
        # is read at most once per sensing round.
        selected = rng.choice(total_ports, size=n_ports, replace=False)
        _, _, n_outage = evaluate_selected_ports(binary_eval, selected)
        n_outage_total += int(n_outage)
        selected_draws.append(selected.astype(int, copy=False))

    n_eval_total = int(n_eval_single * n_times)
    observed_op = float(n_outage_total / n_eval_total)
    return observed_op, n_eval_total, n_outage_total, selected_draws


def build_result_rows(
    *,
    policy_name: str,
    regime_label: str,
    regime_index: int,
    rep_index: int,
    base_seed: int,
    design_size: int,
    eval_size: int,
    full_order: np.ndarray,
    target_ports: tuple[int, ...],
    binary_eval: np.ndarray,
) -> list[dict[str, Any]]:
    """Build per-budget evaluation rows for one policy, regime, and repetition.

    Args:
        policy_name (str): Policy identifier such as ``"mar"`` or
            ``"gc"``.
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index.
        rep_index (int): Zero-based repetition index.
        base_seed (int): Experiment-wide seed used to build the repetition RNG.
        design_size (int): Number of rows used to learn the policy in this
            repetition.
        eval_size (int): Number of evaluation rows in this repetition.
        full_order (np.ndarray): Full zero-based port order produced by the
            policy for the current design split.
        target_ports (tuple[int, ...]): Observation budgets to evaluate.
        binary_eval (np.ndarray): Evaluation exceedance matrix for this
            repetition and regime.

    Returns:
        list[dict[str, Any]]: One result dictionary per budget.

    Raises:
        ValueError: If the largest requested budget exceeds ``full_order``.
    """
    if max(target_ports) > len(full_order):
        raise ValueError("full_order must be at least as long as the largest requested budget.")

    ideal_op, n_ideal_outage = evaluate_ideal_ports(binary_eval)
    rows: list[dict[str, Any]] = []

    # Evaluate each requested prefix of the learned port order independently
    # so the result files can be joined directly across policies later.
    for n_ports in target_ports:
        ordered_selection = full_order[:n_ports].astype(int, copy=False)
        sorted_selection = np.sort(ordered_selection)
        observed_op, n_eval, n_outage = evaluate_selected_ports(binary_eval, ordered_selection)
        wilson_low, wilson_high = compute_wilson_interval(n_outage=n_outage, n_total=n_eval)
        rows.append(
            {
                "policy": policy_name,
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "rep": int(rep_index),
                "base_seed": int(base_seed),
                "n_ports": int(n_ports),
                "design_size": int(design_size),
                "eval_size": int(eval_size),
                "selection_order_0based": format_indices_zero_based(ordered_selection),
                "sorted_pattern_0based": format_indices_zero_based(sorted_selection),
                "selection_order_1based": format_indices_one_based(ordered_selection),
                "sorted_pattern_1based": format_indices_one_based(sorted_selection),
                "observed_op": f"{observed_op:.10f}",
                "ideal_op": f"{ideal_op:.10f}",
                "gap_to_ideal": f"{(observed_op - ideal_op):.10f}",
                "n_eval": int(n_eval),
                "n_outage": int(n_outage),
                "n_ideal_outage": int(n_ideal_outage),
                "wilson_low": f"{wilson_low:.10f}",
                "wilson_high": f"{wilson_high:.10f}",
            }
        )
    return rows


def build_canonical_rows(
    *,
    policy_name: str,
    regime_label: str,
    regime_index: int,
    full_order: np.ndarray,
    target_ports: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Build full-regime canonical selection rows for audit and reporting.

    Args:
        policy_name (str): Policy identifier such as ``"mar"`` or
            ``"gc"``.
        regime_label (str): Human-readable regime label.
        regime_index (int): Zero-based regime index.
        full_order (np.ndarray): Full learned zero-based port order on the
            entire regime.
        target_ports (tuple[int, ...]): Observation budgets to export.

    Returns:
        list[dict[str, Any]]: One row per budget containing the ordered prefix
            and the sorted read pattern.

    Raises:
        ValueError: If the largest requested budget exceeds ``full_order``.
    """
    if max(target_ports) > len(full_order):
        raise ValueError("full_order must be at least as long as the largest requested budget.")

    rows: list[dict[str, Any]] = []
    for n_ports in target_ports:
        ordered_selection = full_order[:n_ports].astype(int, copy=False)
        rows.append(
            {
                "policy": policy_name,
                "dataset": regime_label,
                "regime_index": int(regime_index),
                "n_ports": int(n_ports),
                "selection_order_0based": format_indices_zero_based(ordered_selection),
                "sorted_pattern_0based": format_indices_zero_based(np.sort(ordered_selection)),
                "selection_order_1based": format_indices_one_based(ordered_selection),
                "sorted_pattern_1based": format_indices_one_based(np.sort(ordered_selection)),
            }
        )
    return rows


def list_completed_repetitions(regime_output_dir: str | Path) -> set[int]:
    """List repetition indices that already have a finished CSV artifact.

    Args:
        regime_output_dir (str | Path): Per-regime output directory.

    Returns:
        set[int]: Set of repetition indices derived from files named
            ``rep_<index>.csv``.

    Raises:
        OSError: If the directory cannot be scanned.
    """
    regime_dir = Path(regime_output_dir).resolve()
    if not regime_dir.exists():
        return set()

    completed: set[int] = set()
    for csv_path in regime_dir.glob("rep_*.csv"):
        match = re.fullmatch(r"rep_(\d+)\.csv", csv_path.name)
        if match is None:
            continue
        completed.add(int(match.group(1)))
    return completed


def repetition_result_path(regime_output_dir: str | Path, rep_index: int) -> Path:
    """Build the canonical CSV path for one repetition result file.

    Args:
        regime_output_dir (str | Path): Per-regime output directory.
        rep_index (int): Zero-based repetition index.

    Returns:
        Path: Resolved path such as ``.../rep_00003.csv``.
    """
    return Path(regime_output_dir).resolve() / f"rep_{rep_index:05d}.csv"


def aggregate_result_rows_from_run(run_dir: str | Path, regime_label: str) -> list[dict[str, str]]:
    """Load all repetition CSV rows for one regime from a policy run directory.

    Args:
        run_dir (str | Path): Policy run directory containing one subdirectory
            per regime.
        regime_label (str): Regime whose CSV files should be collected.

    Returns:
        list[dict[str, str]]: Concatenated rows from all completed repetition
            files for the regime, sorted by repetition path order.

    Raises:
        OSError: If any CSV file cannot be read.
        csv.Error: If any file is malformed.
    """
    regime_dir = Path(run_dir).resolve() / regime_label
    all_rows: list[dict[str, str]] = []
    for csv_path in sorted(regime_dir.glob("rep_*.csv")):
        all_rows.extend(read_csv_rows(csv_path))
    return all_rows
