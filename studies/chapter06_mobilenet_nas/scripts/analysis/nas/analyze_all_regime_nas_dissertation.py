"""Build deterministic dissertation aggregates for MobileNetV3 NAS studies."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/analyze_all_regime_nas_dissertation.py'
REPOSITORY_ROOT = _COMPANION_ROOT
SPLIT_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas'
RESULTS_ROOT = SPLIT_ROOT / "results" / "nas_pqt_mobv3_all_regimes"
ANALYSIS_ROOT = RESULTS_ROOT / "nas_analysis"
OUTPUT_DIR = RESULTS_ROOT / "dissertation_analysis"
BUDGETS = (3, 4, 5, 6, 7)
EXPECTED_COMPLETED_COUNTS = {3: 97, 4: 98, 5: 98, 6: 98, 7: 98}
EXPECTED_SCHEDULED_TRIALS = 100
EXPECTED_REGIMES = 14
EXPECTED_SAMPLES_PER_REGIME = 200000
STUDY_NAME_TEMPLATE = "nas_ptq_mobv3_all_regimes_{observed_ports}_ports"
OP_EVALUATION_DIRECTORY = "op_eval_t1.1885399"
OP_RESULTS_FILENAME = "op_results_by_artifact_trial_split_and_regime.csv"
ARTIFACT_TYPES = ("fp32_keras", "int8_tflite")


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    """Require every downstream analysis column in an input table.

    Args:
        frame (pd.DataFrame): Loaded table whose schema must be validated.
        required (set[str]): Exact column names required by downstream logic.
        source (Path): Input path included in explicit schema errors.

    Returns:
        None: The function returns after the schema is confirmed.

    Raises:
        KeyError: If one or more required columns are absent.
    """
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Input {source} is missing columns: {missing}.")


def require_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    """Validate and normalize a CSV boolean column without truthy coercion.

    Args:
        series (pd.Series): Boolean values loaded by pandas. Native booleans or
            the exact strings ``True`` and ``False`` are accepted.
        column_name (str): Column name included in validation errors.

    Returns:
        pd.Series: Boolean series with the same index as the input.

    Raises:
        ValueError: If the column contains any other value.

    Examples:
        >>> require_boolean_series(pd.Series(["True", "False"]), "flag").tolist()
        [True, False]
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    allowed = {"True", "False"}
    values = set(series.dropna().astype(str).unique())
    if series.isna().any() or not values.issubset(allowed):
        raise ValueError(
            f"Column {column_name} must contain only True or False values."
        )
    return series.astype(str).map({"True": True, "False": False})


def load_completed_trials(
    analysis_root: Path = ANALYSIS_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Load completed feasible MobileNetV3 NAS trials for all five budgets.

    Args:
        analysis_root (Path): Root containing one generated NAS analysis
            directory per observation budget. Defaults to the authoritative
            all-regime analysis collection.
        results_root (Path): Root containing study configuration files. The
            default is the authoritative MobileNetV3 all-regime result set.

    Returns:
        pd.DataFrame: The 489 completed valid trials, sorted by observation
        budget and zero-based trial identifier.

    Raises:
        FileNotFoundError: If a study analysis or configuration file is absent.
        KeyError: If a required analysis column is absent.
        ValueError: If a study uses the wrong budget, task, objective, state
            count, feasibility status, selected trial, or Pareto contract.
    """
    required = {
        "trial_id",
        "state",
        "duration_seconds",
        "alpha",
        "truncation_block_name",
        "head_depth",
        "head_width",
        "learning_rate",
        "best_not_qt_val_metric",
        "best_qt_val_metric",
        "num_params",
        "qt_model_size_bytes",
        "trained_ram_peak_bytes",
        "tflite_peak_ram_bytes",
        "tflite_inference_latency_mean",
        "ram_budget_violated",
        "is_pareto",
        "is_selected_trial",
    }
    frames = []
    for observed_ports in BUDGETS:
        study_name = STUDY_NAME_TEMPLATE.format(observed_ports=observed_ports)
        trial_path = analysis_root / study_name / "nas_analysis_results.csv"
        config_path = results_root / study_name / "optuna_study" / "config.json"
        if not trial_path.is_file():
            raise FileNotFoundError(f"Missing NAS analysis table: {trial_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing NAS study configuration: {config_path}")

        config = pd.read_json(config_path, typ="series")
        expected_objectives = [
            "best_qt_val_model_op",
            "trained_ram_peak_bytes",
        ]
        if int(config["observed_ports_budget"]) != observed_ports:
            raise ValueError(f"Study {study_name} has the wrong observation budget.")
        if config["backbone_family"] != "MobileNetV3Small":
            raise ValueError(f"Study {study_name} is not MobileNetV3 Small.")
        if list(config["objectives"]) != expected_objectives:
            raise ValueError(f"Study {study_name} has unexpected NAS objectives.")

        frame = pd.read_csv(trial_path)
        require_columns(frame, required, trial_path)
        frame = frame.loc[frame["state"] == "COMPLETE"].copy()
        expected_count = EXPECTED_COMPLETED_COUNTS[observed_ports]
        if len(frame) != expected_count:
            raise ValueError(
                f"Study {study_name} must contain {expected_count} completed trials."
            )
        if frame["trial_id"].duplicated().any():
            raise ValueError(f"Study {study_name} contains duplicate trial IDs.")

        ram_violated = require_boolean_series(
            frame["ram_budget_violated"],
            "ram_budget_violated",
        )
        if ram_violated.any():
            raise ValueError(f"Study {study_name} contains a RAM-violating trial.")
        frame["is_pareto"] = require_boolean_series(
            frame["is_pareto"],
            "is_pareto",
        )
        frame["is_selected_trial"] = require_boolean_series(
            frame["is_selected_trial"],
            "is_selected_trial",
        )
        selected = frame.loc[frame["is_selected_trial"]]
        if len(selected) != 1:
            raise ValueError(f"Study {study_name} must contain one selected trial.")
        selected_op = float(selected.iloc[0]["best_qt_val_metric"])
        minimum_op = float(frame["best_qt_val_metric"].min())
        if not np.isclose(selected_op, minimum_op, rtol=0.0, atol=1e-15):
            raise ValueError(
                f"Study {study_name} selected trial is not INT8-OP minimal."
            )
        if not bool(selected.iloc[0]["is_pareto"]):
            raise ValueError(f"Study {study_name} selected trial is not Pareto optimal.")

        # The analysis inputs already exclude non-complete Optuna rows. The
        # explicit state filter prevents a future mixed-state table from
        # silently entering ranking, convergence, or Pareto calculations.
        frame["observed_ports"] = observed_ports
        frames.append(frame)

    trials = pd.concat(frames, ignore_index=True)
    if len(trials) != sum(EXPECTED_COMPLETED_COUNTS.values()):
        raise ValueError("The aggregate must contain exactly 489 completed trials.")
    return trials.sort_values(["observed_ports", "trial_id"]).reset_index(drop=True)


def build_study_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Summarize execution, feasibility, Pareto, and duration by budget.

    Args:
        trials (pd.DataFrame): Validated completed trials returned by
            :func:`load_completed_trials`.

    Returns:
        pd.DataFrame: Five budget rows containing scheduled, completed, valid,
        incomplete, selected, Pareto, and duration statistics.

    Raises:
        ValueError: If a budget has no trials or no selected trial.
    """
    rows = []
    for observed_ports in BUDGETS:
        budget_trials = trials.loc[trials["observed_ports"] == observed_ports]
        if budget_trials.empty:
            raise ValueError(f"No completed trials exist for N={observed_ports}.")
        selected = budget_trials.loc[budget_trials["is_selected_trial"]]
        if len(selected) != 1:
            raise ValueError(f"Budget N={observed_ports} must have one selection.")
        completed = len(budget_trials)
        rows.append(
            {
                "observed_ports": observed_ports,
                "scheduled_trials": EXPECTED_SCHEDULED_TRIALS,
                "completed_trials": completed,
                "valid_trials": completed,
                "ram_penalized_trials": 0,
                "incomplete_trials": EXPECTED_SCHEDULED_TRIALS - completed,
                "selected_trial": int(selected.iloc[0]["trial_id"]),
                "pareto_candidates": int(budget_trials["is_pareto"].sum()),
                "recorded_completed_duration_total_hours": float(
                    budget_trials["duration_seconds"].sum() / 3600.0
                ),
                "completed_duration_median_minutes": float(
                    budget_trials["duration_seconds"].median() / 60.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_outputs(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build selected-model, leading-candidate, and quantization tables.

    Args:
        trials (pd.DataFrame): Validated completed all-regime NAS trials.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Selected configurations,
        three minimum-INT8-OP candidates per budget, and FP32/INT8 validation OP
        comparisons for the selected models.

    Raises:
        ValueError: If a budget does not contain one selected trial or at least
            three completed candidates.
    """
    selected_rows = []
    leading_rows = []
    quantization_rows = []
    for observed_ports in BUDGETS:
        budget_trials = trials.loc[trials["observed_ports"] == observed_ports]
        selected = budget_trials.loc[budget_trials["is_selected_trial"]]
        if len(selected) != 1:
            raise ValueError(f"Budget N={observed_ports} must have one selection.")
        if len(budget_trials) < 3:
            raise ValueError(f"Budget N={observed_ports} needs three candidates.")
        row = selected.iloc[0]
        selected_rows.append(
            {
                "observed_ports": observed_ports,
                "trial_id": int(row["trial_id"]),
                "alpha": float(row["alpha"]),
                "truncation_block_name": row["truncation_block_name"],
                "head_depth": int(row["head_depth"]),
                "head_width": int(row["head_width"]),
                "learning_rate": float(row["learning_rate"]),
                "num_params": int(row["num_params"]),
                "int8_model_size_kib": float(row["qt_model_size_bytes"] / 1024.0),
                "trained_peak_ram_kib": float(
                    row["trained_ram_peak_bytes"] / 1024.0
                ),
                "tflite_peak_ram_kib": float(
                    row["tflite_peak_ram_bytes"] / 1024.0
                ),
                "host_latency_ms": float(
                    row["tflite_inference_latency_mean"] * 1000.0
                ),
                "fp32_validation_op": float(row["best_not_qt_val_metric"]),
                "int8_validation_op": float(row["best_qt_val_metric"]),
            }
        )
        quantization_rows.append(
            {
                "observed_ports": observed_ports,
                "fp32_validation_op": float(row["best_not_qt_val_metric"]),
                "int8_validation_op": float(row["best_qt_val_metric"]),
                "absolute_op_increase": float(
                    row["best_qt_val_metric"] - row["best_not_qt_val_metric"]
                ),
            }
        )
        leading = budget_trials.nsmallest(3, "best_qt_val_metric")
        for order, (_, candidate) in enumerate(leading.iterrows(), start=1):
            leading_rows.append(
                {
                    "observed_ports": observed_ports,
                    "op_order": order,
                    "trial_id": int(candidate["trial_id"]),
                    "alpha": float(candidate["alpha"]),
                    "truncation_block_name": candidate["truncation_block_name"],
                    "head_depth": int(candidate["head_depth"]),
                    "head_width": int(candidate["head_width"]),
                    "learning_rate": float(candidate["learning_rate"]),
                    "num_params": int(candidate["num_params"]),
                    "int8_model_size_kib": float(
                        candidate["qt_model_size_bytes"] / 1024.0
                    ),
                    "int8_validation_op": float(candidate["best_qt_val_metric"]),
                    "fp32_validation_op": float(
                        candidate["best_not_qt_val_metric"]
                    ),
                    "trained_peak_ram_kib": float(
                        candidate["trained_ram_peak_bytes"] / 1024.0
                    ),
                    "tflite_peak_ram_kib": float(
                        candidate["tflite_peak_ram_bytes"] / 1024.0
                    ),
                    "host_latency_ms": float(
                        candidate["tflite_inference_latency_mean"] * 1000.0
                    ),
                }
            )
    return (
        pd.DataFrame(selected_rows),
        pd.DataFrame(leading_rows),
        pd.DataFrame(quantization_rows),
    )


def build_search_plot_data(trials: pd.DataFrame) -> pd.DataFrame:
    """Build long-form Pareto and best-so-far plot data.

    Args:
        trials (pd.DataFrame): Validated completed all-regime NAS trials.

    Returns:
        pd.DataFrame: One row per completed trial with objective, memory,
        Pareto, selection, and cumulative-best fields.

    Raises:
        ValueError: If a budget contains no completed trial.
    """
    frames = []
    for observed_ports in BUDGETS:
        budget_trials = trials.loc[
            trials["observed_ports"] == observed_ports
        ].copy()
        if budget_trials.empty:
            raise ValueError(f"No completed trials exist for N={observed_ports}.")
        budget_trials = budget_trials.sort_values("trial_id")
        frames.append(
            pd.DataFrame(
                {
                    "observed_ports": observed_ports,
                    "trial_id": budget_trials["trial_id"].astype(int),
                    "trial_number": budget_trials["trial_id"].astype(int) + 1,
                    "fp32_validation_op": budget_trials[
                        "best_not_qt_val_metric"
                    ],
                    "int8_validation_op": budget_trials["best_qt_val_metric"],
                    "int8_model_size_kib": (
                        budget_trials["qt_model_size_bytes"] / 1024.0
                    ),
                    "trained_peak_ram_kib": (
                        budget_trials["trained_ram_peak_bytes"] / 1024.0
                    ),
                    "absolute_quantization_op_increase": (
                        budget_trials["best_qt_val_metric"]
                        - budget_trials["best_not_qt_val_metric"]
                    ),
                    "relative_quantization_op_change_percent": (
                        100.0
                        * (
                            budget_trials["best_qt_val_metric"]
                            - budget_trials["best_not_qt_val_metric"]
                        )
                        / budget_trials["best_not_qt_val_metric"]
                    ),
                    "is_pareto": budget_trials["is_pareto"].astype(int),
                    "is_selected": budget_trials["is_selected_trial"].astype(int),
                    "best_so_far_int8_op": budget_trials[
                        "best_qt_val_metric"
                    ].cummin(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_heldout_results(
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Load and pair held-out FP32 and INT8 OP rows for selected models.

    Args:
        results_root (Path): Root containing one selected-trial OP evaluation
            directory per observation budget. Defaults to the authoritative
            MobileNetV3 all-regime collection.

    Returns:
        pd.DataFrame: Seventy rows, one per budget and fading regime, with ideal,
        observed-only, FP32, and strict-INT8 model-only OP values.

    Raises:
        FileNotFoundError: If an OP evaluation table is absent.
        KeyError: If a required OP column is absent.
        ValueError: If split, budget, rank, sample count, regime count, or
            FP32/INT8 pairing differs from the approved evaluation contract.
    """
    required = {
        "artifact_type",
        "rank",
        "trial_id",
        "split",
        "regime",
        "num_samples",
        "n_observed_ports",
        "ideal_op",
        "observed_op",
        "model_only_op",
        "model_plus_observed_op",
        "model_only_n_outage_samples",
    }
    paired_frames = []
    for observed_ports in BUDGETS:
        study_name = STUDY_NAME_TEMPLATE.format(observed_ports=observed_ports)
        source = (
            results_root
            / study_name
            / OP_EVALUATION_DIRECTORY
            / OP_RESULTS_FILENAME
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing held-out OP table: {source}")
        frame = pd.read_csv(source)
        require_columns(frame, required, source)
        if set(frame["artifact_type"]) != set(ARTIFACT_TYPES):
            raise ValueError(f"Study {study_name} lacks FP32 or INT8 artifacts.")
        if set(frame["split"]) != {"test"}:
            raise ValueError(f"Study {study_name} must use only the test split.")
        if set(frame["rank"].astype(int)) != {1}:
            raise ValueError(f"Study {study_name} must evaluate only rank one.")
        if set(frame["n_observed_ports"].astype(int)) != {observed_ports}:
            raise ValueError(f"Study {study_name} has the wrong port budget.")
        if set(frame["num_samples"].astype(int)) != {
            EXPECTED_SAMPLES_PER_REGIME
        }:
            raise ValueError(f"Study {study_name} has the wrong sample count.")
        fp32 = frame.loc[frame["artifact_type"] == "fp32_keras"].copy()
        int8 = frame.loc[frame["artifact_type"] == "int8_tflite"].copy()
        if len(fp32) != EXPECTED_REGIMES or len(int8) != EXPECTED_REGIMES:
            raise ValueError(
                f"Study {study_name} must contain paired FP32 and INT8 rows "
                f"for {EXPECTED_REGIMES} regimes."
            )
        if fp32["regime"].duplicated().any() or int8["regime"].duplicated().any():
            raise ValueError(f"Study {study_name} contains duplicate regime rows.")
        paired = fp32.merge(
            int8,
            on="regime",
            how="outer",
            suffixes=("_fp32", "_int8"),
            validate="one_to_one",
            indicator=True,
        )
        if len(paired) != EXPECTED_REGIMES or set(paired["_merge"]) != {"both"}:
            raise ValueError(
                f"Study {study_name} requires paired FP32 and INT8 regimes."
            )
        for shared_column in ("trial_id", "ideal_op", "observed_op", "num_samples"):
            left = paired[f"{shared_column}_fp32"]
            right = paired[f"{shared_column}_int8"]
            if not np.allclose(left, right, rtol=0.0, atol=1e-15):
                raise ValueError(
                    f"Study {study_name} disagrees on {shared_column}."
                )
        paired_frames.append(
            pd.DataFrame(
                {
                    "observed_ports": observed_ports,
                    "regime": paired["regime"],
                    "trial_id": paired["trial_id_fp32"].astype(int),
                    "num_samples": paired["num_samples_fp32"].astype(int),
                    "ideal_op": paired["ideal_op_fp32"],
                    "observed_op": paired["observed_op_fp32"],
                    "fp32_model_op": paired["model_only_op_fp32"],
                    "int8_model_op": paired["model_only_op_int8"],
                    "fp32_outage_samples": paired[
                        "model_only_n_outage_samples_fp32"
                    ].astype(int),
                    "int8_outage_samples": paired[
                        "model_only_n_outage_samples_int8"
                    ].astype(int),
                }
            )
        )
    heldout = pd.concat(paired_frames, ignore_index=True)
    if len(heldout) != len(BUDGETS) * EXPECTED_REGIMES:
        raise ValueError("Held-out aggregation must contain exactly 70 rows.")
    return heldout.sort_values(["observed_ports", "regime"]).reset_index(drop=True)


def build_heldout_summary(heldout: pd.DataFrame) -> pd.DataFrame:
    """Summarize macro held-out OP and quantization differences by budget.

    Args:
        heldout (pd.DataFrame): Paired held-out rows returned by
            :func:`load_heldout_results`.

    Returns:
        pd.DataFrame: Five rows with macro ideal, observed-only, FP32, INT8, and
        maximum absolute INT8-minus-FP32 OP values.

    Raises:
        ValueError: If any observation budget is absent.
    """
    rows = []
    for observed_ports in BUDGETS:
        budget = heldout.loc[heldout["observed_ports"] == observed_ports]
        if len(budget) != EXPECTED_REGIMES:
            raise ValueError(
                f"Budget N={observed_ports} must contain {EXPECTED_REGIMES} regimes."
            )
        difference = budget["int8_model_op"] - budget["fp32_model_op"]
        rows.append(
            {
                "observed_ports": observed_ports,
                "macro_ideal_op": float(budget["ideal_op"].mean()),
                "macro_observed_op": float(budget["observed_op"].mean()),
                "macro_fp32_model_op": float(budget["fp32_model_op"].mean()),
                "macro_int8_model_op": float(budget["int8_model_op"].mean()),
                "maximum_int8_minus_fp32_op": float(difference.max()),
            }
        )
    return pd.DataFrame(rows)


def build_dissertation_outputs(
    analysis_root: Path = ANALYSIS_ROOT,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, pd.DataFrame]:
    """Build every deterministic table required by Chapter 6 and its figures.

    Args:
        analysis_root (Path): Root containing completed per-budget NAS analysis
            tables. Defaults to the authoritative generated analyses.
        results_root (Path): Root containing study configurations and selected
            held-out OP evaluations. Defaults to the authoritative result set.

    Returns:
        dict[str, pd.DataFrame]: Named study, candidate, quantization, search,
        held-out, and macro-summary tables.

    Raises:
        FileNotFoundError: If an authoritative input is absent.
        KeyError: If an input omits a required column.
        ValueError: If a study or evaluation violates an evidence contract.
    """
    trials = load_completed_trials(analysis_root, results_root)
    selected, leading, quantization = build_candidate_outputs(trials)
    heldout = load_heldout_results(results_root)
    expected_trials = dict(
        zip(selected["observed_ports"], selected["trial_id"], strict=True)
    )
    evaluated_trials = dict(
        zip(
            heldout.groupby("observed_ports")["trial_id"].first().index,
            heldout.groupby("observed_ports")["trial_id"].first().values,
            strict=True,
        )
    )
    if expected_trials != evaluated_trials:
        raise ValueError("Held-out OP files do not evaluate the selected trials.")
    return {
        "study_summary": build_study_summary(trials),
        "selected_models": selected,
        "leading_candidates": leading,
        "quantization_summary": quantization,
        "search_plot_data": build_search_plot_data(trials),
        "heldout_by_regime": heldout,
        "heldout_summary": build_heldout_summary(heldout),
    }


def write_dissertation_outputs(
    outputs: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Write deterministic dissertation tables with stable numeric formatting.

    Args:
        outputs (dict[str, pd.DataFrame]): Named tables returned by
            :func:`build_dissertation_outputs`.
        output_dir (Path): Destination for derived CSV evidence. Defaults to the
            new MobileNetV3 result collection's dissertation-analysis folder.

    Returns:
        tuple[Path, ...]: Written CSV paths in insertion order.

    Raises:
        ValueError: If the output mapping is empty.
        OSError: If the destination or a CSV cannot be written.
    """
    if not outputs:
        raise ValueError("At least one dissertation output table is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in outputs.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.12g")
        paths.append(path)
    return tuple(paths)


def analyze_all_regime_nas_results(
    analysis_root: Path = ANALYSIS_ROOT,
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Aggregate and write the complete all-regime MobileNetV3 evidence set.

    Args:
        analysis_root (Path): Root containing the generated per-budget NAS
            analyses. Defaults to the authoritative analysis collection.
        results_root (Path): Root containing study configurations and held-out
            OP evaluations. Defaults to the authoritative result collection.
        output_dir (Path): Destination for deterministic dissertation CSVs.
            Defaults to the result collection's dissertation-analysis folder.

    Returns:
        tuple[Path, ...]: Paths of all written dissertation-analysis tables.

    Raises:
        FileNotFoundError: If an authoritative input is absent.
        KeyError: If an input omits a required column.
        ValueError: If a numerical or completeness contract is violated.
        OSError: If an output directory or CSV cannot be written.
    """
    outputs = build_dissertation_outputs(analysis_root, results_root)
    return write_dissertation_outputs(outputs, output_dir)
