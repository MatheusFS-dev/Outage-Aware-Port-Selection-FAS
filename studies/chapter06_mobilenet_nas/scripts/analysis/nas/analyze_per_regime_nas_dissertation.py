"""Build dissertation aggregates and figures for per-regime MobileNetV3 NAS."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/analyze_per_regime_nas_dissertation.py'
REPOSITORY_ROOT = _COMPANION_ROOT
SPLIT_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas'
RESULTS_ROOT = SPLIT_ROOT / "results" / "nas_ptq_mobv3_per_regime"
OUTPUT_DIR = RESULTS_ROOT / "dissertation_analysis"
ALL_REGIME_ANALYSIS_DIR = (
    SPLIT_ROOT / "results" / "nas_pqt_mobv3_all_regimes" / "dissertation_analysis"
)
FIGURE_DIR = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/figures'
APPENDIX_DIR = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/tables'
BUDGETS = (3, 4, 5, 6, 7)
REGIMES = (
    "kappa0_mu1_m0",
    "kappa0_mu1_m2",
    "kappa0_mu1_m50",
    "kappa0_mu2_m50",
    "kappa0_mu5_m50",
    "kappa5_mu1_m0",
    "kappa5_mu1_m2",
    "kappa5_mu1_m50",
    "kappa5_mu2_m0",
    "kappa5_mu2_m2",
    "kappa5_mu2_m50",
    "kappa5_mu5_m0",
    "kappa5_mu5_m2",
    "kappa5_mu5_m50",
)
REGIME_LABELS = (
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!\to\!0$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!=\!2$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!\to\!\infty$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!2,m\!\to\!\infty$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!5,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!\to\!\infty$",
)
RADAR_COLORS = {
    "Ideal": "black",
    "Observed only": "#009E73",
    "FP32": "#666666",
    "INT8": "#D55E00",
}
REPRESENTATIVE_REGIMES = (
    ("kappa5_mu5_m50", "kappa5_mu5_minf", "Easy"),
    ("kappa5_mu2_m2", "kappa5_mu2_m2", "Medium"),
    ("kappa5_mu2_m0", "kappa5_mu2_m0", "Very hard"),
)
STUDY_PATTERN = re.compile(
    r"nas_ptq_mobv3_regime_(kappa\d+_mu\d+_m\d+)_(\d+)_ports"
)
OP_EVALUATION_DIRECTORY = "op_eval_t1.1885399"
OP_RESULTS_FILENAME = "op_results_by_artifact_trial_split_and_regime.csv"
EXPECTED_SAMPLES = 200000
EXPECTED_TRIALS = 100
RAM_LIMIT_BYTES = 224 * 1024


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Input {source} is missing columns: {missing}.")


def _decode_parameter(value: float, distribution_json: str):
    distribution = json.loads(distribution_json)
    attributes = distribution.get("attributes", {})
    choices = attributes.get("choices")
    if choices is not None:
        return choices[int(value)]
    return value


def _load_attributes(connection: sqlite3.Connection, trial_id: int) -> dict:
    rows = connection.execute(
        "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?",
        (trial_id,),
    ).fetchall()
    return {key: json.loads(value_json) for key, value_json in rows}


def _load_parameters(connection: sqlite3.Connection, trial_id: int) -> dict:
    rows = connection.execute(
        "SELECT param_name, param_value, distribution_json "
        "FROM trial_params WHERE trial_id = ?",
        (trial_id,),
    ).fetchall()
    return {
        name: _decode_parameter(value, distribution)
        for name, value, distribution in rows
    }


def _pareto_membership(objective: np.ndarray, memory: np.ndarray) -> np.ndarray:
    membership = np.ones(len(objective), dtype=bool)
    for index in range(len(objective)):
        dominated = (
            (objective <= objective[index])
            & (memory <= memory[index])
            & ((objective < objective[index]) | (memory < memory[index]))
        )
        membership[index] = not bool(np.any(dominated))
    return membership


def _load_completed_candidates(
    connection: sqlite3.Connection,
    regime: str,
    observed_ports: int,
) -> pd.DataFrame:
    query = """
        SELECT
            trials.number AS trial_id,
            MAX(CASE WHEN trial_values.objective = 0 THEN trial_values.value END)
                AS int8_validation_op,
            MAX(CASE WHEN trial_values.objective = 1 THEN trial_values.value END)
                AS trained_ram_peak_bytes
        FROM trials
        LEFT JOIN trial_values ON trial_values.trial_id = trials.trial_id
        WHERE trials.state = 'COMPLETE'
        GROUP BY trials.trial_id, trials.number
        ORDER BY trials.number
    """
    frame = pd.read_sql_query(query, connection)
    if len(frame) == 0 or frame.isna().any().any():
        raise ValueError(
            f"Finalized objective pairs are incomplete for {regime}, N={observed_ports}."
        )
    if frame["trial_id"].duplicated().any():
        raise ValueError(f"Duplicate finalized trials for {regime}, N={observed_ports}.")
    if (frame["trained_ram_peak_bytes"] >= RAM_LIMIT_BYTES).any():
        raise ValueError(f"A finalized trial violates RAM for {regime}, N={observed_ports}.")
    frame["is_pareto"] = _pareto_membership(
        frame["int8_validation_op"].to_numpy(dtype=float),
        frame["trained_ram_peak_bytes"].to_numpy(dtype=float),
    )
    frame["regime"] = regime
    frame["observed_ports"] = observed_ports
    frame["trial_position"] = frame["trial_id"] + 1
    return frame


def _load_evaluation(study_dir: Path, regime: str, observed_ports: int) -> dict:
    source = (
        study_dir
        / OP_EVALUATION_DIRECTORY
        / OP_RESULTS_FILENAME
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing held-out evaluation summary: {source}")
    frame = pd.read_csv(source)
    required = {
        "artifact_type",
        "trial_id",
        "split",
        "regime",
        "num_samples",
        "n_observed_ports",
        "ideal_op",
        "observed_op",
        "model_only_op",
    }
    _require_columns(frame, required, source)
    expected_keys = {
        ("fp32_keras", "val"),
        ("fp32_keras", "test"),
        ("int8_tflite", "val"),
        ("int8_tflite", "test"),
    }
    actual_keys = set(zip(frame["artifact_type"], frame["split"], strict=True))
    if len(frame) != 4 or actual_keys != expected_keys:
        raise ValueError(f"Evaluation rows are incomplete in {source}.")
    if set(frame["regime"]) != {regime}:
        raise ValueError(f"Evaluation regime mismatch in {source}.")
    if set(frame["n_observed_ports"].astype(int)) != {observed_ports}:
        raise ValueError(f"Evaluation budget mismatch in {source}.")
    if set(frame["num_samples"].astype(int)) != {EXPECTED_SAMPLES}:
        raise ValueError(f"Evaluation sample count mismatch in {source}.")
    trial_ids = set(frame["trial_id"].astype(int))
    if len(trial_ids) != 1:
        raise ValueError(f"FP32 and INT8 artifacts use different trials in {source}.")
    rows = {
        (row.artifact_type, row.split): row
        for row in frame.itertuples(index=False)
    }
    for split in ("val", "test"):
        fp32 = rows[("fp32_keras", split)]
        int8 = rows[("int8_tflite", split)]
        if not np.isclose(fp32.ideal_op, int8.ideal_op, rtol=0.0, atol=0.0):
            raise ValueError(f"Ideal reference mismatch in {source}.")
        if not np.isclose(fp32.observed_op, int8.observed_op, rtol=0.0, atol=0.0):
            raise ValueError(f"Observed reference mismatch in {source}.")
    return {
        "trial_id": trial_ids.pop(),
        "validation_fp32_op": float(rows[("fp32_keras", "val")].model_only_op),
        "validation_int8_op": float(rows[("int8_tflite", "val")].model_only_op),
        "ideal_test_op": float(rows[("int8_tflite", "test")].ideal_op),
        "observed_test_op": float(rows[("int8_tflite", "test")].observed_op),
        "fp32_test_op": float(rows[("fp32_keras", "test")].model_only_op),
        "int8_test_op": float(rows[("int8_tflite", "test")].model_only_op),
    }


def load_per_regime_results(results_root: Path = RESULTS_ROOT) -> dict[str, pd.DataFrame]:
    """Load and validate the 70 per-regime MobileNetV3 NAS studies.

    Args:
        results_root (Path): Root containing one copied NAS study for every
            fading-regime and observation-budget pair.

    Returns:
        dict[str, pd.DataFrame]: Selected-model, study-state, and finalized
        candidate tables used by all downstream dissertation outputs.

    Raises:
        FileNotFoundError: If a configuration, database, model, INT8 model, or
            evaluation summary is missing.
        KeyError: If a required evaluation column is absent.
        ValueError: If study counts, contracts, artifacts, or metrics conflict.
    """
    study_dirs = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and STUDY_PATTERN.fullmatch(path.name) is not None
    )
    if len(study_dirs) != len(REGIMES) * len(BUDGETS):
        raise ValueError(f"Expected 70 per-regime studies, found {len(study_dirs)}.")
    selected_rows = []
    study_rows = []
    candidate_frames = []
    observed_pairs = set()
    for study_dir in study_dirs:
        match = STUDY_PATTERN.fullmatch(study_dir.name)
        if match is None:
            raise ValueError(f"Unsupported per-regime study directory: {study_dir}")
        regime = match.group(1)
        observed_ports = int(match.group(2))
        observed_pairs.add((regime, observed_ports))
        config_path = study_dir / "optuna_study" / "config.json"
        database_path = study_dir / "optuna_study" / "optuna_study.db"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing study configuration: {config_path}")
        if not database_path.is_file():
            raise FileNotFoundError(f"Missing Optuna database: {database_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("regime_label") != regime:
            raise ValueError(f"Regime mismatch in {config_path}.")
        if int(config.get("observed_ports_budget", -1)) != observed_ports:
            raise ValueError(f"Observation-budget mismatch in {config_path}.")
        if config.get("train_regime_labels") != [regime]:
            raise ValueError(f"Training-regime contract mismatch in {config_path}.")
        if config.get("backbone_family") != "MobileNetV3Small":
            raise ValueError(f"Backbone mismatch in {config_path}.")
        evaluation = _load_evaluation(study_dir, regime, observed_ports)
        selected_trial = int(evaluation["trial_id"])
        with sqlite3.connect(database_path) as connection:
            state_rows = connection.execute("SELECT state FROM trials").fetchall()
            states = Counter(state for (state,) in state_rows)
            if sum(states.values()) != EXPECTED_TRIALS:
                raise ValueError(f"Study {study_dir.name} does not contain 100 trials.")
            candidates = _load_completed_candidates(connection, regime, observed_ports)
            candidate_frames.append(candidates)
            storage_row = connection.execute(
                "SELECT trial_id, state FROM trials WHERE number = ?",
                (selected_trial,),
            ).fetchone()
            if storage_row is None:
                raise ValueError(f"Selected trial {selected_trial} is absent in {database_path}.")
            storage_id, storage_state = storage_row
            attributes = _load_attributes(connection, storage_id)
            parameters = _load_parameters(connection, storage_id)
        required_attributes = {
            "best_not_qt_val_model_op",
            "best_qt_val_model_op",
            "num_params",
            "trained_ram_peak_bytes",
            "qt_model_size_bytes",
            "tflite_inference_latency",
        }
        missing_attributes = sorted(required_attributes - set(attributes))
        if missing_attributes:
            raise ValueError(
                f"Selected trial {selected_trial} lacks attributes {missing_attributes}."
            )
        required_parameters = {
            "alpha",
            "truncation_block_name",
            "head_depth",
            "head_width",
            "learning_rate",
        }
        missing_parameters = sorted(required_parameters - set(parameters))
        if missing_parameters:
            raise ValueError(
                f"Selected trial {selected_trial} lacks parameters {missing_parameters}."
            )
        ranked_config = json.loads(
            (study_dir / OP_EVALUATION_DIRECTORY / "op_eval_config.json").read_text(
                encoding="utf-8"
            )
        )
        ranked_trials = ranked_config.get("ranked_trials", [])
        if len(ranked_trials) != 1 or int(ranked_trials[0]["trial_id"]) != selected_trial:
            raise ValueError(f"Selected evaluation rank mismatch in {study_dir}.")
        model_path = study_dir / "optuna_study" / "model" / f"top_1_trial_{selected_trial}.keras"
        tflite_path = study_dir / "optuna_study" / "tflite" / f"top_1_trial_{selected_trial}.tflite"
        if not model_path.is_file() or not tflite_path.is_file():
            raise FileNotFoundError(f"Selected model artifacts are incomplete in {study_dir}.")
        selected_rows.append(
            {
                "regime": regime,
                "observed_ports": observed_ports,
                "trial_id": selected_trial,
                "trial_position": selected_trial + 1,
                "database_state": storage_state,
                "database_finalized": storage_state == "COMPLETE",
                "alpha": float(parameters["alpha"]),
                "truncation_block_name": str(parameters["truncation_block_name"]),
                "head_depth": int(parameters["head_depth"]),
                "head_width": int(parameters["head_width"]),
                "learning_rate": float(parameters["learning_rate"]),
                "num_params": int(attributes["num_params"]),
                "trained_ram_peak_bytes": int(attributes["trained_ram_peak_bytes"]),
                "qt_model_size_bytes": int(attributes["qt_model_size_bytes"]),
                "host_int8_latency_seconds": float(
                    attributes["tflite_inference_latency"]["mean_seconds"]
                ),
                "fp32_validation_op": float(attributes["best_not_qt_val_model_op"]),
                "int8_validation_op": float(attributes["best_qt_val_model_op"]),
                **evaluation,
            }
        )
        study_rows.append(
            {
                "regime": regime,
                "observed_ports": observed_ports,
                "scheduled_trials": EXPECTED_TRIALS,
                "complete_trials": int(states["COMPLETE"]),
                "unfinalized_trials": EXPECTED_TRIALS - int(states["COMPLETE"]),
                "pareto_candidates": int(candidates["is_pareto"].sum()),
                "selected_trial_id": selected_trial,
                "selected_trial_position": selected_trial + 1,
                "selected_database_state": storage_state,
            }
        )
    expected_pairs = {(regime, budget) for regime in REGIMES for budget in BUDGETS}
    if observed_pairs != expected_pairs:
        raise ValueError("Per-regime study matrix is incomplete.")
    selected = pd.DataFrame(selected_rows).sort_values(
        ["regime", "observed_ports"]
    ).reset_index(drop=True)
    studies = pd.DataFrame(study_rows).sort_values(
        ["regime", "observed_ports"]
    ).reset_index(drop=True)
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["regime", "observed_ports", "trial_id"]
    ).reset_index(drop=True)
    return {
        "selected_models": selected,
        "study_summary": studies,
        "completed_candidates": candidates,
    }


def _build_convergence(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, budget), group in candidates.groupby(["regime", "observed_ports"]):
        values = np.full(EXPECTED_TRIALS, np.nan)
        trial_ids = group["trial_id"].to_numpy(dtype=int)
        values[trial_ids] = group["int8_validation_op"].to_numpy(dtype=float)
        series = pd.Series(values).ffill()
        first_valid = int(series.first_valid_index())
        series.iloc[:first_valid] = series.iloc[first_valid]
        cumulative = np.minimum.accumulate(series.to_numpy(dtype=float))
        initial = float(cumulative[0])
        final = float(cumulative[-1])
        if np.isclose(initial, final, rtol=0.0, atol=1e-15):
            progress = np.ones(EXPECTED_TRIALS)
        else:
            progress = (initial - cumulative) / (initial - final)
        for trial_id, fraction in enumerate(progress):
            rows.append(
                {
                    "regime": regime,
                    "observed_ports": budget,
                    "trial_position": trial_id + 1,
                    "best_int8_validation_op": float(cumulative[trial_id]),
                    "finalized_improvement_fraction": float(fraction),
                }
            )
    return pd.DataFrame(rows)


def build_summaries(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build budget, held-out, quantization, and convergence summaries.

    Args:
        outputs (dict[str, pd.DataFrame]): Validated frames returned by
            :func:`load_per_regime_results`.

    Returns:
        dict[str, pd.DataFrame]: Original frames plus deterministic summary
        tables used by plots and LaTeX generation.

    Raises:
        KeyError: If a required validated frame is absent.
        ValueError: If selected-model quantization does not increase test OP.
    """
    selected = outputs["selected_models"].copy()
    studies = outputs["study_summary"].copy()
    candidates = outputs["completed_candidates"].copy()
    selected["quantization_validation_delta"] = (
        selected["int8_validation_op"] - selected["fp32_validation_op"]
    )
    selected["quantization_test_delta"] = (
        selected["int8_test_op"] - selected["fp32_test_op"]
    )
    selected["int8_minus_observed"] = (
        selected["int8_test_op"] - selected["observed_test_op"]
    )
    selected["int8_beats_observed"] = selected["int8_minus_observed"] < 0.0
    if not selected["quantization_test_delta"].gt(0.0).all():
        raise ValueError("Every selected model must have positive test quantization delta.")
    macro_rows = []
    budget_rows = []
    for budget in BUDGETS:
        selected_budget = selected.loc[selected["observed_ports"] == budget]
        study_budget = studies.loc[studies["observed_ports"] == budget]
        macro_rows.append(
            {
                "observed_ports": budget,
                "ideal_test_op": selected_budget["ideal_test_op"].mean(),
                "observed_test_op": selected_budget["observed_test_op"].mean(),
                "fp32_test_op": selected_budget["fp32_test_op"].mean(),
                "int8_test_op": selected_budget["int8_test_op"].mean(),
                "quantization_test_delta": selected_budget[
                    "quantization_test_delta"
                ].mean(),
                "int8_beats_observed_count": int(
                    selected_budget["int8_beats_observed"].sum()
                ),
            }
        )
        budget_rows.append(
            {
                "observed_ports": budget,
                "scheduled_trials": int(study_budget["scheduled_trials"].sum()),
                "complete_trials": int(study_budget["complete_trials"].sum()),
                "unfinalized_trials": int(study_budget["unfinalized_trials"].sum()),
                "pareto_candidates": int(study_budget["pareto_candidates"].sum()),
                "parameters_min": int(selected_budget["num_params"].min()),
                "parameters_median": int(selected_budget["num_params"].median()),
                "parameters_max": int(selected_budget["num_params"].max()),
                "ram_kib_min": selected_budget["trained_ram_peak_bytes"].min() / 1024.0,
                "ram_kib_median": selected_budget["trained_ram_peak_bytes"].median() / 1024.0,
                "ram_kib_max": selected_budget["trained_ram_peak_bytes"].max() / 1024.0,
                "int8_size_kib_min": selected_budget["qt_model_size_bytes"].min() / 1024.0,
                "int8_size_kib_median": selected_budget["qt_model_size_bytes"].median() / 1024.0,
                "int8_size_kib_max": selected_budget["qt_model_size_bytes"].max() / 1024.0,
            }
        )
    return {
        "selected_models": selected,
        "study_summary": studies,
        "completed_candidates": candidates,
        "heldout_macro": pd.DataFrame(macro_rows),
        "budget_summary": pd.DataFrame(budget_rows),
        "convergence": _build_convergence(candidates),
    }


def build_all_regime_comparison(
    selected: pd.DataFrame,
    all_regime_analysis_dir: Path = ALL_REGIME_ANALYSIS_DIR,
) -> dict[str, pd.DataFrame]:
    """Compare selected per-regime and all-regime held-out results.

    Args:
        selected (pd.DataFrame): Validated per-regime selected-model results.
        all_regime_analysis_dir (Path): Directory containing the authoritative
            all-regime held-out summary.

    Returns:
        dict[str, pd.DataFrame]: Complete regime-budget comparisons and the
        five-row budget summary.

    Raises:
        FileNotFoundError: If the all-regime held-out table is absent.
        KeyError: If either result table lacks required columns.
        ValueError: If regime-budget keys are incomplete or duplicated.
    """
    source = all_regime_analysis_dir / "heldout_by_regime.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing all-regime held-out summary: {source}")
    all_regime = pd.read_csv(source)
    per_required = {
        "regime",
        "observed_ports",
        "ideal_test_op",
        "observed_test_op",
        "fp32_test_op",
        "int8_test_op",
    }
    all_required = {
        "regime",
        "observed_ports",
        "ideal_op",
        "observed_op",
        "fp32_model_op",
        "int8_model_op",
    }
    _require_columns(selected, per_required, RESULTS_ROOT / "selected models")
    _require_columns(all_regime, all_required, source)
    per_columns = selected[
        [
            "regime",
            "observed_ports",
            "ideal_test_op",
            "observed_test_op",
            "fp32_test_op",
            "int8_test_op",
        ]
    ].rename(
        columns={
            "ideal_test_op": "per_regime_ideal_op",
            "observed_test_op": "per_regime_observed_op",
            "fp32_test_op": "per_regime_fp32_op",
            "int8_test_op": "per_regime_int8_op",
        }
    )
    all_columns = all_regime[
        [
            "regime",
            "observed_ports",
            "ideal_op",
            "observed_op",
            "fp32_model_op",
            "int8_model_op",
        ]
    ].rename(
        columns={
            "ideal_op": "all_regime_ideal_op",
            "observed_op": "all_regime_observed_op",
            "fp32_model_op": "all_regime_fp32_op",
            "int8_model_op": "all_regime_int8_op",
        }
    )
    by_regime = per_columns.merge(
        all_columns,
        on=["regime", "observed_ports"],
        how="inner",
        validate="one_to_one",
    )
    if len(by_regime) != len(REGIMES) * len(BUDGETS):
        raise ValueError("All-regime/per-regime comparison matrix is incomplete.")
    by_regime["int8_op_delta"] = (
        by_regime["per_regime_int8_op"] - by_regime["all_regime_int8_op"]
    )
    by_regime["lower_op_campaign"] = np.where(
        by_regime["int8_op_delta"] < 0.0,
        "per-regime",
        np.where(by_regime["int8_op_delta"] > 0.0, "all-regime", "equal"),
    )
    by_regime = by_regime.sort_values(
        ["regime", "observed_ports"]
    ).reset_index(drop=True)
    summary_rows = []
    for budget in BUDGETS:
        subset = by_regime.loc[by_regime["observed_ports"] == budget]
        all_ideal = float(subset["all_regime_ideal_op"].mean())
        all_observed = float(subset["all_regime_observed_op"].mean())
        all_int8 = float(subset["all_regime_int8_op"].mean())
        per_ideal = float(subset["per_regime_ideal_op"].mean())
        per_observed = float(subset["per_regime_observed_op"].mean())
        per_int8 = float(subset["per_regime_int8_op"].mean())
        summary_rows.append(
            {
                "observed_ports": budget,
                "all_regime_int8_op": all_int8,
                "per_regime_int8_op": per_int8,
                "int8_op_delta": per_int8 - all_int8,
                "per_regime_lower_count": int(
                    (subset["int8_op_delta"] < 0.0).sum()
                ),
                "equal_count": int((subset["int8_op_delta"] == 0.0).sum()),
                "all_regime_lower_count": int(
                    (subset["int8_op_delta"] > 0.0).sum()
                ),
                "all_regime_recoverable_gap_closed": (
                    (all_observed - all_int8) / (all_observed - all_ideal)
                ),
                "per_regime_recoverable_gap_closed": (
                    (per_observed - per_int8) / (per_observed - per_ideal)
                ),
            }
        )
    return {
        "by_regime": by_regime,
        "summary": pd.DataFrame(summary_rows),
    }


def write_analysis_outputs(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Write deterministic dissertation-analysis CSV outputs.

    Args:
        summaries (dict[str, pd.DataFrame]): Frames returned by
            :func:`build_summaries`.
        output_dir (Path): Directory receiving the generated CSV files.

    Returns:
        list[Path]: Generated CSV paths in deterministic key order.

    Raises:
        OSError: If an output directory or CSV cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in sorted(summaries):
        path = output_dir / f"{key}.csv"
        summaries[key].to_csv(path, index=False)
        paths.append(path)
    return paths


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(axis) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.grid(True, which="major", color="0.82", linestyle="-", linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)


def _set_op_scale(axis, values: list[np.ndarray]) -> None:
    combined = np.concatenate([np.asarray(value, dtype=float) for value in values])
    if np.any(~np.isfinite(combined)) or np.any(combined <= 0.0) or np.any(combined > 1.0):
        raise ValueError("OP plot values must be finite probabilities in (0, 1].")
    minimum = float(combined.min())
    maximum = float(combined.max())
    log_span = np.log(maximum) - np.log(minimum)
    padding = max(0.08 * log_span, 0.001)
    lower = np.exp(np.log(minimum) - padding)
    upper = min(1.0, np.exp(np.log(maximum) + padding))
    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    if maximum / minimum < 10.0:
        ticks = np.geomspace(lower, upper, 5)
        precision = max(2, int(np.ceil(-np.log10(upper - lower))) + 1)
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.{precision}f}")
        )
    else:
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _regime_display(regime: str) -> str:
    match = re.fullmatch(r"kappa(\d+)_mu(\d+)_m(\d+)", regime)
    if match is None:
        raise ValueError(f"Unsupported regime label: {regime}")
    kappa, mu, shadowing = match.groups()
    kappa_text = r"\kappa\!\to\!0" if kappa == "0" else rf"\kappa={kappa}"
    if shadowing == "0":
        shadowing_text = r"m\!\to\!0"
    elif shadowing == "50":
        shadowing_text = r"m\!\to\!\infty"
    else:
        shadowing_text = f"m={shadowing}"
    return f"${kappa_text},\\ \\mu={mu},\\ {shadowing_text}$"


def _regime_annotation(regime: str) -> str:
    return _regime_display(regime).replace(r"\ ", r"\quad ")


def _regime_slug(regime: str) -> str:
    return regime.replace("kappa0", "kappa0").replace("_m50", "_minf")


def _plot_representative_convergence(
    convergence: pd.DataFrame,
    regime: str,
    difficulty: str,
    output_path: Path,
) -> None:
    _set_plot_style()
    figure, axis = plt.subplots(figsize=(5.31, 4.0))
    styles = (
        ("-", "o"),
        ("--", "s"),
        ("-.", "^"),
        (":", "D"),
        ((0, (5, 2, 1, 2)), "v"),
    )
    for budget, (linestyle, marker) in zip(BUDGETS, styles, strict=True):
        subset = convergence.loc[
            (convergence["regime"] == regime)
            & (convergence["observed_ports"] == budget)
        ].sort_values("trial_position")
        if len(subset) != EXPECTED_TRIALS:
            raise ValueError(
                f"Convergence is incomplete for {regime}, N={budget}."
            )
        axis.plot(
            subset["trial_position"],
            subset["best_int8_validation_op"],
            color="black",
            linestyle=linestyle,
            marker=marker,
            markevery=10,
            markersize=3.2,
            markerfacecolor="white",
            linewidth=0.9,
            label=f"$N={budget}$",
        )
    axis.set_xlim(1, 100)
    axis.set_title(f"{difficulty}\n{_regime_display(regime)}")
    axis.set_xlabel("Scheduled trial position")
    axis.set_ylabel(r"Best INT8 validation $P_{\mathrm{out}}$")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    _style_axis(axis)
    axis.legend(frameon=True, fancybox=False, edgecolor="black", ncol=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_representative_quantization(
    selected: pd.DataFrame,
    regime: str,
    difficulty: str,
    output_path: Path,
) -> None:
    _set_plot_style()
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    markers = ("o", "s", "^", "D", "v")
    subset = selected.loc[selected["regime"] == regime]
    if len(subset) != len(BUDGETS):
        raise ValueError(f"Quantization data are incomplete for {regime}.")
    minimum = float(
        min(
            subset["fp32_validation_op"].min(),
            subset["int8_validation_op"].min(),
        )
    )
    maximum = float(
        max(
            subset["fp32_validation_op"].max(),
            subset["int8_validation_op"].max(),
        )
    )
    padding = max((maximum - minimum) * 0.08, maximum * 0.002)
    lower = max(minimum - padding, np.finfo(float).tiny)
    upper = maximum + padding
    axis.plot([lower, upper], [lower, upper], color="0.45", linestyle="--")
    for budget, marker in zip(BUDGETS, markers, strict=True):
        row = subset.loc[subset["observed_ports"] == budget]
        axis.scatter(
            row["fp32_validation_op"],
            row["int8_validation_op"],
            marker=marker,
            s=32,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            label=f"$N={budget}$",
        )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_title(f"{difficulty}\n{_regime_display(regime)}")
    axis.set_xlabel(r"FP32 validation $P_{\mathrm{out}}$")
    axis.set_ylabel(r"INT8 validation $P_{\mathrm{out}}$")
    axis.ticklabel_format(axis="both", style="plain", useOffset=False)
    _style_axis(axis)
    axis.legend(frameon=True, fancybox=False, edgecolor="black", ncol=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_op_curve(selected: pd.DataFrame, regime: str, output_path: Path) -> None:
    subset = selected.loc[selected["regime"] == regime].sort_values("observed_ports")
    if len(subset) != len(BUDGETS):
        raise ValueError(f"OP curve is incomplete for {regime}.")
    _set_plot_style()
    figure, axis = plt.subplots(figsize=(13.2 / 2.54, 9.6 / 2.54))
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.99)
    x = subset["observed_ports"].to_numpy(dtype=int)
    series = (
        ("Ideal", "ideal_test_op", "-", None),
        ("Observed only", "observed_test_op", "-", "*"),
        ("FP32", "fp32_test_op", "--", "o"),
        ("INT8", "int8_test_op", ":", "s"),
    )
    plotted = []
    for label, column, linestyle, marker in series:
        values = subset[column].to_numpy(dtype=float)
        plotted.append(values)
        axis.plot(
            x,
            values,
            color="black",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white" if marker in {"o", "s"} else "black",
            markersize=4.0,
            linewidth=1.0,
            label=label,
        )
    axis.set_xticks(BUDGETS)
    axis.set_xlabel(r"$N$")
    axis.set_ylabel(r"$P_{\mathrm{out}}$")
    _set_op_scale(axis, plotted)
    _style_axis(axis)
    axis.text(
        0.98,
        0.98,
        _regime_annotation(regime),
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=12,
        bbox={
            "boxstyle": "square,pad=0.25",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )
    axis.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        facecolor="white",
        framealpha=1.0,
        borderpad=0.25,
        labelspacing=0.0,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=2.0 / 72.0)
    plt.close(figure)


def _plot_radar(selected: pd.DataFrame, budget: int, output_path: Path) -> None:
    subset = selected.loc[selected["observed_ports"] == budget].set_index("regime")
    if set(subset.index) != set(REGIMES):
        raise ValueError(f"Radar data are incomplete for N={budget}.")
    subset = subset.loc[list(REGIMES)]
    _set_plot_style()
    angles = np.linspace(0.0, 2.0 * np.pi, len(REGIMES), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    figure, axis = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
    series = (
        ("Ideal", "ideal_test_op", "-", None),
        ("Observed only", "observed_test_op", "-", "*"),
        ("FP32", "fp32_test_op", "--", "o"),
        ("INT8", "int8_test_op", ":", "s"),
    )
    for label, column, linestyle, marker in series:
        values = subset[column].to_numpy(dtype=float)
        closed_values = np.concatenate([values, values[:1]])
        line_color = RADAR_COLORS[label]
        axis.plot(
            closed_angles,
            closed_values,
            color=line_color,
            linestyle=linestyle,
            marker=None,
            linewidth=1.8 if label != "Ideal" else 1.0,
            label=label,
        )
        if label == "Observed only":
            axis.fill(
                closed_angles,
                closed_values,
                color=RADAR_COLORS["Observed only"],
                alpha=0.10,
            )
    axis.set_xticks(angles)
    axis.set_xticklabels(REGIME_LABELS)
    axis.tick_params(axis="x", pad=10)
    axis.grid(True, color="0.80", linestyle="--", linewidth=0.45)
    axis.spines["polar"].set_color("0.55")
    axis.spines["polar"].set_linewidth(0.45)
    axis.set_yscale("log")
    axis.set_ylim(0.001, 1.0)
    axis.set_yticks((0.01, 0.1, 1.0))
    axis.set_yticklabels((r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"))
    axis.set_rlabel_position(12.8)
    axis.tick_params(axis="y", pad=-2.0)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_representative_pareto(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    regime: str,
    difficulty: str,
    output_path: Path,
) -> None:
    _set_plot_style()
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    markers = ("o", "s", "^", "D", "v")
    for budget, marker in zip(BUDGETS, markers, strict=True):
        group = candidates.loc[
            (candidates["regime"] == regime)
            & (candidates["observed_ports"] == budget)
        ]
        pareto = group.loc[group["is_pareto"]].sort_values(
            "trained_ram_peak_bytes"
        )
        axis.scatter(
            group["trained_ram_peak_bytes"] / 1024.0,
            group["int8_validation_op"],
            s=6,
            color="0.78",
            linewidths=0.0,
        )
        axis.plot(
            pareto["trained_ram_peak_bytes"] / 1024.0,
            pareto["int8_validation_op"],
            color="black",
            marker=marker,
            markerfacecolor="white",
            markersize=3.5,
            linewidth=0.8,
            label=f"$N={budget}$",
        )
        selected_row = selected.loc[
            (selected["regime"] == regime)
            & (selected["observed_ports"] == budget)
        ].iloc[0]
        axis.scatter(
            selected_row["trained_ram_peak_bytes"] / 1024.0,
            selected_row["int8_validation_op"],
            marker="*",
            s=45,
            color="black",
            zorder=4,
        )
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_title(f"{difficulty}\n{_regime_display(regime)}")
    axis.set_xlabel("Estimated peak RAM [KiB]")
    axis.set_ylabel(r"INT8 validation $P_{\mathrm{out}}$")
    _style_axis(axis)
    axis.legend(frameon=True, fancybox=False, edgecolor="black", ncol=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def generate_figures(
    summaries: dict[str, pd.DataFrame],
    figure_dir: Path = FIGURE_DIR,
) -> list[Path]:
    """Generate the approved Chapter 6 and Appendix C PDF figures.

    Args:
        summaries (dict[str, pd.DataFrame]): Validated analysis summaries.
        figure_dir (Path): Dissertation PDF figure directory.

    Returns:
        list[Path]: Paths to all generated PDF assets.

    Raises:
        OSError: If a PDF cannot be written.
        ValueError: If a required plot group is incomplete.
    """
    selected = summaries["selected_models"]
    paths = []
    for figure_number, (regime, slug, difficulty) in enumerate(
        REPRESENTATIVE_REGIMES,
        start=22,
    ):
        path = figure_dir / f"fig6.{figure_number}_convergence_{slug}.pdf"
        _plot_representative_convergence(
            summaries["convergence"],
            regime,
            difficulty,
            path,
        )
        paths.append(path)
    for figure_number, (regime, slug, difficulty) in enumerate(
        REPRESENTATIVE_REGIMES,
        start=25,
    ):
        path = figure_dir / f"fig6.{figure_number}_pareto_{slug}.pdf"
        _plot_representative_pareto(
            summaries["completed_candidates"],
            selected,
            regime,
            difficulty,
            path,
        )
        paths.append(path)
    for figure_number, (regime, slug, difficulty) in enumerate(
        REPRESENTATIVE_REGIMES,
        start=28,
    ):
        path = figure_dir / f"fig6.{figure_number}_quantization_{slug}.pdf"
        _plot_representative_quantization(selected, regime, difficulty, path)
        paths.append(path)
    for figure_number, (regime, slug, _) in enumerate(REPRESENTATIVE_REGIMES, start=31):
        path = figure_dir / f"fig6.{figure_number}_op_per_regime_nas_{slug}.pdf"
        _plot_op_curve(selected, regime, path)
        paths.append(path)
    radar_path = figure_dir / "fig6.34.pdf"
    _plot_radar(selected, 3, radar_path)
    paths.append(radar_path)
    for offset, regime in enumerate(REGIMES):
        path = figure_dir / (
            f"figC.{78 + offset}_op_per_regime_nas_{_regime_slug(regime)}.pdf"
        )
        _plot_op_curve(selected, regime, path)
        paths.append(path)
    for offset, budget in enumerate(BUDGETS):
        path = figure_dir / f"figC.{92 + offset}.pdf"
        _plot_radar(selected, budget, path)
        paths.append(path)
    return paths


def _format_range(minimum: float, median: float, maximum: float, decimals: int = 1) -> str:
    return f"{minimum:.{decimals}f}--{median:.{decimals}f}--{maximum:.{decimals}f}"


def _write_appendix_b_tables(summaries: dict[str, pd.DataFrame], path: Path) -> None:
    budget = summaries["budget_summary"]
    selected = summaries["selected_models"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Per-Regime MobileNetV3 NAS Execution and Selected-Model Complexity by Observation Budget.}",
            r"\label{tabB.6}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{c r r r r c c}",
        r"\hline",
        r"$N$ & Scheduled & Finalized & Unfinalized & Pareto & Parameters min--median--max & RAM [KiB] min--median--max \\",
        r"\hline",
    ]
    for row in budget.itertuples(index=False):
        parameter_range = f"{row.parameters_min:,}--{row.parameters_median:,}--{row.parameters_max:,}"
        ram_range = _format_range(row.ram_kib_min, row.ram_kib_median, row.ram_kib_max)
        lines.append(
            f"{row.observed_ports} & {row.scheduled_trials:,} & {row.complete_trials:,} & "
            f"{row.unfinalized_trials} & {row.pareto_candidates} & {parameter_range} & {ram_range} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\endgroup",
            r"\end{table}",
            "",
            r"\begingroup",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\setlength{\LTcapwidth}{\linewidth}",
            r"\captionof{table}{Selected MobileNetV3 Per-Regime NAS Configurations and Host-Side Measurements.}",
            r"\label{tabB.7}",
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.23\linewidth} >{\centering\arraybackslash}p{0.035\linewidth} >{\centering\arraybackslash}p{0.045\linewidth} *{2}{>{\centering\arraybackslash}p{0.04\linewidth}} >{\centering\arraybackslash}p{0.07\linewidth} >{\centering\arraybackslash}p{0.07\linewidth} >{\centering\arraybackslash}p{0.085\linewidth} >{\centering\arraybackslash}p{0.09\linewidth} *{2}{>{\centering\arraybackslash}p{0.07\linewidth}}@{}}",
            r"\hline",
            r"Regime & $N$ & Trial & $\alpha$ & Cut & Head & LR & INT8 OP & Params & RAM [KiB] & INT8 [KiB] \\",
            r"\hline",
            r"\endfirsthead",
            r"\multicolumn{11}{c}{\tablename~\ref{tabB.7} -- continued from previous page} \\",
            r"\hline",
            r"Regime & $N$ & Trial & $\alpha$ & Cut & Head & LR & INT8 OP & Params & RAM [KiB] & INT8 [KiB] \\",
            r"\hline",
            r"\endhead",
            r"\hline",
            r"\multicolumn{11}{r}{Continued on next page} \\",
            r"\endfoot",
            r"\hline",
            r"\endlastfoot",
        ]
    )
    for row in selected.itertuples(index=False):
        block_match = re.search(r"v3_block_(\d+)", row.truncation_block_name)
        if block_match is None:
            raise ValueError(f"Cannot parse truncation block {row.truncation_block_name}.")
        lines.append(
            f"{_regime_display(row.regime)} & {row.observed_ports} & {row.trial_position} & "
            f"{row.alpha:.1f} & {block_match.group(1)} & {row.head_depth}$\\times${row.head_width} & "
            f"{row.learning_rate:.2e} & {row.int8_validation_op:.6f} & {row.num_params:,} & "
            f"{row.trained_ram_peak_bytes / 1024.0:.1f} & {row.qt_model_size_bytes / 1024.0:.1f} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_appendix_c_results(summaries: dict[str, pd.DataFrame], path: Path) -> None:
    selected = summaries["selected_models"]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\captionof{table}{Selected MobileNetV3 Per-Regime Model-Only Held-Out Outage Probability by Fading Regime and Observation Budget.}",
        r"\label{tabC.5}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.27\linewidth} >{\centering\arraybackslash}p{0.05\linewidth} *{4}{>{\centering\arraybackslash}p{0.13\linewidth}}@{}}",
        r"\hline",
        r"Regime & $N$ & Ideal & Observed only & FP32 model & INT8 model \\",
        r"\hline",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename~\ref{tabC.5} -- continued from previous page} \\",
        r"\hline",
        r"Regime & $N$ & Ideal & Observed only & FP32 model & INT8 model \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{_regime_display(row.regime)} & {row.observed_ports} & {row.ideal_test_op:.6f} & "
            f"{row.observed_test_op:.6f} & {row.fp32_test_op:.6f} & {row.int8_test_op:.6f} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    for offset, regime in enumerate(REGIMES):
        number = 78 + offset
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.80\linewidth]{{Figuras/pdf/figC.{number}_op_per_regime_nas_{_regime_slug(regime)}.pdf}}",
                rf"\caption[Selected per-regime MobileNetV3 OP for {_regime_display(regime)}.]{{Selected per-regime MobileNetV3 held-out \displayac{{op}} for {_regime_display(regime)}.}}",
                rf"\label{{figC.{number}}}",
                r"\end{figure}",
                "",
            ]
        )
    for offset, budget in enumerate(BUDGETS):
        number = 92 + offset
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.80\linewidth]{{Figuras/pdf/figC.{number}.pdf}}",
                rf"\caption[Selected per-regime MobileNetV3 OP across fading regimes for N={budget}.]{{Selected per-regime MobileNetV3 \displayac{{op}} across 14 fading regimes for $N={budget}$ observed ports. The radial axis is logarithmic, and smaller radii indicate lower \displayac{{op}}.}}",
                rf"\label{{figC.{number}}}",
                r"\end{figure}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_appendix_c_comparison(
    summaries: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    comparison = summaries["comparison_by_regime"]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\captionof{table}{INT8 All-Regime and Per-Regime Held-Out Outage Probability Comparison.}",
        r"\label{tabC.7}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.27\linewidth} >{\centering\arraybackslash}p{0.05\linewidth} *{2}{>{\centering\arraybackslash}p{0.12\linewidth}} >{\centering\arraybackslash}p{0.105\linewidth} >{\centering\arraybackslash}p{0.155\linewidth}@{}}",
        r"\hline",
        r"Regime & $N$ & All-regime OP & Per-regime OP & Per $-$ all & Lower-OP campaign \\",
        r"\hline",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename~\ref{tabC.7} -- continued from previous page} \\",
        r"\hline",
        r"Regime & $N$ & All-regime OP & Per-regime OP & Per $-$ all & Lower-OP campaign \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for row in comparison.itertuples(index=False):
        campaign = row.lower_op_campaign.replace("-", "--")
        lines.append(
            f"{_regime_display(row.regime)} & {row.observed_ports} & "
            f"{row.all_regime_int8_op:.6f} & {row.per_regime_int8_op:.6f} & "
            f"{row.int8_op_delta:+.6f} & {campaign} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_inputs(
    summaries: dict[str, pd.DataFrame],
    appendix_dir: Path = APPENDIX_DIR,
) -> list[Path]:
    """Write generated Appendix B and C LaTeX tables and figure blocks.

    Args:
        summaries (dict[str, pd.DataFrame]): Validated analysis summaries.
        appendix_dir (Path): Directory receiving generated LaTeX inputs.

    Returns:
        list[Path]: Generated LaTeX input paths.

    Raises:
        OSError: If an output file cannot be written.
        ValueError: If a selected truncation identifier is malformed.
    """
    appendix_dir.mkdir(parents=True, exist_ok=True)
    appendix_b_path = appendix_dir / "per_regime_nas_search_tables.tex"
    appendix_c_path = appendix_dir / "per_regime_nas_selected_results.tex"
    comparison_path = appendix_dir / "per_regime_nas_comparison.tex"
    _write_appendix_b_tables(summaries, appendix_b_path)
    _write_appendix_c_results(summaries, appendix_c_path)
    _write_appendix_c_comparison(summaries, comparison_path)
    return [appendix_b_path, appendix_c_path, comparison_path]


def analyze_per_regime_nas_results(
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURE_DIR,
    appendix_dir: Path = APPENDIX_DIR,
) -> dict[str, list[Path]]:
    """Run the complete per-regime NAS dissertation analysis.

    Args:
        results_root (Path): Authoritative copied per-regime result root.
        output_dir (Path): Directory receiving aggregate CSV files.
        figure_dir (Path): Directory receiving dissertation PDF figures.
        appendix_dir (Path): Directory receiving generated LaTeX inputs.

    Returns:
        dict[str, list[Path]]: Generated CSV, figure, and LaTeX paths.

    Raises:
        FileNotFoundError: If required campaign evidence is missing.
        KeyError: If an evaluation schema is incomplete.
        OSError: If an output cannot be written.
        ValueError: If any campaign contract or result is inconsistent.
    """
    outputs = load_per_regime_results(results_root)
    summaries = build_summaries(outputs)
    comparison = build_all_regime_comparison(summaries["selected_models"])
    summaries["comparison_by_regime"] = comparison["by_regime"]
    summaries["comparison_summary"] = comparison["summary"]
    csv_paths = write_analysis_outputs(summaries, output_dir)
    figure_paths = generate_figures(summaries, figure_dir)
    latex_paths = write_latex_inputs(summaries, appendix_dir)
    return {
        "csv": csv_paths,
        "figures": figure_paths,
        "latex": latex_paths,
    }


if __name__ == "__main__":
    generated = analyze_per_regime_nas_results()
    for category, paths in generated.items():
        print(f"{category}: {len(paths)} files")
