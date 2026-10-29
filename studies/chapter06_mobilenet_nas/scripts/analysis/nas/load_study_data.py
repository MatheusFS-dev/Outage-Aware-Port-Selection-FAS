"""Load, validate, filter, and enrich NAS study data for analysis."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.common import DEFAULT_OUTPUT_ROOT, build_output_directories


PENALTY_OBJECTIVE_MIN = 1e14
KIB_IN_BYTES = 1024.0
NUMERIC_TRIAL_COLUMNS = (
    "best_not_qt_val_metric",
    "best_qt_val_metric",
    "qt_model_size_bytes",
    "trained_ram_peak_bytes",
    "tflite_model_size",
    "tflite_peak_ram_bytes",
    "tflite_inference_latency_mean",
    "tflite_inference_latency_peak",
)
TFLITE_NUMERIC_STAT_PATHS = {
    "tflite_model_size": ("model_size",),
    "tflite_peak_ram_bytes": ("peak_ram_bytes",),
    "tflite_inference_latency_mean": ("inference_latency", "mean_seconds"),
    "tflite_inference_latency_peak": ("inference_latency", "max_seconds"),
}
TOP_TABLE_COLUMNS = (
    "rank",
    "trial_id",
    "alpha",
    "truncation_block_name",
    "head_depth",
    "head_width",
    "learning_rate",
    "best_qt_val_metric",
    "best_not_qt_val_metric",
    "qt_model_size_kib",
    "trained_ram_peak_kib",
    "tflite_peak_ram_kib",
    "tflite_inference_latency_mean",
    "tflite_inference_latency_peak",
)
OPTUNA_STORAGE_GLOBS = (
    "study.pkl",
    "*.sqlite3",
    "*.sqlite",
    "*.db",
    "*.log",
    "*.journal",
    "*.jsonl",
)


@dataclass(frozen=True)
class NasStudyContext:
    """Prepared NAS study data shared by the orchestrator and plot scripts."""

    study_run_dir: Path
    study_dir: Path
    config_path: Path
    config: dict[str, Any]
    study_scope: str
    objective_semantics: str
    objective_label: str
    rank_key: str
    rank_order: str
    top_k: int
    selected_trial_id: int
    directories: dict[str, Path]
    raw_trials_df: pd.DataFrame
    filtered_trials_df: pd.DataFrame
    ranked_trials_df: pd.DataFrame
    top_ranked_trials_df: pd.DataFrame
    feasibility_table_df: pd.DataFrame
    pareto_summary_df: pd.DataFrame
    hyperparameter_names: list[str]
    filtering_counts: dict[str, int]


def load_nas_study_context(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> NasStudyContext:
    """Load one NAS study directory and prepare all shared analysis tables.

    Args:
        study_run_dir (Path): Top-level NAS study output directory containing an
            ``optuna_study`` subdirectory. The directory must include
            ``optuna_study/config.json``, ranked ``top_*_trial.txt`` summaries,
            and one supported Optuna storage artifact so full trial metadata can
            be reconstructed.
        output_root_dir (Path): Root output directory for the derived analysis
            artifacts. The same prepared context can be reused by the full
            orchestrator and each single-plot script, so callers may direct
            them to the same output tree or to separate temporary roots.

    Returns:
        NasStudyContext: Prepared study metadata, filtered and derived trial
        dataframes, mandatory tables, output directories, and plot-selection
        metadata.

    Raises:
        FileNotFoundError: If required study metadata, ranked summaries, or a
            supported Optuna storage artifact is missing.
        ValueError: If the study metadata is malformed, the storage contains no
            unique study, or the trial dataframe cannot satisfy the analysis
            contract.

    Examples:
        >>> isinstance(DEFAULT_OUTPUT_ROOT, Path)
        True
    """
    resolved_study_run_dir = study_run_dir.resolve()
    config_path = resolved_study_run_dir / "optuna_study" / "config.json"
    config = load_nas_config(resolved_study_run_dir)
    study_scope = detect_study_scope(config)
    ranked_trials_df = load_ranked_trials(config_path.parent, config["artifact_subdirs"]["args"])
    raw_trials_df = load_trial_dataframe(config_path.parent)
    objective_semantics = (
        "model_only_op"
        if int(config.get("schema_version", 1)) >= 2
        else "validation_mse"
    )
    normalized_trials_df = normalize_trial_dataframe(
        raw_trials_df,
        ranked_trials_df,
        objective_semantics=objective_semantics,
    )
    filtered_trials_df, filtering_counts = apply_filtering_contract(normalized_trials_df)
    derived_trials_df = add_derived_metrics(filtered_trials_df)
    top_ranked_trials_df = build_top_ranked_trials_table(
        trials_df=derived_trials_df,
        ranked_trials_df=ranked_trials_df,
        top_k=int(config["top_k"]),
    )
    feasibility_table_df = build_feasibility_table(
        raw_trials_df=normalized_trials_df,
        filtered_trials_df=derived_trials_df,
    )
    pareto_summary_df = build_pareto_summary_table(derived_trials_df)
    directories = build_output_directories(output_root_dir=output_root_dir)
    selected_trial_id = int(ranked_trials_df.sort_values("rank", kind="mergesort").iloc[0]["trial_id"])
    return NasStudyContext(
        study_run_dir=resolved_study_run_dir,
        study_dir=config_path.parent,
        config_path=config_path,
        config=config,
        study_scope=study_scope,
        objective_semantics=objective_semantics,
        objective_label=(
            "Validation model-only OP"
            if objective_semantics == "model_only_op"
            else "Validation MSE"
        ),
        rank_key=str(config["rank_key"]),
        rank_order=str(config["order"]),
        top_k=int(config["top_k"]),
        selected_trial_id=selected_trial_id,
        directories=directories,
        raw_trials_df=normalized_trials_df,
        filtered_trials_df=derived_trials_df,
        ranked_trials_df=ranked_trials_df,
        top_ranked_trials_df=top_ranked_trials_df,
        feasibility_table_df=feasibility_table_df,
        pareto_summary_df=pareto_summary_df,
        hyperparameter_names=discover_searched_hyperparameters(derived_trials_df),
        filtering_counts=filtering_counts,
    )


def load_nas_config(study_run_dir: Path) -> dict[str, Any]:
    """Load and validate the strict NAS study configuration JSON.

    Args:
        study_run_dir (Path): Top-level NAS study output directory containing
            ``optuna_study/config.json``. The loader trusts this file rather
            than guessing study scope from the directory name.

    Returns:
        dict[str, Any]: Parsed JSON payload with the required study metadata.

    Raises:
        FileNotFoundError: If ``optuna_study/config.json`` is missing.
        ValueError: If the file is invalid JSON or missing required keys.
    """
    config_path = study_run_dir / "optuna_study" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing NAS config at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as config_file:
        try:
            config = json.load(config_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {config_path}: {error}") from error

    required_keys = {
        "regime_label",
        "observed_ports_budget",
        "train_regime_labels",
        "top_k",
        "rank_key",
        "order",
        "artifact_subdirs",
    }
    missing_keys = sorted(required_keys.difference(config))
    if missing_keys:
        raise ValueError(f"NAS config at {config_path} is missing required keys: {missing_keys}.")
    return config


def detect_study_scope(config: dict[str, Any]) -> str:
    """Detect whether a NAS config describes a per-regime or all-regimes study.

    Args:
        config (dict[str, Any]): Parsed NAS ``config.json`` payload. The
            function expects ``regime_label`` and ``train_regime_labels`` to be
            present and internally consistent.

    Returns:
        str: ``"per_regime"`` when the study optimized one regime and
        ``"all_regimes"`` when it optimized multiple regimes together.

    Raises:
        ValueError: If the config does not describe either supported study
            scope cleanly.
    """
    regime_label = str(config["regime_label"])
    train_regime_labels = list(config["train_regime_labels"])
    if regime_label == "all_regimes" or len(train_regime_labels) > 1:
        return "all_regimes"
    if len(train_regime_labels) == 1:
        return "per_regime"
    raise ValueError("Unable to detect study scope from NAS config.")


def load_ranked_trials(study_dir: Path, args_subdir: str) -> pd.DataFrame:
    """Parse the ranked ``top_*_trial.txt`` summaries for one NAS study.

    Args:
        study_dir (Path): Path to the ``optuna_study`` directory containing the
            ranked summary subdirectory.
        args_subdir (str): Relative directory name holding ``top_*_trial.txt``
            files. This comes from ``config.json["artifact_subdirs"]["args"]``.

    Returns:
        pd.DataFrame: Ranked-trial table with the columns ``rank``, ``trial_id``,
        and ``summary_path`` sorted by ascending rank.

    Raises:
        FileNotFoundError: If the summary directory does not exist.
        ValueError: If no ranked summaries exist or the file contents are
            inconsistent with the filename ranks.
    """
    args_dir = study_dir / args_subdir
    if not args_dir.exists():
        raise FileNotFoundError(f"Missing ranked-trial summary directory at {args_dir}.")

    ranked_rows = []
    for summary_path in sorted(args_dir.glob("top_*_trial.txt")):
        rank = None
        trial_id = None
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Rank:"):
                rank = int(line.split(":", 1)[1].strip())
            elif line.startswith("Trial ID:"):
                trial_id = int(line.split(":", 1)[1].strip())
            if rank is not None and trial_id is not None:
                break
        if rank is None or trial_id is None:
            raise ValueError(f"Summary file {summary_path} must contain Rank and Trial ID lines.")

        match = re.fullmatch(r"top_(\d+)_trial\.txt", summary_path.name)
        if match is not None and int(match.group(1)) != rank:
            raise ValueError(f"Rank mismatch between filename and contents in {summary_path}.")

        ranked_rows.append(
            {
                "rank": int(rank),
                "trial_id": int(trial_id),
                "summary_path": str(summary_path),
            }
        )

    if not ranked_rows:
        raise ValueError(f"No ranked ``top_*_trial.txt`` summaries found in {args_dir}.")

    ranked_df = pd.DataFrame(ranked_rows).sort_values("rank", kind="mergesort").reset_index(drop=True)
    if ranked_df["rank"].duplicated().any():
        raise ValueError(f"Duplicate ranked-trial entries found in {args_dir}.")
    return ranked_df


def load_trial_dataframe(study_dir: Path) -> pd.DataFrame:
    """Load the full Optuna trial table for one NAS study.

    Args:
        study_dir (Path): Path to the ``optuna_study`` directory. The function
            searches only this directory for one supported Optuna storage
            artifact because the analysis requires the complete trial history,
            not only the top-k summaries.

    Returns:
        pd.DataFrame: One row per Optuna trial with flattened params, user
        attributes, state, and objective values.

    Raises:
        FileNotFoundError: If no supported Optuna storage artifact is found.
        ValueError: If the storage contains zero studies, multiple studies, or
            no trial rows.
    """
    storage_kind, storage_path = discover_storage_artifact(study_dir)
    study = load_optuna_study(storage_kind, storage_path)
    flattened_rows = []
    for trial in study.trials:
        trial_row: dict[str, Any] = {
            "trial_id": int(trial.number),
            "state": getattr(trial.state, "name", str(trial.state)),
            "datetime_start": (
                trial.datetime_start.isoformat(sep=" ")
                if trial.datetime_start is not None
                else None
            ),
            "datetime_complete": (
                trial.datetime_complete.isoformat(sep=" ")
                if trial.datetime_complete is not None
                else None
            ),
            "duration_seconds": (
                (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_start is not None and trial.datetime_complete is not None
                else None
            ),
        }
        if trial.values is not None:
            for objective_index, objective_value in enumerate(trial.values):
                trial_row[f"objective_{objective_index}"] = objective_value
        elif trial.value is not None:
            trial_row["objective_0"] = trial.value
        for param_name, param_value in trial.params.items():
            trial_row[param_name] = param_value
        for attr_name, attr_value in trial.user_attrs.items():
            trial_row[attr_name] = attr_value
        flattened_rows.append(trial_row)

    if not flattened_rows:
        raise ValueError(f"Optuna study in {storage_path} contains no trial rows.")
    return pd.DataFrame(flattened_rows).sort_values("trial_id", kind="mergesort").reset_index(drop=True)


def discover_storage_artifact(study_dir: Path) -> tuple[str, Path]:
    """Discover exactly one supported Optuna storage artifact for a study.

    Args:
        study_dir (Path): Path to the ``optuna_study`` directory. Only the top
            level is scanned because the analysis should fail loudly when the
            study history is not stored in the standard study root.

    Returns:
        tuple[str, Path]: Storage kind and artifact path. Supported kinds are
        ``"pickle"``, ``"sqlite"``, and ``"journal"``.

    Raises:
        FileNotFoundError: If no supported storage artifact is found.
        ValueError: If multiple candidate artifacts are found and the loader
            cannot choose one unambiguously.
    """
    candidates: list[tuple[str, Path]] = []
    for pattern in OPTUNA_STORAGE_GLOBS:
        for candidate_path in sorted(study_dir.glob(pattern)):
            if candidate_path.name == "config.json":
                continue
            suffixes = candidate_path.suffixes
            if candidate_path.name == "study.pkl":
                candidates.append(("pickle", candidate_path))
            elif any(suffix in {".sqlite3", ".sqlite", ".db"} for suffix in suffixes):
                candidates.append(("sqlite", candidate_path))
            elif any(suffix in {".log", ".journal", ".jsonl"} for suffix in suffixes):
                candidates.append(("journal", candidate_path))

    if not candidates:
        searched = ", ".join(OPTUNA_STORAGE_GLOBS)
        raise FileNotFoundError(
            f"No supported Optuna storage artifact found in {study_dir}. "
            f"Searched for: {searched}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one Optuna storage artifact in {study_dir}, "
            f"found {[str(path) for _, path in candidates]}."
        )
    return candidates[0]


def load_optuna_study(storage_kind: str, storage_path: Path) -> optuna.study.Study:
    """Load one Optuna study object from a supported local storage artifact.

    Args:
        storage_kind (str): Storage selector returned by
            ``discover_storage_artifact``. ``"pickle"`` loads a pickled study
            object, ``"sqlite"`` loads a database-backed study, and
            ``"journal"`` loads a file-backed journal study.
        storage_path (Path): Path to the selected local storage artifact.

    Returns:
        optuna.study.Study: Loaded Optuna study with the full trial history.

    Raises:
        FileNotFoundError: If ``storage_path`` does not exist.
        ValueError: If the artifact does not contain exactly one study or is of
            an unsupported kind.
    """
    if not storage_path.exists():
        raise FileNotFoundError(f"Missing Optuna storage artifact at {storage_path}.")

    if storage_kind == "pickle":
        with open(storage_path, "rb") as storage_file:
            study = pickle.load(storage_file)
        if not isinstance(study, optuna.study.Study):
            raise ValueError(f"Pickle at {storage_path} does not contain an Optuna Study.")
        return study

    if storage_kind == "sqlite":
        storage = f"sqlite:///{storage_path.as_posix()}"
    elif storage_kind == "journal":
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(storage_path))
        )
    else:
        raise ValueError(f"Unsupported Optuna storage kind: {storage_kind}.")

    study_summaries = optuna.get_all_study_summaries(storage=storage)
    if len(study_summaries) != 1:
        raise ValueError(
            f"Expected exactly one study in {storage_path}, found {len(study_summaries)}."
        )
    return optuna.load_study(study_name=study_summaries[0].study_name, storage=storage)


def normalize_trial_dataframe(
    trials_df: pd.DataFrame,
    ranked_trials_df: pd.DataFrame,
    objective_semantics: str = "validation_mse",
) -> pd.DataFrame:
    """Normalize trial column types and attach ranked-trial membership fields.

    Args:
        trials_df (pd.DataFrame): Raw flattened trial dataframe returned by
            ``load_trial_dataframe``. The dataframe may contain a mix of numeric
            and string columns because Optuna params and user attributes are
            flattened into one table.
        ranked_trials_df (pd.DataFrame): Ranked top-trial dataframe returned by
            ``load_ranked_trials``. It is merged on ``trial_id`` to mark top-k
            membership and the selected final trial.

    Returns:
        pd.DataFrame: Trial dataframe with normalized numeric columns plus the
        added columns ``rank``, ``is_top_k``, and ``is_selected_trial``.

    Raises:
        KeyError: If the required ``trial_id`` column is missing.
    """
    normalized_df = trials_df.copy()
    if "trial_id" not in normalized_df.columns:
        raise KeyError("Trial dataframe must contain a trial_id column.")

    if "ram_budget_violated" not in normalized_df.columns:
        normalized_df["ram_budget_violated"] = np.nan
    normalized_df["ram_budget_violated"] = normalized_df["ram_budget_violated"].map(_normalize_bool_like)

    if objective_semantics == "model_only_op":
        source_columns = (
            "best_not_qt_val_model_op",
            "best_qt_val_model_op",
        )
    elif objective_semantics == "validation_mse":
        source_columns = (
            "best_not_qt_val_mse",
            "best_qt_val_mse",
        )
    else:
        raise ValueError(
            f"Unsupported objective_semantics={objective_semantics!r}."
        )
    for source_column, canonical_column in zip(
        source_columns,
        ("best_not_qt_val_metric", "best_qt_val_metric"),
        strict=True,
    ):
        if source_column not in normalized_df.columns:
            normalized_df[source_column] = np.nan
        normalized_df[canonical_column] = normalized_df[source_column]
    normalized_df["objective_semantics"] = objective_semantics

    for column_name in NUMERIC_TRIAL_COLUMNS + ("objective_0", "objective_1"):
        if column_name not in normalized_df.columns:
            normalized_df[column_name] = np.nan
        normalized_df[column_name] = pd.to_numeric(normalized_df[column_name], errors="coerce")

    if "tflite_model_stats" in normalized_df.columns:
        for column_name, key_path in TFLITE_NUMERIC_STAT_PATHS.items():
            structured_values = normalized_df["tflite_model_stats"].map(
                lambda stats, path=key_path: _extract_nested_numeric_value(stats, path)
            )
            normalized_df[column_name] = normalized_df[column_name].fillna(structured_values)

    ranked_subset = ranked_trials_df[["rank", "trial_id"]].copy()
    normalized_df = normalized_df.merge(ranked_subset, on="trial_id", how="left")
    normalized_df["is_top_k"] = normalized_df["rank"].notna()
    normalized_df["is_selected_trial"] = normalized_df["rank"] == 1
    return normalized_df.sort_values("trial_id", kind="mergesort").reset_index(drop=True)


def _extract_nested_numeric_value(value: Any, key_path: tuple[str, ...]) -> float:
    """Extract one numeric value from a nested Optuna user attribute."""
    current_value = value
    for key in key_path:
        if not isinstance(current_value, dict) or key not in current_value:
            return np.nan
        current_value = current_value[key]

    try:
        return float(current_value)
    except (TypeError, ValueError):
        return np.nan


def _normalize_bool_like(value: Any) -> bool | float:
    """Normalize one Optuna user-attribute value to a boolean when possible.

    Args:
        value (Any): Raw flattened user-attribute value. The function accepts
            real booleans plus common string and integer encodings used in CSV
            or JSON intermediates.

    Returns:
        bool | float: ``True`` or ``False`` when the value is recognized as a
        boolean encoding, otherwise ``np.nan`` so the caller can treat the
        field as missing rather than guessing.

    Raises:
        None.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized_text = value.strip().lower()
        if normalized_text in {"true", "1"}:
            return True
        if normalized_text in {"false", "0"}:
            return False
    return np.nan


def apply_filtering_contract(trials_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the strict NAS analysis filtering contract to the trial table.

    Args:
        trials_df (pd.DataFrame): Normalized trial dataframe containing state,
        objective values, and flattened user attributes. The function keeps the
        original columns and removes only rows that violate the documented
        analysis contract.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: Filtered trial dataframe and a
        dictionary containing per-rule row counts for logging. The output rows
        always satisfy the contract exactly; rows that do not are excluded
        rather than repaired.

    Raises:
        ValueError: If the filtering contract removes every row, or if the
            remaining rows cannot support quantization-degradation analysis.
    """
    state_mask = trials_df["state"] == "COMPLETE"
    penalty_mask = trials_df.apply(has_penalty_objective_values, axis=1)
    ram_mask = trials_df["ram_budget_violated"].fillna(True).astype(bool)
    missing_best_qt_mask = trials_df["best_qt_val_metric"].isna()
    numeric_mask = (
        trials_df["qt_model_size_bytes"].notna()
        & trials_df["trained_ram_peak_bytes"].notna()
        & trials_df["tflite_model_size"].notna()
        & trials_df["tflite_peak_ram_bytes"].notna()
        & trials_df["tflite_inference_latency_mean"].notna()
        & trials_df["tflite_inference_latency_peak"].notna()
    )
    valid_mask = state_mask & (~penalty_mask) & (~ram_mask) & (~missing_best_qt_mask) & numeric_mask
    filtered_df = trials_df.loc[valid_mask].copy()
    if filtered_df.empty:
        raise ValueError("Filtering contract removed every trial from the NAS study.")

    if filtered_df["best_not_qt_val_metric"].isna().any():
        raise ValueError(
            "Filtered trials must contain the FP32 validation metric."
        )

    filtering_counts = {
        "loaded_trial_count": int(trials_df.shape[0]),
        "completed_trial_count": int(state_mask.sum()),
        "penalty_trial_count": int((state_mask & penalty_mask).sum()),
        "ram_violating_trial_count": int((state_mask & ram_mask).sum()),
        "missing_best_qt_trial_count": int((state_mask & (~penalty_mask) & missing_best_qt_mask).sum()),
        "conversion_failure_count": int((state_mask & (~penalty_mask) & (~ram_mask) & (~numeric_mask)).sum()),
        "valid_trial_count": int(filtered_df.shape[0]),
    }
    return filtered_df.reset_index(drop=True), filtering_counts


def has_penalty_objective_values(trial_row: pd.Series) -> bool:
    """Return whether one trial row contains the penalty objective sentinel.

    Args:
        trial_row (pd.Series): One normalized trial row. The function inspects
        both Optuna objective columns and the mirrored user-attribute metrics
        because some reports are built from attributes while others are built
        from objective tuples.

    Returns:
        bool: ``True`` when any known objective-like field is at or above the
        penalty sentinel range used by the NAS scripts, otherwise ``False``.

    Raises:
        None.
    """
    candidate_columns = (
        "objective_0",
        "objective_1",
        "best_qt_val_metric",
        "trained_ram_peak_bytes",
    )
    for column_name in candidate_columns:
        value = trial_row.get(column_name)
        if pd.notna(value) and float(value) >= PENALTY_OBJECTIVE_MIN:
            return True
    return False


def add_derived_metrics(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Add KiB conversions and derived NAS analysis metrics to valid trials.

    Args:
        trials_df (pd.DataFrame): Filtered valid-trial dataframe returned by
            ``apply_filtering_contract``. All rows must already satisfy the
            documented filtering rules because the derived metrics assume
            numeric RAM, size, latency, and FP32/INT8 loss columns.

    Returns:
        pd.DataFrame: Copy of ``trials_df`` with KiB columns,
        ``quantization_degradation``, and ``is_pareto`` added.

    Raises:
        None.
    """
    derived_df = trials_df.copy()
    derived_df["qt_model_size_kib"] = derived_df["qt_model_size_bytes"] / KIB_IN_BYTES
    derived_df["trained_ram_peak_kib"] = derived_df["trained_ram_peak_bytes"] / KIB_IN_BYTES
    derived_df["tflite_model_size_kib"] = derived_df["tflite_model_size"] / KIB_IN_BYTES
    derived_df["tflite_peak_ram_kib"] = derived_df["tflite_peak_ram_bytes"] / KIB_IN_BYTES
    derived_df["quantization_degradation"] = (
        derived_df["best_qt_val_metric"]
        - derived_df["best_not_qt_val_metric"]
    )
    derived_df["is_pareto"] = compute_pareto_membership(
        x_values=derived_df["trained_ram_peak_kib"].to_numpy(dtype=float),
        y_values=derived_df["best_qt_val_metric"].to_numpy(dtype=float),
    )
    return derived_df.sort_values("trial_id", kind="mergesort").reset_index(drop=True)


def compute_pareto_membership(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Return the Pareto-membership mask for two minimization objectives.

    Args:
        x_values (np.ndarray): One-dimensional array for the first objective.
            Lower values are better; in this analysis it represents peak RAM in
            KiB.
        y_values (np.ndarray): One-dimensional array for the second objective.
            Lower values are better; in this analysis it represents INT8
            validation MSE.

    Returns:
        np.ndarray: Boolean array where ``True`` marks a non-dominated trial.

    Raises:
        ValueError: If the arrays are not one-dimensional or have different
            lengths.

    Examples:
        >>> compute_pareto_membership(np.array([1.0, 2.0]), np.array([2.0, 1.0])).tolist()
        [True, True]
    """
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("Pareto objectives must be one-dimensional arrays.")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("Pareto objective arrays must have the same length.")

    pareto_mask = np.ones(x_values.shape[0], dtype=bool)
    for row_index in range(x_values.shape[0]):
        dominates_mask = (
            (x_values <= x_values[row_index])
            & (y_values <= y_values[row_index])
            & ((x_values < x_values[row_index]) | (y_values < y_values[row_index]))
        )
        dominates_mask[row_index] = False
        if bool(np.any(dominates_mask)):
            pareto_mask[row_index] = False
    return pareto_mask


def build_top_ranked_trials_table(
    trials_df: pd.DataFrame,
    ranked_trials_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build the mandatory top-k trial summary table.

    Args:
        trials_df (pd.DataFrame): Derived valid-trial dataframe returned by
            ``add_derived_metrics``. The dataframe must already contain KiB
            columns and trial params used by the final top-k table.
        ranked_trials_df (pd.DataFrame): Ranked summary dataframe parsed from
            ``top_*_trial.txt`` files. These ranks define the report order and
            selected-final-trial marker.
        top_k (int): Number of ranked summary rows to keep. The value is read
            from the NAS config and limits the final output even if more ranked
            summaries happen to exist on disk.

    Returns:
        pd.DataFrame: Up to ``top_k`` valid ranked trials in the exact required
        column order. Ranked summaries for incomplete or otherwise invalid
        trials are excluded.

    Raises:
        KeyError: If the trial or ranked-summary dataframes omit the columns
            required to identify and order trials.
    """
    valid_trial_ids = set(trials_df["trial_id"])
    requested_ranked_df = ranked_trials_df[
        ranked_trials_df["trial_id"].isin(valid_trial_ids)
    ].sort_values("rank", kind="mergesort").head(top_k).copy()
    trial_columns = [
        column_name
        for column_name in trials_df.columns
        if column_name == "trial_id" or column_name not in ranked_trials_df.columns
    ]
    merged_df = requested_ranked_df.merge(
        trials_df.loc[:, trial_columns],
        on="trial_id",
        how="inner",
    )

    for column_name in TOP_TABLE_COLUMNS:
        if column_name not in merged_df.columns:
            merged_df[column_name] = np.nan
    return merged_df.loc[:, TOP_TABLE_COLUMNS].reset_index(drop=True)


def build_feasibility_table(raw_trials_df: pd.DataFrame, filtered_trials_df: pd.DataFrame) -> pd.DataFrame:
    """Build the one-row NAS feasibility summary table.

    Args:
        raw_trials_df (pd.DataFrame): Normalized pre-filter trial dataframe.
            This dataframe is used to count completed, penalized, RAM-violating,
            and conversion-failure trials across the entire study.
        filtered_trials_df (pd.DataFrame): Derived valid-trial dataframe after
            the filtering contract. The table reports best feasible values using
            only this valid subset.

    Returns:
        pd.DataFrame: One-row feasibility table with the required output
        columns.

    Raises:
        ValueError: If ``filtered_trials_df`` is empty.
    """
    if filtered_trials_df.empty:
        raise ValueError("Cannot build a feasibility table from an empty valid-trial dataframe.")

    completed_mask = raw_trials_df["state"] == "COMPLETE"
    penalty_mask = raw_trials_df.apply(has_penalty_objective_values, axis=1)
    ram_mask = raw_trials_df["ram_budget_violated"].fillna(True).astype(bool)
    numeric_failure_mask = (
        raw_trials_df["qt_model_size_bytes"].isna()
        | raw_trials_df["trained_ram_peak_bytes"].isna()
        | raw_trials_df["tflite_model_size"].isna()
        | raw_trials_df["tflite_peak_ram_bytes"].isna()
        | raw_trials_df["tflite_inference_latency_mean"].isna()
        | raw_trials_df["tflite_inference_latency_peak"].isna()
    )
    feasibility_row = {
        "completed_trial_count": int(completed_mask.sum()),
        "valid_trial_count": int(filtered_trials_df.shape[0]),
        "ram_violating_trial_count": int((completed_mask & ram_mask).sum()),
        "penalty_trial_count": int((completed_mask & penalty_mask).sum()),
        "conversion_failure_count": int((completed_mask & (~penalty_mask) & (~ram_mask) & numeric_failure_mask).sum()),
        "best_feasible_qt_val_metric": float(
            filtered_trials_df["best_qt_val_metric"].min()
        ),
        "best_feasible_ram_kib": float(filtered_trials_df["trained_ram_peak_kib"].min()),
    }
    return pd.DataFrame([feasibility_row])


def build_pareto_summary_table(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Build the mandatory Pareto summary table for all valid trials.

    Args:
        trials_df (pd.DataFrame): Derived valid-trial dataframe returned by
            ``add_derived_metrics``. The dataframe must include the boolean
            ``is_pareto`` column plus the KiB and latency columns requested by
            the plan.

    Returns:
        pd.DataFrame: Pareto summary table sorted by ``trial_id``.

    Raises:
        KeyError: If any required summary column is missing.
    """
    required_columns = [
        "trial_id",
        "is_pareto",
        "best_qt_val_metric",
        "trained_ram_peak_kib",
        "qt_model_size_kib",
        "tflite_inference_latency_mean",
    ]
    missing_columns = [column_name for column_name in required_columns if column_name not in trials_df.columns]
    if missing_columns:
        raise KeyError(f"Pareto summary requires missing columns: {missing_columns}.")
    return trials_df.loc[:, required_columns].sort_values("trial_id", kind="mergesort").reset_index(drop=True)


def discover_searched_hyperparameters(trials_df: pd.DataFrame) -> list[str]:
    """Return the NAS hyperparameters that were actually searched in the study.

    Args:
        trials_df (pd.DataFrame): Derived valid-trial dataframe. Hyperparameters
        are considered "searched" only when their columns exist and contain more
        than one distinct non-null value across the valid trials. This avoids
        plotting fixed parameters as if they were part of the sampled search
        space.

    Returns:
        list[str]: Ordered hyperparameter names that should receive effect plots.

    Raises:
        None.
    """
    candidate_names = [
        "alpha",
        "truncation_block_name",
        "head_depth",
        "head_width",
        "learning_rate",
    ]
    searched_names: list[str] = []
    for candidate_name in candidate_names:
        if candidate_name not in trials_df.columns:
            continue
        non_null_values = trials_df[candidate_name].dropna()
        if non_null_values.nunique(dropna=True) > 1:
            searched_names.append(candidate_name)
    return searched_names
