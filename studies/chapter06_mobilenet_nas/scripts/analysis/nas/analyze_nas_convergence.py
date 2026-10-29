"""Analyze convergence of the five MobileNetV3 all-regime NAS studies."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
from pathlib import Path

import numpy as np
import pandas as pd

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import compute_pareto_membership


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/analyze_nas_convergence.py'
RESULTS_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/nas_pqt_mobv3_all_regimes'
ANALYSIS_DIR = RESULTS_ROOT / "dissertation_analysis"
SEARCH_DATA_PATH = ANALYSIS_DIR / "search_plot_data.csv"
STUDY_SUMMARY_PATH = ANALYSIS_DIR / "study_summary.csv"
CONVERGENCE_PATH = ANALYSIS_DIR / "convergence_by_trial.csv"
PLOT_DATA_PATH = ANALYSIS_DIR / "convergence_plot_data.csv"
SUMMARY_PATH = ANALYSIS_DIR / "convergence_summary.csv"
METADATA_PATH = ANALYSIS_DIR / "convergence_metadata.json"
OBSERVED_PORT_BUDGETS = (3, 4, 5, 6, 7)
SCHEDULED_TRIALS_PER_STUDY = 100
EXPECTED_COMPLETED_TRIALS = 489
EXPECTED_INCOMPLETE_TRIALS = 11
REFERENCE_MARGIN = 0.05


def validate_analysis_inputs(
    search_data: pd.DataFrame,
    study_summary: pd.DataFrame,
) -> None:
    """Validate completed-trial data against the per-study execution summary.

    Args:
        search_data (pd.DataFrame): Completed valid MobileNetV3 trial rows with
            scheduled trial identifiers, validation OP, peak RAM, and selected
            candidate markers.
        study_summary (pd.DataFrame): One execution-summary row for each
            observation budget, including scheduled, completed, valid,
            incomplete, and selected-trial counts.

    Returns:
        None: Validation succeeds without modifying either input table.

    Raises:
        KeyError: If either input omits a required column.
        ValueError: If budgets, counts, trial identifiers, objective values,
            best-so-far values, or selected candidates are inconsistent.

    Examples:
        >>> search = pd.read_csv(SEARCH_DATA_PATH)
        >>> studies = pd.read_csv(STUDY_SUMMARY_PATH)
        >>> validate_analysis_inputs(search, studies)
    """
    required_search_columns = {
        "observed_ports",
        "trial_id",
        "trial_number",
        "int8_validation_op",
        "trained_peak_ram_kib",
        "is_selected",
        "best_so_far_int8_op",
    }
    required_summary_columns = {
        "observed_ports",
        "scheduled_trials",
        "completed_trials",
        "valid_trials",
        "ram_penalized_trials",
        "incomplete_trials",
        "selected_trial",
    }
    missing_search_columns = sorted(required_search_columns - set(search_data.columns))
    missing_summary_columns = sorted(
        required_summary_columns - set(study_summary.columns)
    )
    if missing_search_columns:
        raise KeyError(
            f"Search data is missing columns: {missing_search_columns}."
        )
    if missing_summary_columns:
        raise KeyError(
            f"Study summary is missing columns: {missing_summary_columns}."
        )

    expected_budgets = set(OBSERVED_PORT_BUDGETS)
    if set(search_data["observed_ports"].astype(int)) != expected_budgets:
        raise ValueError("Search data must contain all five observation budgets.")
    if set(study_summary["observed_ports"].astype(int)) != expected_budgets:
        raise ValueError("Study summary must contain all five observation budgets.")
    if len(study_summary) != len(OBSERVED_PORT_BUDGETS):
        raise ValueError("Study summary must contain exactly one row per budget.")
    if not pd.api.types.is_bool_dtype(search_data["is_selected"]):
        raise ValueError("Selected-candidate markers must be Boolean values.")

    numeric_search = search_data[
        ["int8_validation_op", "trained_peak_ram_kib"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_search).all() or (numeric_search <= 0.0).any():
        raise ValueError("Completed-trial OP and RAM objectives must be positive and finite.")

    if set(study_summary["scheduled_trials"].astype(int)) != {
        SCHEDULED_TRIALS_PER_STUDY
    }:
        raise ValueError("Every study must schedule exactly 100 trials.")
    if int(study_summary["completed_trials"].sum()) != EXPECTED_COMPLETED_TRIALS:
        raise ValueError("The five studies must contain 489 completed trials.")
    if int(study_summary["valid_trials"].sum()) != EXPECTED_COMPLETED_TRIALS:
        raise ValueError("All 489 completed trials must be valid.")
    if int(study_summary["ram_penalized_trials"].sum()) != 0:
        raise ValueError("The completed MobileNetV3 studies contain no RAM penalties.")
    if int(study_summary["incomplete_trials"].sum()) != EXPECTED_INCOMPLETE_TRIALS:
        raise ValueError("The five studies must contain 11 incomplete trials.")
    if len(search_data) != EXPECTED_COMPLETED_TRIALS:
        raise ValueError("Search data must contain one row per completed trial.")

    for observed_ports in OBSERVED_PORT_BUDGETS:
        budget_data = search_data[
            search_data["observed_ports"] == observed_ports
        ].sort_values("trial_id")
        budget_summary = study_summary[
            study_summary["observed_ports"] == observed_ports
        ].iloc[0]
        completed_trials = int(budget_summary["completed_trials"])
        incomplete_trials = int(budget_summary["incomplete_trials"])
        if len(budget_data) != completed_trials:
            raise ValueError(
                f"Budget {observed_ports} completed-trial count is inconsistent."
            )
        if completed_trials + incomplete_trials != SCHEDULED_TRIALS_PER_STUDY:
            raise ValueError(
                f"Budget {observed_ports} scheduled-trial accounting is inconsistent."
            )

        trial_ids = budget_data["trial_id"].astype(int)
        if trial_ids.duplicated().any() or not trial_ids.between(0, 99).all():
            raise ValueError(
                f"Budget {observed_ports} trial IDs must be unique values from 0 to 99."
            )
        if not np.array_equal(
            budget_data["trial_number"].to_numpy(dtype=int),
            trial_ids.to_numpy() + 1,
        ):
            raise ValueError(
                f"Budget {observed_ports} trial numbers must equal trial IDs plus one."
            )

        expected_best = budget_data["int8_validation_op"].cummin().to_numpy()
        recorded_best = budget_data["best_so_far_int8_op"].to_numpy(dtype=float)
        if not np.allclose(expected_best, recorded_best, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Budget {observed_ports} best-so-far OP values are inconsistent."
            )

        selected = budget_data[budget_data["is_selected"]]
        if len(selected) != 1:
            raise ValueError(
                f"Budget {observed_ports} must contain exactly one selected trial."
            )
        selected_trial_id = int(selected.iloc[0]["trial_id"])
        if selected_trial_id != int(budget_summary["selected_trial"]):
            raise ValueError(
                f"Budget {observed_ports} selected trial disagrees with the summary."
            )
        if float(selected.iloc[0]["int8_validation_op"]) != float(
            budget_data["int8_validation_op"].min()
        ):
            raise ValueError(
                f"Budget {observed_ports} selected trial is not OP-minimal."
            )


def calculate_normalized_hypervolume(
    outage_probabilities: np.ndarray,
    ram_kib: np.ndarray,
    reference: dict[str, float],
) -> float:
    """Calculate two-objective hypervolume inside a fixed normalized rectangle.

    Args:
        outage_probabilities (np.ndarray): Feasible strict-INT8 validation
            model-only outage probabilities. Lower values are better, and all
            values must be finite.
        ram_kib (np.ndarray): Feasible trained-model peak-RAM estimates in KiB.
            Lower values are better, and the array must align with
            ``outage_probabilities``.
        reference (dict[str, float]): Fixed normalization rectangle containing
            ``best_op``, ``reference_op``, ``best_ram_kib``, and
            ``reference_ram_kib``.

    Returns:
        float: Dominated hypervolume in the normalized unit square.

    Raises:
        ValueError: If the objective arrays are empty, misaligned, non-finite,
            or outside the reference rectangle.

    Examples:
        >>> reference = {"best_op": 0.0, "reference_op": 2.0,
        ...              "best_ram_kib": 0.0, "reference_ram_kib": 2.0}
        >>> calculate_normalized_hypervolume(
        ...     np.array([1.0]), np.array([1.0]), reference
        ... )
        0.25
    """
    if (
        outage_probabilities.ndim != 1
        or ram_kib.ndim != 1
        or len(outage_probabilities) != len(ram_kib)
    ):
        raise ValueError(
            "Hypervolume objectives must be aligned one-dimensional arrays."
        )
    if (
        len(outage_probabilities) == 0
        or not np.isfinite(outage_probabilities).all()
        or not np.isfinite(ram_kib).all()
    ):
        raise ValueError("Hypervolume objectives must be nonempty and finite.")

    op_range = reference["reference_op"] - reference["best_op"]
    ram_range = reference["reference_ram_kib"] - reference["best_ram_kib"]
    normalized_op = (outage_probabilities - reference["best_op"]) / op_range
    normalized_ram = (ram_kib - reference["best_ram_kib"]) / ram_range
    if (
        op_range <= 0.0
        or ram_range <= 0.0
        or (normalized_op < 0.0).any()
        or (normalized_op > 1.0).any()
        or (normalized_ram < 0.0).any()
        or (normalized_ram > 1.0).any()
    ):
        raise ValueError(
            "Hypervolume objectives must lie inside a positive reference rectangle."
        )

    pareto_mask = compute_pareto_membership(normalized_op, normalized_ram)
    pareto_points = sorted(
        zip(normalized_op[pareto_mask], normalized_ram[pareto_mask]),
        key=lambda point: point[0],
    )
    hypervolume = 0.0
    current_ram = 1.0
    for outage_probability, ram in pareto_points:
        if ram < current_ram:
            hypervolume += (1.0 - outage_probability) * (current_ram - ram)
            current_ram = ram
    return float(hypervolume)


def build_reference(search_data: pd.DataFrame) -> dict[str, float]:
    """Build one fixed cross-study hypervolume reference rectangle.

    Args:
        search_data (pd.DataFrame): Validated completed-trial table containing
            strict-INT8 validation model-only OP and trained peak RAM in KiB.

    Returns:
        dict[str, float]: Global best values and reference values placed 5%
        beyond the observed completed-trial range for each objective.

    Raises:
        KeyError: If a required objective column is missing.
        ValueError: If objectives are empty, non-finite, or have no positive
            range.

    Examples:
        >>> data = pd.read_csv(SEARCH_DATA_PATH)
        >>> reference = build_reference(data)
        >>> reference["reference_op"] > reference["best_op"]
        True
    """
    required_columns = {"int8_validation_op", "trained_peak_ram_kib"}
    missing_columns = sorted(required_columns - set(search_data.columns))
    if missing_columns:
        raise KeyError(f"Search data is missing columns: {missing_columns}.")
    if search_data.empty:
        raise ValueError("No completed trials are available for the reference.")

    outage_probabilities = search_data["int8_validation_op"].to_numpy(dtype=float)
    ram_kib = search_data["trained_peak_ram_kib"].to_numpy(dtype=float)
    if not np.isfinite(outage_probabilities).all() or not np.isfinite(ram_kib).all():
        raise ValueError("Hypervolume objectives must be finite.")

    best_op = float(outage_probabilities.min())
    worst_op = float(outage_probabilities.max())
    best_ram = float(ram_kib.min())
    worst_ram = float(ram_kib.max())
    if worst_op <= best_op or worst_ram <= best_ram:
        raise ValueError("Hypervolume objectives must have positive observed ranges.")
    return {
        "best_op": best_op,
        "reference_op": worst_op + REFERENCE_MARGIN * (worst_op - best_op),
        "best_ram_kib": best_ram,
        "reference_ram_kib": worst_ram
        + REFERENCE_MARGIN * (worst_ram - best_ram),
    }


def build_convergence_table(
    search_data: pd.DataFrame,
    reference: dict[str, float],
) -> pd.DataFrame:
    """Build cumulative convergence data at all 100 scheduled checkpoints.

    Incomplete trials retain their scheduled positions but never enter the
    cumulative feasible set.

    Args:
        search_data (pd.DataFrame): Validated completed-trial rows for all five
            MobileNetV3 observation-budget studies.
        reference (dict[str, float]): Fixed cross-study hypervolume reference
            returned by :func:`build_reference`.

    Returns:
        pd.DataFrame: Five hundred rows containing completed-trial counts,
        best strict-INT8 validation model-only OP, Pareto-front sizes,
        normalized hypervolume, and percentage of final hypervolume.

    Raises:
        KeyError: If a required search-data column is missing.
        ValueError: If a budget has no completed trial by the final checkpoint
            or has non-positive final hypervolume.

    Examples:
        >>> data = pd.read_csv(SEARCH_DATA_PATH)
        >>> len(build_convergence_table(data, build_reference(data)))
        500
    """
    required_columns = {
        "observed_ports",
        "trial_id",
        "int8_validation_op",
        "trained_peak_ram_kib",
    }
    missing_columns = sorted(required_columns - set(search_data.columns))
    if missing_columns:
        raise KeyError(f"Search data is missing columns: {missing_columns}.")

    rows = []
    for observed_ports in OBSERVED_PORT_BUDGETS:
        budget_data = search_data[
            search_data["observed_ports"] == observed_ports
        ].sort_values("trial_id")
        for trial_number in range(1, SCHEDULED_TRIALS_PER_STUDY + 1):
            completed = budget_data[budget_data["trial_id"] < trial_number]
            if completed.empty:
                rows.append(
                    {
                        "observed_ports": observed_ports,
                        "trial_number": trial_number,
                        "completed_trial_count": 0,
                        "best_int8_validation_op": np.nan,
                        "pareto_front_size": 0,
                        "normalized_hypervolume": np.nan,
                    }
                )
                continue

            outage_probabilities = completed["int8_validation_op"].to_numpy(
                dtype=float
            )
            ram_kib = completed["trained_peak_ram_kib"].to_numpy(dtype=float)
            rows.append(
                {
                    "observed_ports": observed_ports,
                    "trial_number": trial_number,
                    "completed_trial_count": len(completed),
                    "best_int8_validation_op": float(outage_probabilities.min()),
                    "pareto_front_size": int(
                        compute_pareto_membership(
                            outage_probabilities,
                            ram_kib,
                        ).sum()
                    ),
                    "normalized_hypervolume": calculate_normalized_hypervolume(
                        outage_probabilities,
                        ram_kib,
                        reference,
                    ),
                }
            )

    convergence = pd.DataFrame(rows)
    final_hypervolume = convergence[
        convergence["trial_number"] == SCHEDULED_TRIALS_PER_STUDY
    ].set_index("observed_ports")["normalized_hypervolume"]
    if final_hypervolume.isna().any() or (final_hypervolume <= 0.0).any():
        raise ValueError("Every study must have a positive final hypervolume.")
    convergence["hypervolume_percent_of_final"] = convergence.apply(
        lambda row: 100.0
        * row["normalized_hypervolume"]
        / final_hypervolume.loc[row["observed_ports"]],
        axis=1,
    )
    return convergence


def build_summary_table(
    search_data: pd.DataFrame,
    study_summary: pd.DataFrame,
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize convergence checkpoints and selected-trial timing.

    Args:
        search_data (pd.DataFrame): Validated completed-trial rows containing
            one selected candidate for each observation budget.
        study_summary (pd.DataFrame): Validated study execution summary used
            to report completed and incomplete trial counts.
        convergence (pd.DataFrame): Cumulative convergence table returned by
            :func:`build_convergence_table`.

    Returns:
        pd.DataFrame: One row per budget with execution counts, hypervolume
        checkpoints, best-OP checkpoints, selected-trial timing, and final
        Pareto-front size.

    Raises:
        ValueError: If a study lacks exactly one selected trial, a required
            checkpoint, or an OP-minimal selected candidate.

    Examples:
        >>> search = pd.read_csv(SEARCH_DATA_PATH)
        >>> studies = pd.read_csv(STUDY_SUMMARY_PATH)
        >>> convergence = build_convergence_table(search, build_reference(search))
        >>> len(build_summary_table(search, studies, convergence))
        5
    """
    rows = []
    for observed_ports in OBSERVED_PORT_BUDGETS:
        budget_data = search_data[search_data["observed_ports"] == observed_ports]
        selected = budget_data[budget_data["is_selected"]]
        if len(selected) != 1:
            raise ValueError(
                f"Budget {observed_ports} must contain exactly one selected trial."
            )
        selected_trial_number = int(selected.iloc[0]["trial_number"])
        budget_convergence = convergence[
            convergence["observed_ports"] == observed_ports
        ].set_index("trial_number")
        checkpoints = {50, 75, SCHEDULED_TRIALS_PER_STUDY}
        if not checkpoints.issubset(budget_convergence.index):
            raise ValueError(
                f"Budget {observed_ports} is missing a convergence checkpoint."
            )
        selected_op = float(selected.iloc[0]["int8_validation_op"])
        final_op = float(
            budget_convergence.loc[
                SCHEDULED_TRIALS_PER_STUDY,
                "best_int8_validation_op",
            ]
        )
        if selected_op != final_op:
            raise ValueError(
                f"Budget {observed_ports} selected trial is not OP-minimal."
            )

        execution = study_summary[
            study_summary["observed_ports"] == observed_ports
        ].iloc[0]
        rows.append(
            {
                "observed_ports": observed_ports,
                "completed_trials": int(execution["completed_trials"]),
                "incomplete_trials": int(execution["incomplete_trials"]),
                "selected_trial_number": selected_trial_number,
                "scheduled_trials_after_selected": (
                    SCHEDULED_TRIALS_PER_STUDY - selected_trial_number
                ),
                "hypervolume_percent_at_trial_50": float(
                    budget_convergence.loc[50, "hypervolume_percent_of_final"]
                ),
                "hypervolume_percent_at_trial_75": float(
                    budget_convergence.loc[75, "hypervolume_percent_of_final"]
                ),
                "best_int8_validation_op_at_trial_50": float(
                    budget_convergence.loc[50, "best_int8_validation_op"]
                ),
                "best_int8_validation_op_at_trial_75": float(
                    budget_convergence.loc[75, "best_int8_validation_op"]
                ),
                "final_best_int8_validation_op": final_op,
                "final_pareto_front_size": int(
                    budget_convergence.loc[
                        SCHEDULED_TRIALS_PER_STUDY,
                        "pareto_front_size",
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_plot_table(
    convergence: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build wide best-so-far OP data for the standalone LaTeX figure.

    Args:
        convergence (pd.DataFrame): Cumulative all-regime convergence data from
            :func:`build_convergence_table`.
        summary (pd.DataFrame): Per-budget convergence summary from
            :func:`build_summary_table`, including each selected trial number.

    Returns:
        pd.DataFrame: One hundred scheduled checkpoints with best-so-far OP and
        a sparse selected-trial marker column for each observation budget.

    Raises:
        ValueError: If a budget does not contain exactly 100 checkpoints or its
            selected-trial OP differs from the final best OP.

    Examples:
        >>> search = pd.read_csv(SEARCH_DATA_PATH)
        >>> studies = pd.read_csv(STUDY_SUMMARY_PATH)
        >>> convergence = build_convergence_table(search, build_reference(search))
        >>> summary = build_summary_table(search, studies, convergence)
        >>> len(build_plot_table(convergence, summary))
        100
    """
    plot_data = pd.DataFrame(
        {"trial_number": range(1, SCHEDULED_TRIALS_PER_STUDY + 1)}
    )
    for observed_ports in OBSERVED_PORT_BUDGETS:
        budget_data = convergence[
            convergence["observed_ports"] == observed_ports
        ].sort_values("trial_number")
        expected_checkpoints = list(range(1, SCHEDULED_TRIALS_PER_STUDY + 1))
        if budget_data["trial_number"].astype(int).tolist() != expected_checkpoints:
            raise ValueError(
                f"Budget {observed_ports} must contain 100 plotting checkpoints."
            )
        selected_trial = int(
            summary.loc[
                summary["observed_ports"] == observed_ports,
                "selected_trial_number",
            ].iloc[0]
        )
        op_column = f"budget_{observed_ports}_op"
        selected_column = f"budget_{observed_ports}_selected_op"
        plot_data[op_column] = budget_data[
            "best_int8_validation_op"
        ].to_numpy()
        plot_data[selected_column] = np.nan
        selected_op = float(
            budget_data.loc[
                budget_data["trial_number"] == selected_trial,
                "best_int8_validation_op",
            ].iloc[0]
        )
        final_op = float(budget_data.iloc[-1]["best_int8_validation_op"])
        if selected_op != final_op:
            raise ValueError(
                f"Budget {observed_ports} selected marker is not OP-minimal."
            )
        plot_data.loc[
            plot_data["trial_number"] == selected_trial,
            selected_column,
        ] = selected_op
    return plot_data


def write_convergence_analysis() -> dict[str, Path]:
    """Generate the MobileNetV3 all-regime convergence artifacts.

    Args:
        None.

    Returns:
        dict[str, Path]: Paths to the checkpoint-level CSV, LaTeX plot-data
        CSV, summary CSV, and metadata JSON.

    Raises:
        FileNotFoundError: If a required dissertation-analysis input is absent.
        KeyError: If a required input column is missing.
        ValueError: If execution counts, trial ordering, selected candidates,
            or objective values are invalid.
        OSError: If an output cannot be written.

    Examples:
        >>> sorted(write_convergence_analysis())
        ['convergence', 'metadata', 'plot_data', 'summary']
    """
    if not SEARCH_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing NAS search data: {SEARCH_DATA_PATH}.")
    if not STUDY_SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing NAS study summary: {STUDY_SUMMARY_PATH}.")

    search_data = pd.read_csv(SEARCH_DATA_PATH)
    study_summary = pd.read_csv(STUDY_SUMMARY_PATH)
    validate_analysis_inputs(search_data, study_summary)
    reference = build_reference(search_data)
    convergence = build_convergence_table(search_data, reference)
    summary = build_summary_table(search_data, study_summary, convergence)
    plot_data = build_plot_table(convergence, summary)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(
        CONVERGENCE_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    plot_data.to_csv(
        PLOT_DATA_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    metadata = {
        "scope": "five MobileNetV3 all-regime NAS studies",
        "scheduled_trials_per_study": SCHEDULED_TRIALS_PER_STUDY,
        "completed_trials": EXPECTED_COMPLETED_TRIALS,
        "incomplete_trials": EXPECTED_INCOMPLETE_TRIALS,
        "incomplete_trial_treatment": "excluded from cumulative feasible sets",
        "hypervolume": {
            "objectives": [
                "int8_validation_op",
                "trained_peak_ram_kib",
            ],
            "direction": "minimize_both",
            "normalization": "global best to one fixed cross-study reference",
            "reference_margin_fraction_of_global_range": REFERENCE_MARGIN,
            **reference,
        },
        "evidence_summary": {
            "minimum_hypervolume_percent_at_trial_50": float(
                summary["hypervolume_percent_at_trial_50"].min()
            ),
            "minimum_hypervolume_percent_at_trial_75": float(
                summary["hypervolume_percent_at_trial_75"].min()
            ),
            "median_selected_trial_number": float(
                summary["selected_trial_number"].median()
            ),
            "latest_selected_trial_number": int(
                summary["selected_trial_number"].max()
            ),
            "minimum_scheduled_trials_after_selected": int(
                summary["scheduled_trials_after_selected"].min()
            ),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "convergence": CONVERGENCE_PATH,
        "plot_data": PLOT_DATA_PATH,
        "summary": SUMMARY_PATH,
        "metadata": METADATA_PATH,
    }


if __name__ == "__main__":
    write_convergence_analysis()
