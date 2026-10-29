"""Aggregate five-seed all-regime MobileNet backbone test results."""

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


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/analyze_all_regime_backbone_test.py'
RESULTS_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_all_regimes'
OUTPUT_DIR = RESULTS_ROOT / "evaluation_outputs" / "analysis" / "backbone_test"
SEED_RESULTS_PATH = OUTPUT_DIR / "backbone_test_by_seed_and_regime.csv"
REGIME_SUMMARY_PATH = OUTPUT_DIR / "backbone_test_summary.csv"
PLOT_DATA_PATH = OUTPUT_DIR / "backbone_test_plot_data.csv"
BUDGET_SUMMARY_PATH = OUTPUT_DIR / "backbone_test_by_observation_budget.csv"
SEEDS = (0, 1, 2, 3, 4)
BACKBONES = {
    "mobv1": "MobileNetV1",
    "mobv2": "MobileNetV2",
    "mobv3": "MobileNetV3",
}
BACKBONE_PREFIXES = {
    "MobileNetV1": "mobilenet_v1",
    "MobileNetV2": "mobilenet_v2",
    "MobileNetV3": "mobilenet_v3",
}
OBSERVED_PORT_BUDGETS = (3, 4, 5, 6, 7)
EXPECTED_REGIME_COUNT = 14
REQUIRED_COLUMNS = {
    "num_samples",
    "n_observed_ports",
    "ideal_op",
    "ideal_n_outage_samples",
    "observed_op",
    "observed_n_outage_samples",
    "model_only_op",
    "model_only_n_outage_samples",
    "model_plus_observed_op",
    "model_plus_observed_n_outage_samples",
    "split",
    "regime",
}


def build_test_summary_path(
    results_root: Path,
    seed: int,
    backbone_slug: str,
    observed_ports: int,
) -> Path:
    """Build the exact test-summary path for one trained backbone.

    Args:
        results_root (Path): Safe-selector all-regime results directory that
            contains the five ``seed_*`` directories.
        seed (int): Training seed. Stage 4 accepts the integers from zero
            through four.
        backbone_slug (str): Training-directory backbone identifier. Accepted
            values are ``mobv1``, ``mobv2``, and ``mobv3``.
        observed_ports (int): Number of observed ports. Accepted budgets are
            the integers from three through seven.

    Returns:
        Path: Expected aggregate test CSV for the requested training run.

    Raises:
        ValueError: If the seed, backbone identifier, or observation budget is
            outside the approved Stage 4 experiment grid.

    Examples:
        >>> path = build_test_summary_path(RESULTS_ROOT, 0, "mobv1", 3)
        >>> path.name
        'op_results_by_regime.csv'
    """
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed: {seed}.")
    if backbone_slug not in BACKBONES:
        raise ValueError(f"Unsupported backbone identifier: {backbone_slug}.")
    if observed_ports not in OBSERVED_PORT_BUDGETS:
        raise ValueError(f"Unsupported observation budget: {observed_ports}.")

    training_name = (
        f"train_{backbone_slug}_alpha02_60_20_20_all_regimes_"
        f"{observed_ports}_ports"
    )
    return (
        results_root
        / f"seed_{seed}"
        / training_name
        / "op_eval_t1.1885399"
        / "test"
        / "op_results_by_regime.csv"
    )


def load_seed_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Load and validate every approved all-regime backbone test summary.

    Args:
        results_root (Path): All-regime safe-selector result directory. The
            default is the authoritative completed result family. The function
            reads only exact ``test/op_results_by_regime.csv`` paths and never
            searches validation directories.

    Returns:
        pd.DataFrame: Exactly 1,050 rows, one for each seed, backbone,
        observation budget, and fading regime. Each row contains test outage
        probability and the complementary non-outage rate.

    Raises:
        FileNotFoundError: If any of the 75 expected test summaries is absent.
        KeyError: If an input summary omits a required column.
        ValueError: If an input has the wrong split, budget, regime coverage,
            sample counts, outage counts, outage probabilities, reference
            split, or disagreement between model-only and embedded-policy OP.

    Examples:
        >>> results = load_seed_results()
        >>> len(results)
        1050
        >>> results["seed"].nunique()
        5
    """
    rows = []
    expected_regimes = None

    # Enumerating the complete experimental grid prevents a partial aggregate
    # from looking valid merely because every file discovered by a glob loaded.
    for seed in SEEDS:
        for backbone_slug, backbone_name in BACKBONES.items():
            for observed_ports in OBSERVED_PORT_BUDGETS:
                summary_path = build_test_summary_path(
                    results_root,
                    seed,
                    backbone_slug,
                    observed_ports,
                )
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Missing test summary: {summary_path}")

                summary = pd.read_csv(summary_path)
                missing_columns = sorted(REQUIRED_COLUMNS - set(summary.columns))
                if missing_columns:
                    raise KeyError(
                        f"Test summary {summary_path} is missing columns: {missing_columns}."
                    )
                if len(summary) != EXPECTED_REGIME_COUNT:
                    raise ValueError(
                        f"Test summary {summary_path} must contain "
                        f"{EXPECTED_REGIME_COUNT} regime rows."
                    )
                if set(summary["split"].astype(str)) != {"test"}:
                    raise ValueError(f"Non-test rows found in {summary_path}.")
                if set(summary["n_observed_ports"].astype(int)) != {observed_ports}:
                    raise ValueError(f"Observation-budget mismatch in {summary_path}.")

                regimes = tuple(sorted(summary["regime"].astype(str)))
                if len(set(regimes)) != EXPECTED_REGIME_COUNT:
                    raise ValueError(f"Duplicate regime rows found in {summary_path}.")
                if expected_regimes is None:
                    expected_regimes = regimes
                elif regimes != expected_regimes:
                    raise ValueError(f"Regime coverage mismatch in {summary_path}.")

                # Counts are the numerical source of truth. The explicit
                # comparisons also prove that the saved safety policy and the
                # model-only decision agree throughout the completed campaign.
                for record in summary.to_dict(orient="records"):
                    num_samples = int(record["num_samples"])
                    if num_samples <= 0:
                        raise ValueError(f"Non-positive sample count in {summary_path}.")
                    if num_samples != 200000:
                        raise ValueError(f"Unexpected sample count in {summary_path}.")

                    checked_metrics = {}
                    for metric in (
                        "ideal",
                        "observed",
                        "model_only",
                        "model_plus_observed",
                    ):
                        outage_samples = int(record[f"{metric}_n_outage_samples"])
                        stored_op = float(record[f"{metric}_op"])
                        if outage_samples < 0 or outage_samples > num_samples:
                            raise ValueError(f"Invalid {metric} count in {summary_path}.")
                        count_op = outage_samples / num_samples
                        if not np.isclose(stored_op, count_op, rtol=0.0, atol=1e-12):
                            raise ValueError(
                                f"Stored {metric} OP does not match its count in "
                                f"{summary_path}."
                            )
                        checked_metrics[metric] = (outage_samples, count_op)

                    if checked_metrics["model_only"] != checked_metrics["model_plus_observed"]:
                        raise ValueError(
                            "Model-only and embedded-policy results differ in "
                            f"{summary_path}."
                        )

                    rows.append(
                        {
                            "seed": seed,
                            "backbone": backbone_name,
                            "observed_ports": observed_ports,
                            "regime": str(record["regime"]),
                            "num_samples": num_samples,
                            "ideal_n_outage_samples": checked_metrics["ideal"][0],
                            "ideal_op": checked_metrics["ideal"][1],
                            "observed_n_outage_samples": checked_metrics["observed"][0],
                            "observed_op": checked_metrics["observed"][1],
                            "model_only_n_outage_samples": checked_metrics["model_only"][0],
                            "model_only_op": checked_metrics["model_only"][1],
                            "test_non_outage_rate_pct": 100.0
                            * (1.0 - checked_metrics["model_only"][1]),
                        }
                    )

    results = pd.DataFrame(rows)
    expected_rows = (
        len(SEEDS)
        * len(BACKBONES)
        * len(OBSERVED_PORT_BUDGETS)
        * EXPECTED_REGIME_COUNT
    )
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows, found {len(results)}.")

    # Every training seed and backbone evaluated the same fixed held-out split.
    # Exact reference counts therefore provide a deterministic split-identity
    # check without relying on directory discovery or filename conventions.
    reference_variation = (
        results.groupby(["observed_ports", "regime"])[
            ["num_samples", "ideal_n_outage_samples", "observed_n_outage_samples"]
        ]
        .nunique()
        .to_numpy()
    )
    if not np.all(reference_variation == 1):
        raise ValueError("Ideal or observed-only counts vary across the held-out grid.")
    return results.sort_values(
        ["backbone", "observed_ports", "regime", "seed"]
    ).reset_index(drop=True)


def summarize_regimes(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize test OP and non-outage rate across the five seeds.

    Args:
        seed_results (pd.DataFrame): Validated seed-level output from
            :func:`load_seed_results`, containing one row per regime for each
            seed, backbone, and observation budget.

    Returns:
        pd.DataFrame: Exactly 210 regime-level rows with five-seed arithmetic
        means and sample standard deviations using ``ddof=1``. OP is reported
        as a probability, while non-outage rate is reported as a percentage.

    Raises:
        KeyError: If a required seed-level column is absent.
        ValueError: If an aggregate does not contain exactly five seeds.

    Examples:
        >>> summary = summarize_regimes(load_seed_results())
        >>> len(summary)
        210
        >>> set(summary["seed_count"])
        {5}
    """
    required_columns = {
        "seed",
        "backbone",
        "observed_ports",
        "regime",
        "model_only_op",
        "test_non_outage_rate_pct",
    }
    missing_columns = sorted(required_columns - set(seed_results.columns))
    if missing_columns:
        raise KeyError(f"Seed results are missing columns: {missing_columns}.")

    group_columns = ["backbone", "observed_ports", "regime"]
    summary = (
        seed_results.groupby(group_columns, as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_test_op=("model_only_op", "mean"),
            sd_test_op=("model_only_op", "std"),
            ideal_op=("ideal_op", "first"),
            observed_op=("observed_op", "first"),
            mean_test_non_outage_rate_pct=("test_non_outage_rate_pct", "mean"),
            sd_test_non_outage_rate_pct=("test_non_outage_rate_pct", "std"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    if set(summary["seed_count"].astype(int)) != {len(SEEDS)}:
        raise ValueError("Every regime-level aggregate must contain all five seeds.")
    return summary


def build_plot_data(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Build regime-specific test-OP plot data with seed uncertainty.

    Args:
        seed_results (pd.DataFrame): Validated seed-level output from
            :func:`load_seed_results`.

    Returns:
        pd.DataFrame: Seventy rows, one per regime and observation budget,
        containing separate OP and non-outage five-seed means and sample
        standard deviations for the three backbones.

    Raises:
        KeyError: If a required seed-level column is absent.
        ValueError: If a regime aggregate omits a seed or backbone, or if
            OP/non-outage complementarity is violated.

    Examples:
        >>> plot_data = build_plot_data(load_seed_results())
        >>> len(plot_data)
        70
        >>> set(plot_data["seed_count"])
        {5}
    """
    required_columns = {
        "seed",
        "backbone",
        "observed_ports",
        "regime",
        "model_only_op",
        "ideal_op",
        "observed_op",
        "test_non_outage_rate_pct",
    }
    missing_columns = sorted(required_columns - set(seed_results.columns))
    if missing_columns:
        raise KeyError(f"Seed results are missing columns: {missing_columns}.")

    regime_summary = summarize_regimes(seed_results)
    if set(regime_summary["seed_count"].astype(int)) != {len(SEEDS)}:
        raise ValueError("Every plotted regime must contain all five seeds.")
    if not np.allclose(
        regime_summary["mean_test_non_outage_rate_pct"],
        100.0 * (1.0 - regime_summary["mean_test_op"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Regime OP and non-outage means are not complementary.")
    if not np.allclose(
        regime_summary["sd_test_non_outage_rate_pct"],
        100.0 * regime_summary["sd_test_op"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Regime OP and non-outage deviations are inconsistent.")

    regimes = sorted(regime_summary["regime"].astype(str).unique())
    if len(regimes) != EXPECTED_REGIME_COUNT:
        raise ValueError("Plot data must contain all 14 fading regimes.")
    plot_rows = []
    for regime_index, regime in enumerate(regimes, start=1):
        for observed_ports in OBSERVED_PORT_BUDGETS:
            budget_summary = regime_summary[
                (regime_summary["regime"] == regime)
                & (regime_summary["observed_ports"] == observed_ports)
            ].set_index("backbone")
            if set(budget_summary.index) != set(BACKBONE_PREFIXES):
                raise ValueError(
                    f"Regime {regime}, budget {observed_ports} omits a backbone."
                )
            row = {
                "regime_index": regime_index,
                "regime": regime,
                "observed_ports": observed_ports,
                "seed_count": len(SEEDS),
                "ideal_op": float(budget_summary["ideal_op"].iloc[0]),
                "observed_only_op": float(budget_summary["observed_op"].iloc[0]),
            }
            if budget_summary["ideal_op"].nunique() != 1:
                raise ValueError(f"Ideal OP differs by backbone for {regime}.")
            if budget_summary["observed_op"].nunique() != 1:
                raise ValueError(f"Observed-only OP differs by backbone for {regime}.")
            for backbone_name, prefix in BACKBONE_PREFIXES.items():
                values = budget_summary.loc[backbone_name]
                row[f"{prefix}_mean_test_op"] = float(values["mean_test_op"])
                row[f"{prefix}_sd_test_op"] = float(values["sd_test_op"])
                row[f"{prefix}_mean_test_non_outage_rate_pct"] = float(
                    values["mean_test_non_outage_rate_pct"]
                )
                row[f"{prefix}_sd_test_non_outage_rate_pct"] = float(
                    values["sd_test_non_outage_rate_pct"]
                )
            plot_rows.append(row)
    return pd.DataFrame(plot_rows)


def summarize_observation_budgets(regime_summary: pd.DataFrame) -> pd.DataFrame:
    """Average held-out outage probability across regimes for each budget.

    Args:
        regime_summary (pd.DataFrame): Validated regime-level table produced by
            :func:`summarize_regimes`. It must contain all three backbones, all
            five observation budgets, all 14 regimes, and five seeds per
            backbone--budget--regime combination.

    Returns:
        pd.DataFrame: Five rows, one for each observation budget, containing
        the regime count, seed count, ideal and observed-only OP, and the mean
        held-out OP for MobileNetV1, MobileNetV2, and MobileNetV3.

    Raises:
        KeyError: If a required regime-summary column is absent.
        ValueError: If a budget omits a regime, seed, or backbone, or if the
            ideal or observed-only references differ across backbones.

    Examples:
        >>> summary = summarize_observation_budgets(
        ...     summarize_regimes(load_seed_results())
        ... )
        >>> list(summary["observed_ports"])
        [3, 4, 5, 6, 7]
    """
    required_columns = {
        "backbone",
        "observed_ports",
        "regime",
        "seed_count",
        "mean_test_op",
        "ideal_op",
        "observed_op",
    }
    missing_columns = sorted(required_columns - set(regime_summary.columns))
    if missing_columns:
        raise KeyError(f"Regime summary is missing columns: {missing_columns}.")

    output_rows = []
    for observed_ports in OBSERVED_PORT_BUDGETS:
        budget_summary = regime_summary[
            regime_summary["observed_ports"] == observed_ports
        ]
        if budget_summary["regime"].nunique() != EXPECTED_REGIME_COUNT:
            raise ValueError(
                f"Budget {observed_ports} must contain all fading regimes."
            )
        if set(budget_summary["seed_count"].astype(int)) != {len(SEEDS)}:
            raise ValueError(
                f"Budget {observed_ports} must contain all training seeds."
            )
        if set(budget_summary["backbone"]) != set(BACKBONE_PREFIXES):
            raise ValueError(
                f"Budget {observed_ports} must contain all three backbones."
            )

        reference_summary = budget_summary.groupby("regime").agg(
            ideal_count=("ideal_op", "nunique"),
            observed_count=("observed_op", "nunique"),
            ideal_op=("ideal_op", "first"),
            observed_op=("observed_op", "first"),
        )
        if not np.all(reference_summary[["ideal_count", "observed_count"]] == 1):
            raise ValueError(
                f"Reference OP differs by backbone for budget {observed_ports}."
            )

        output_row = {
            "observed_ports": observed_ports,
            "regime_count": EXPECTED_REGIME_COUNT,
            "seed_count": len(SEEDS),
            "ideal_op": float(reference_summary["ideal_op"].mean()),
            "observed_only_op": float(reference_summary["observed_op"].mean()),
        }
        for backbone_name, prefix in BACKBONE_PREFIXES.items():
            backbone_summary = budget_summary[
                budget_summary["backbone"] == backbone_name
            ]
            if len(backbone_summary) != EXPECTED_REGIME_COUNT:
                raise ValueError(
                    f"Budget {observed_ports} has incomplete {backbone_name} results."
                )
            output_row[f"{prefix}_mean_test_op"] = float(
                backbone_summary["mean_test_op"].mean()
            )
        output_rows.append(output_row)
    return pd.DataFrame(output_rows)


def write_analysis_outputs(
    seed_results: pd.DataFrame,
    regime_summary: pd.DataFrame,
    plot_data: pd.DataFrame,
    seed_results_path: Path = SEED_RESULTS_PATH,
    regime_summary_path: Path = REGIME_SUMMARY_PATH,
    plot_data_path: Path = PLOT_DATA_PATH,
) -> tuple[Path, Path, Path]:
    """Write deterministic Stage 4 analysis and plot-data CSVs.

    Args:
        seed_results (pd.DataFrame): Validated 1,050-row seed and regime table.
        regime_summary (pd.DataFrame): Five-seed regime-level summary produced
            by :func:`summarize_regimes`.
        plot_data (pd.DataFrame): Seventy-row regime-specific plot table produced by
            :func:`build_plot_data`.
        seed_results_path (Path): Destination for seed-level test data. The
            default is the centralized safe-selector analysis directory.
        regime_summary_path (Path): Destination for regime-level five-seed
            statistics. The default is
            ``backbone_test/backbone_test_summary.csv``.
        plot_data_path (Path): Destination for the PGFPlots input. The default
            is ``backbone_test/backbone_test_plot_data.csv``.

    Returns:
        tuple[Path, Path, Path]: Seed-level, regime-summary, and plot-data paths
        in that order.

    Raises:
        OSError: If an output directory or CSV cannot be created.

    Examples:
        >>> seed_results = load_seed_results()
        >>> outputs = write_analysis_outputs(
        ...     seed_results,
        ...     summarize_regimes(seed_results),
        ...     build_plot_data(seed_results),
        ... )
        >>> len(outputs)
        3
    """
    for output_path in (seed_results_path, regime_summary_path, plot_data_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_results.to_csv(seed_results_path, index=False, float_format="%.12g")
    regime_summary.to_csv(regime_summary_path, index=False, float_format="%.12g")
    plot_data.to_csv(plot_data_path, index=False, float_format="%.12g")
    return seed_results_path, regime_summary_path, plot_data_path


def analyze_backbone_test_results(
    results_root: Path = RESULTS_ROOT,
) -> tuple[Path, Path, Path, Path]:
    """Run the deterministic safe-selector backbone test analysis.

    Args:
        results_root (Path): Authoritative all-regime safe-selector result
            directory containing all 75 trained runs.

    Returns:
        tuple[Path, Path, Path, Path]: Paths written for seed-level results,
        regime-level summaries, dissertation plot data, and observation-budget
        summaries, in that order.

    Raises:
        FileNotFoundError: If an expected test summary is absent.
        KeyError: If an input summary omits required data.
        ValueError: If any completeness or numerical validation fails.
        OSError: If an output cannot be written.

    Examples:
        >>> outputs = analyze_backbone_test_results()
        >>> all(path.is_file() for path in outputs)
        True
    """
    seed_results = load_seed_results(results_root)
    regime_summary = summarize_regimes(seed_results)
    plot_data = build_plot_data(seed_results)
    output_paths = write_analysis_outputs(seed_results, regime_summary, plot_data)
    budget_summary = summarize_observation_budgets(regime_summary)
    budget_summary.to_csv(BUDGET_SUMMARY_PATH, index=False, float_format="%.12g")
    return (*output_paths, BUDGET_SUMMARY_PATH)
