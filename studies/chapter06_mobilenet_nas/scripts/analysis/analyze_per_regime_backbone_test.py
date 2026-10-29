"""Aggregate five-seed per-regime MobileNet backbone test results."""

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

from analyze_all_regime_backbone_test import (
    BACKBONES,
    EXPECTED_REGIME_COUNT,
    OBSERVED_PORT_BUDGETS,
    REQUIRED_COLUMNS,
    SEEDS,
    build_plot_data,
    summarize_regimes,
    write_analysis_outputs,
)


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/analyze_per_regime_backbone_test.py'
REPOSITORY_ROOT = _COMPANION_ROOT
RESULTS_ROOT = (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime')
OUTPUT_DIR = RESULTS_ROOT / "test_analysis"
SEED_RESULTS_PATH = OUTPUT_DIR / "backbone_test_by_seed_and_regime.csv"
REGIME_SUMMARY_PATH = OUTPUT_DIR / "backbone_test_summary.csv"
PLOT_DATA_PATH = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/figures/sources/fig6.3.csv'
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


def build_test_summary_path(
    results_root: Path,
    seed: int,
    backbone_slug: str,
    regime: str,
    observed_ports: int,
) -> Path:
    """Build the exact held-out summary path for one specialized model.

    Args:
        results_root (Path): Per-regime sparse-input results directory that
            contains the five ``seed_*`` directories.
        seed (int): Training seed. Accepted values are zero through four.
        backbone_slug (str): Training-directory backbone identifier. Accepted
            values are ``mobv1``, ``mobv2``, and ``mobv3``.
        regime (str): Fading-regime identifier. The value must be one of the 14
            regimes in the approved experiment grid.
        observed_ports (int): Number of observed ports. Accepted budgets are
            the integers from three through seven.

    Returns:
        Path: Expected top-level test CSV for the requested specialized model.

    Raises:
        ValueError: If any grid coordinate is outside the approved experiment.

    Examples:
        >>> path = build_test_summary_path(
        ...     RESULTS_ROOT, 0, "mobv1", "kappa0_mu1_m0", 3
        ... )
        >>> path.name
        'op_results_by_regime.csv'
    """
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed: {seed}.")
    if backbone_slug not in BACKBONES:
        raise ValueError(f"Unsupported backbone identifier: {backbone_slug}.")
    if regime not in REGIMES:
        raise ValueError(f"Unsupported regime: {regime}.")
    if observed_ports not in OBSERVED_PORT_BUDGETS:
        raise ValueError(f"Unsupported observation budget: {observed_ports}.")

    training_name = (
        f"train_{backbone_slug}_alpha02_60_20_20_regime_"
        f"{regime}_{observed_ports}_ports"
    )
    return (
        results_root
        / f"seed_{seed}"
        / "trainings"
        / training_name
        / "op_eval_t1.1885399"
        / "test"
        / "op_results_by_regime.csv"
    )


def load_seed_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Load and validate every per-regime backbone held-out summary.

    Args:
        results_root (Path): Authoritative per-regime sparse-input result
            directory. The default reads only the exact top-level held-out CSV
            for each model and never reads validation or nested duplicate files.

    Returns:
        pd.DataFrame: Exactly 1,050 rows, one for every seed, backbone, regime,
        and observation budget, with test outage probability and complementary
        non-outage rate.

    Raises:
        FileNotFoundError: If any expected held-out summary is absent.
        KeyError: If an input summary omits a required column.
        ValueError: If an input has the wrong row count, split, regime, budget,
            sample count, outage count, or outage probability.

    Examples:
        >>> results = load_seed_results()
        >>> len(results)
        1050
        >>> results["regime"].nunique()
        14
    """
    rows = []

    # Enumerating all coordinates makes missing experiments fail explicitly
    # instead of allowing a partial directory scan to produce an aggregate.
    for seed in SEEDS:
        for backbone_slug, backbone_name in BACKBONES.items():
            for regime in REGIMES:
                for observed_ports in OBSERVED_PORT_BUDGETS:
                    summary_path = build_test_summary_path(
                        results_root,
                        seed,
                        backbone_slug,
                        regime,
                        observed_ports,
                    )
                    if not summary_path.is_file():
                        raise FileNotFoundError(f"Missing test summary: {summary_path}")

                    summary = pd.read_csv(summary_path)
                    missing_columns = sorted(REQUIRED_COLUMNS - set(summary.columns))
                    if missing_columns:
                        raise KeyError(
                            f"Test summary {summary_path} is missing columns: "
                            f"{missing_columns}."
                        )
                    if len(summary) != 1:
                        raise ValueError(
                            f"Test summary {summary_path} must contain one regime row."
                        )

                    record = summary.iloc[0]
                    if str(record["split"]) != "test":
                        raise ValueError(f"Non-test row found in {summary_path}.")
                    if str(record["regime"]) != regime:
                        raise ValueError(f"Regime mismatch in {summary_path}.")
                    if int(record["n_observed_ports"]) != observed_ports:
                        raise ValueError(f"Observation-budget mismatch in {summary_path}.")

                    num_samples = int(record["num_samples"])
                    outage_samples = int(record["model_plus_observed_n_outage_samples"])
                    stored_op = float(record["model_plus_observed_op"])
                    if num_samples <= 0:
                        raise ValueError(f"Non-positive sample count in {summary_path}.")
                    if outage_samples < 0 or outage_samples > num_samples:
                        raise ValueError(f"Invalid outage count in {summary_path}.")
                    count_op = outage_samples / num_samples
                    if not np.isclose(stored_op, count_op, rtol=0.0, atol=1e-12):
                        raise ValueError(
                            f"Stored OP does not match the outage count in {summary_path}."
                        )

                    rows.append(
                        {
                            "seed": seed,
                            "backbone": backbone_name,
                            "observed_ports": observed_ports,
                            "regime": regime,
                            "num_samples": num_samples,
                            "model_plus_observed_n_outage_samples": outage_samples,
                            "test_op": count_op,
                            "test_non_outage_rate_pct": 100.0 * (1.0 - count_op),
                        }
                    )

    results = pd.DataFrame(rows)
    expected_rows = (
        len(SEEDS)
        * len(BACKBONES)
        * EXPECTED_REGIME_COUNT
        * len(OBSERVED_PORT_BUDGETS)
    )
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows, found {len(results)}.")
    return results.sort_values(
        ["backbone", "observed_ports", "regime", "seed"]
    ).reset_index(drop=True)


def analyze_backbone_test_results(
    results_root: Path = RESULTS_ROOT,
) -> tuple[Path, Path, Path]:
    """Run the deterministic per-regime backbone held-out analysis.

    Args:
        results_root (Path): Authoritative per-regime result directory. The
            default is the sparse ``32 x 32 x 3`` five-seed result family.

    Returns:
        tuple[Path, Path, Path]: Paths written for seed-level results,
        regime-level summaries, and dissertation plot data.

    Raises:
        FileNotFoundError: If an expected held-out summary is absent.
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
    return write_analysis_outputs(
        seed_results,
        regime_summary,
        plot_data,
        SEED_RESULTS_PATH,
        REGIME_SUMMARY_PATH,
        PLOT_DATA_PATH,
    )
