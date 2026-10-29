"""Regression tests for Chapter 8 dissertation figure styling."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = _COMPANION_ROOT
MODULE_PATH = (
    _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/analysis/analyze_retrain_all_regimes_dissertation.py'
)


def load_analysis_module():
    """Load the Chapter 8 dissertation analysis module.

    Args:
        None.

    Returns:
        module: Imported analysis module under test.

    Raises:
        RuntimeError: If Python cannot create an import specification or loader
            for the analysis module.

    Examples:
        >>> module = load_analysis_module()
        >>> hasattr(module, "_plot_representative_metric")
        True
    """
    specification = importlib.util.spec_from_file_location(
        "analyze_retrain_all_regimes_dissertation",
        MODULE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import analysis module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MODULE = load_analysis_module()


def representative_frame(regime: str) -> pd.DataFrame:
    """Create a complete synthetic representative-regime result frame.

    Args:
        regime (str): Internal regime key assigned to every generated row.

    Returns:
        pd.DataFrame: Ten rows covering five budgets and both model artifacts
        for UNI, with probabilities concentrated near one.

    Raises:
        ValueError: If ``regime`` is empty.

    Examples:
        >>> len(representative_frame("kappa5_mu2_m0"))
        10
    """
    if not regime:
        raise ValueError("regime must be nonempty.")
    rows = []
    for artifact_type in MODULE.ARTIFACTS:
        for offset, budget in enumerate(MODULE.BUDGETS):
            rows.append(
                {
                    "policy": "UNI",
                    "budget": budget,
                    "artifact_type": artifact_type,
                    "regime": regime,
                    "ideal_op_mean": 0.97546,
                    "observed_op_mean": 0.97680 - offset * 0.00005,
                    "model_only_op_mean": 0.97620 - offset * 0.00005,
                    "model_only_op_std": 0.00002,
                }
            )
    return pd.DataFrame(rows)


class PlotStyleTest(unittest.TestCase):
    """Verify the Chapter 8 figures retain the dissertation plot contract."""

    def tearDown(self) -> None:
        """Close figures retained for post-render assertions.

        Args:
            None.

        Returns:
            None: All Matplotlib figures are closed.

        Raises:
            RuntimeError: If Matplotlib cannot close an open figure.
        """
        plt.close("all")

    def test_very_hard_op_uses_log_scale_and_contains_all_curves(self) -> None:
        """Keep the current logarithmic axis readable around the supplied values.

        Args:
            None.

        Returns:
            None: Assertions complete without returning a value.

        Raises:
            AssertionError: If the OP axis is linear, excludes a curve, or has no visible labels.
            OSError: If the temporary PDF cannot be written.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(MODULE.plt, "close"):
                MODULE._plot_representative_metric(
                    representative_frame("kappa5_mu2_m0"),
                    "kappa5_mu2_m0",
                    "Very-hard regime\n$\\kappa=5$, $\\mu=2$, $m\\rightarrow0$",
                    "UNI",
                    "op",
                    Path(temporary_directory) / "figure.pdf",
                )

            axis = MODULE.plt.gcf().axes[0]
            axis.figure.canvas.draw()
            labels = {label.get_text() for label in axis.get_yticklabels()}

            self.assertEqual(axis.get_yscale(), "log")
            self.assertLess(axis.get_ylim()[0], 0.97546)
            self.assertGreater(axis.get_ylim()[1], 0.97680)
            self.assertTrue(any(label.strip() for label in labels))

    def test_representative_figures_exclude_redundant_non_outage_views(self) -> None:
        """Generate only the nine representative OP figures for Chapter 8.

        Args:
            None.

        Returns:
            None: Assertions complete without returning a value.

        Raises:
            AssertionError: If non-outage figures are requested or the OP
                sequence does not contain exactly Figs. 8.1--8.9.
        """
        with mock.patch.object(
            MODULE,
            "_plot_representative_metric",
            autospec=True,
        ) as plot_mock:
            paths = MODULE.plot_representative_figures(
                pd.DataFrame(),
                Path("dissertation/Figuras/pdf"),
            )

        self.assertEqual(len(paths), 9)
        self.assertEqual(
            [plot_call.args[4] for plot_call in plot_mock.call_args_list],
            ["op"] * 9,
        )
        self.assertTrue(all("non_outage" not in path.name for path in paths))
