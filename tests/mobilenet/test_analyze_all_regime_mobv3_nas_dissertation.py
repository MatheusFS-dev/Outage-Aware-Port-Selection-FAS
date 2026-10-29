"""Tests for deterministic all-regime MobileNetV3 dissertation aggregation."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest


REPOSITORY_ROOT = _COMPANION_ROOT
MODULE_PATH = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/analyze_all_regime_nas_dissertation.py'
)


def load_analysis_module():
    """Load the dissertation aggregation module from its repository path.

    Args:
        None.

    Returns:
        module: Imported Python module under test.

    Raises:
        RuntimeError: If Python cannot create an import specification or loader
            for the target module.

    Examples:
        >>> module = load_analysis_module()
        >>> hasattr(module, "RESULTS_ROOT")
        True
    """
    specification = importlib.util.spec_from_file_location(
        "analyze_all_regime_nas_dissertation",
        MODULE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import analysis module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MODULE = load_analysis_module()


def copy_heldout_results(destination: Path) -> Path:
    """Copy only the five held-out CSV inputs into a temporary result tree.

    Args:
        destination (Path): Empty temporary directory that will receive the
            five study directories and their OP evaluation CSV files.

    Returns:
        Path: The populated temporary result root.

    Raises:
        FileNotFoundError: If an authoritative held-out CSV is absent.
        OSError: If a temporary directory or copied CSV cannot be created.

    Examples:
        >>> root = copy_heldout_results(Path("temporary_results"))
        >>> root.name
        'temporary_results'
    """
    for observed_ports in MODULE.BUDGETS:
        study_name = MODULE.STUDY_NAME_TEMPLATE.format(
            observed_ports=observed_ports
        )
        source = (
            MODULE.RESULTS_ROOT
            / study_name
            / MODULE.OP_EVALUATION_DIRECTORY
            / MODULE.OP_RESULTS_FILENAME
        )
        target = (
            destination
            / study_name
            / MODULE.OP_EVALUATION_DIRECTORY
            / MODULE.OP_RESULTS_FILENAME
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def test_load_completed_trials_excludes_incomplete_rows() -> None:
    """Exclude incomplete trials before every reported NAS calculation.

    Args:
        None.

    Returns:
        None: Assertions complete without returning a value.

    Raises:
        AssertionError: If incomplete trials enter the aggregated evidence or
            the authoritative completed counts change unexpectedly.
    """
    trials = MODULE.load_completed_trials()

    assert len(trials) == 489
    assert set(trials["state"]) == {"COMPLETE"}
    assert trials.groupby("observed_ports").size().to_dict() == {
        3: 97,
        4: 98,
        5: 98,
        6: 98,
        7: 98,
    }


def test_build_dissertation_outputs_selects_minimum_int8_op() -> None:
    """Select the completed feasible trial with minimum INT8 validation OP.

    Args:
        None.

    Returns:
        None: Assertions complete without returning a value.

    Raises:
        AssertionError: If selected trial identities or validation objectives
            differ from the authoritative campaign evidence.
    """
    outputs = MODULE.build_dissertation_outputs()

    selected = outputs["selected_models"]
    assert selected["trial_id"].tolist() == [64, 94, 46, 84, 38]
    assert selected["int8_validation_op"].tolist() == pytest.approx(
        [
            0.37215857142857145,
            0.3237157142857143,
            0.31422964285714283,
            0.3079975,
            0.3077992857142857,
        ],
        abs=1e-15,
    )


def test_load_heldout_results_requires_paired_selected_artifacts(
    tmp_path: Path,
) -> None:
    """Reject an OP table missing one selected FP32/INT8 regime pair.

    Args:
        tmp_path (Path): Pytest-managed temporary directory used to isolate the
            intentionally malformed held-out result collection.

    Returns:
        None: Assertions complete without returning a value.

    Raises:
        AssertionError: If the malformed evidence is accepted.
        OSError: If the temporary fixture cannot be copied or modified.
    """
    results_root = copy_heldout_results(tmp_path / "results")
    study_name = MODULE.STUDY_NAME_TEMPLATE.format(observed_ports=3)
    malformed_path = (
        results_root
        / study_name
        / MODULE.OP_EVALUATION_DIRECTORY
        / MODULE.OP_RESULTS_FILENAME
    )
    malformed = pd.read_csv(malformed_path)
    missing_pair = (
        (malformed["artifact_type"] == "int8_tflite")
        & (malformed["regime"] == "kappa0_mu1_m0")
    )
    malformed.loc[~missing_pair].to_csv(malformed_path, index=False)

    with pytest.raises(ValueError, match="paired FP32 and INT8"):
        MODULE.load_heldout_results(results_root)
