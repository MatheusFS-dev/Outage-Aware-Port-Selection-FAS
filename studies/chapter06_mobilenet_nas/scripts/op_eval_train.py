"""Run OP evaluation for one or more MobileNet Keras training runs."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/op_eval_train.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mobilenet.keras.sparse_input import SparsePortImagePredictAdapter
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    ExpandedPortImage,
    SafeDecisionScores,
    SparseImageObservedValues,
)
from mobilenet.keras.variants import get_variant_metadata
from utils.constants import BLUE, GREEN, RESET
from utils.data import KMS_REGIME_LABELS, split_kms_60_20_20_cache, split_kms_ratio_cache
from utils.op_eval import evaluate_op_per_regime_for_thresholds, save_op_outputs

# Set ROOT_SEARCH_DIR to a list of paths to automatically discover all
# MobileNet training runs within them. Each direct child directory is treated
# as a candidate run and validated explicitly. If None, the explicit
# TRAIN_RUN_DIR list is used instead.
ROOT_SEARCH_DIR: list[Path] | None = [
    (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime/seed_0'),
    (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime/seed_1'),
    (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime/seed_2'),
    (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime/seed_3'),
    (_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime/seed_4'),
]

# Set TRAIN_RUN_DIR to a list of MobileNet training runs. The script reads
# model_training/args/mobilenet_config.json from each directory and fails
# explicitly if any listed run metadata is missing or invalid. Used only if
# ROOT_SEARCH_DIR is None.
TRAIN_RUN_DIR = [
    # Path("./results/test_001/trainings/train_mobv1_alpha02_60_20_20_4_ports"),
    # Path("./results/test_001/trainings/train_mobv2_alpha02_60_20_20_4_ports"),
    # Path("./results/test_001/trainings/train_mobv3_alpha02_60_20_20_4_ports"),
]

TOTAL_PORTS = 100
DATA_SEED = 0
THRESHOLD = [1.1885398626327515,]
SNR_LINEAR = 1.0
BATCH_SIZE = 256

EVALUATE_TRAIN_SPLIT = False
EVALUATE_VAL_SPLIT = True
EVALUATE_TEST_SPLIT = False

# If True, only the regimes the model was trained on (found in its config)
# will be evaluated. If False, the model is evaluated on all regimes.
EVALUATE_ONLY_TRAINED_REGIMES = True

# If True, skip evaluation for a run if its op_eval directory already exists.
SKIP_IF_OP_EVAL_EXISTS = True

print(f"{BLUE}Starting MobileNet Keras OP evaluation...{RESET}")

_SPLIT_PARTS_CACHE: dict[
    tuple[str, int | None, int, float, float, float],
    dict[str, tuple[object, ...]],
] = {}
_SAFE_SELECTOR_CUSTOM_OBJECTS = {
    "ExpandedPortImage": ExpandedPortImage,
    "SafeDecisionScores": SafeDecisionScores,
    "SparseImageObservedValues": SparseImageObservedValues,
}


def _get_mobilenet_config_path(train_run_dir: Path) -> Path:
    """Return the strict MobileNet metadata path for one training run.

    Args:
        train_run_dir (Path): Top-level run directory produced by one of the
            MobileNet training entry scripts in ``src/mobilenet``.

    Returns:
        Path: Path to ``model_training/args/mobilenet_config.json`` inside the
            provided run directory.

    Raises:
        None.
    """
    return train_run_dir / "model_training" / "args" / "mobilenet_config.json"


def _is_non_empty_file(path: Path) -> bool:
    """Return whether a path is a regular file with content.

    Args:
        path (Path): Artifact path to inspect. Missing paths, directories, and
            zero-byte files are invalid OP-evaluation inputs.

    Returns:
        bool: ``True`` only when ``path`` is a file and its size is greater
            than zero bytes.

    Raises:
        None.
    """
    return path.is_file() and path.stat().st_size > 0


def _load_mobilenet_config(train_run_dir: Path) -> dict[str, Any]:
    """Load and validate strict MobileNet metadata for one training run.

    Args:
        train_run_dir (Path): Top-level MobileNet run directory. This function
            does not guess the variant from the directory name; it only trusts
            the explicit JSON metadata saved during training.

    Returns:
        dict[str, Any]: Parsed metadata payload containing the required keys
            ``variant_name``, ``backbone_family``, ``alpha``, ``input_shape``,
            ``num_observed_ports``, ``total_ports``, and
            ``train_regime_labels``. Legacy 60/20/20 runs include
            ``samples_per_regime``. Ratio-analysis runs include ``split_mode``
            and split ratios so OP evaluation can reconstruct the same split
            policy used during training.

    Raises:
        FileNotFoundError: If ``mobilenet_config.json`` is missing.
        ValueError: If the JSON is empty, malformed, missing required keys, or
            names an unsupported variant.

    Examples:
        >>> import tempfile
        >>> tmp_dir = Path(tempfile.mkdtemp())
        >>> _load_mobilenet_config(tmp_dir)
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
    """
    config_path = _get_mobilenet_config_path(train_run_dir)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected MobileNet metadata not found at {config_path}. "
            "Run one of the src/mobilenet/keras/split_60_20_20/train_* scripts first to create mobilenet_config.json."
        )
    if not _is_non_empty_file(config_path):
        raise ValueError(f"Expected non-empty MobileNet metadata at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as config_file:
        try:
            config_payload = json.load(config_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {config_path}: {error}") from error

    required_keys = {
        "variant_name",
        "backbone_family",
        "alpha",
        "input_shape",
        "num_observed_ports",
        "observed_indices",
        "total_ports",
        "train_regime_labels",
        "data_seed",
    }
    missing_keys = sorted(required_keys.difference(config_payload))
    if missing_keys:
        raise ValueError(f"MobileNet metadata at {config_path} is missing required keys: {missing_keys}.")

    variant_name = str(config_payload["variant_name"])
    variant_metadata = get_variant_metadata(variant_name)
    if config_payload["backbone_family"] != variant_metadata["backbone_family"]:
        raise ValueError(
            "MobileNet metadata backbone_family does not match the supported "
            f"variant registry for {variant_name!r}."
        )
    if float(config_payload["alpha"]) != float(variant_metadata["alpha"]):
        raise ValueError(
            f"MobileNet metadata alpha does not match the supported variant registry for {variant_name!r}."
        )

    input_shape = config_payload["input_shape"]
    if input_shape != [32, 32, 3]:
        raise ValueError(f"Unexpected MobileNet input_shape {input_shape!r}. Expected [32, 32, 3].")

    return config_payload


def _get_artifact_paths(train_run_dir: Path) -> tuple[Path, Path, Path]:
    """Return the saved model and scaler artifact paths for one run.

    Args:
        train_run_dir (Path): Top-level MobileNet run directory.

    Returns:
        tuple[Path, Path, Path]: ``(model_path, scaler_x_path, scaler_y_path)``.

    Raises:
        None.
    """
    model_path = train_run_dir / "model_training" / "model" / "model.keras"
    scaler_x_path = train_run_dir / "model_training" / "scaler" / "scaler_X.pkl"
    scaler_y_path = train_run_dir / "model_training" / "scaler" / "scaler_y.pkl"
    return model_path, scaler_x_path, scaler_y_path


def _validate_artifacts(model_path: Path, scaler_x_path: Path, scaler_y_path: Path) -> None:
    """Validate required model and scaler artifacts are usable for OP evaluation.

    Args:
        model_path (Path): Path to saved Keras model.
        scaler_x_path (Path): Path to input scaler pickle.
        scaler_y_path (Path): Path to target scaler pickle.

    Returns:
        None.

    Raises:
        FileNotFoundError: If any required artifact is missing.
        ValueError: If any required artifact is empty or is not a regular file.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model not found at {model_path}.")
    if not _is_non_empty_file(model_path):
        raise ValueError(f"Expected non-empty model artifact at {model_path}.")
    if not scaler_x_path.exists():
        raise FileNotFoundError(f"Expected scaler_X.pkl not found at {scaler_x_path}.")
    if not _is_non_empty_file(scaler_x_path):
        raise ValueError(f"Expected non-empty scaler_X.pkl artifact at {scaler_x_path}.")
    if not scaler_y_path.exists():
        raise FileNotFoundError(f"Expected scaler_y.pkl not found at {scaler_y_path}.")
    if not _is_non_empty_file(scaler_y_path):
        raise ValueError(f"Expected non-empty scaler_y.pkl artifact at {scaler_y_path}.")


def _load_model_and_scalers(
    model_path: Path,
    scaler_x_path: Path,
    scaler_y_path: Path,
) -> tuple[tf.keras.Model, object, object]:
    """Load the trained Keras model and fitted scalers for OP evaluation.

    Args:
        model_path (Path): Path to saved Keras model.
        scaler_x_path (Path): Path to input scaler pickle.
        scaler_y_path (Path): Path to target scaler pickle.

    Returns:
        tuple[tf.keras.Model, object, object]: ``(model, scaler_x, scaler_y)``.

    Raises:
        OSError: If artifact files cannot be read.
        pickle.UnpicklingError: If scaler pickles are invalid.
    """
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=_SAFE_SELECTOR_CUSTOM_OBJECTS,
        compile=False,
    )
    with open(scaler_x_path, "rb") as scaler_file:
        scaler_x = pickle.load(scaler_file)
    with open(scaler_y_path, "rb") as scaler_file:
        scaler_y = pickle.load(scaler_file)
    return model, scaler_x, scaler_y


def _get_enabled_split_names() -> tuple[str, ...]:
    """Return the enabled OP-evaluation split names in fixed order.

    Args:
        None.

    Returns:
        tuple[str, ...]: Enabled split names.

    Raises:
        ValueError: If all split toggles are disabled.
    """
    enabled_split_names = []
    if EVALUATE_TRAIN_SPLIT:
        enabled_split_names.append("train")
    if EVALUATE_VAL_SPLIT:
        enabled_split_names.append("val")
    if EVALUATE_TEST_SPLIT:
        enabled_split_names.append("test")

    if not enabled_split_names:
        raise ValueError("At least one of the OP split toggles must be enabled.")

    return tuple(enabled_split_names)


def _is_op_eval_complete(
    output_dir: Path,
    split_names: tuple[str, ...],
    regime_labels: list[str],
) -> bool:
    """Return whether all requested OP-evaluation outputs are complete.

    Args:
        output_dir (Path): Threshold-specific OP-evaluation output directory.
        split_names (tuple[str, ...]): Split names enabled for the current run.
        regime_labels (list[str]): Regimes expected in every enabled split.

    Returns:
        bool: ``True`` only when every aggregate, split, and per-regime output
            file expected for the current run exists and is non-empty.

    Raises:
        OSError: If an existing output file cannot be inspected.
    """
    required_paths = [
        output_dir / "op_eval_config.json",
        output_dir / "op_results_by_split_and_regime.csv",
    ]
    for split_name in split_names:
        split_output_dir = output_dir / split_name
        required_paths.append(split_output_dir / "op_results_by_regime.csv")
        for regime_label in regime_labels:
            regime_output_dir = split_output_dir / regime_label
            required_paths.extend(
                (
                    regime_output_dir / "op_results_by_regime.csv",
                    regime_output_dir / "model_predictions_by_sample.csv",
                )
            )

    return all(_is_non_empty_file(path) for path in required_paths)


def _load_split_parts(
    split_mode: str,
    samples_per_regime: int | None,
    data_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[object, ...]]:
    """Load per-regime train, validation, and test parts for OP evaluation.

    Args:
        split_mode (str): Split strategy read from training metadata. If
            ``"60_20_20"``, reconstruct the standard split. If ``"ratio"``,
            reconstruct the configurable ratio split. Missing legacy metadata
            is normalized to ``"60_20_20"`` before this function is called.
        samples_per_regime (int | None): Optional per-regime sample budget from
            the training metadata. Only standard 60/20/20 runs use this value.
        data_seed (int): Seed from the training metadata. This must match the
            seed used during training so OP evaluation reconstructs the same
            train, validation, and test memberships for seed-robustness runs.
        train_ratio (float): Training fraction used by ratio-analysis runs.
        val_ratio (float): Validation fraction used by ratio-analysis runs.
        test_ratio (float): Test fraction used by ratio-analysis runs.

    Returns:
        dict[str, tuple[object, ...]]: Mapping from split name to per-regime
            dataset parts.

    Raises:
        FileNotFoundError: If the KMS runtime cache is missing.
        ValueError: If the recorded split cannot be reconstructed.
    """
    cache_key = (
        split_mode,
        samples_per_regime,
        data_seed,
        float(train_ratio),
        float(val_ratio),
        float(test_ratio),
    )
    if cache_key in _SPLIT_PARTS_CACHE:
        print(f"{BLUE}Reusing cached {split_mode} OP split data for data_seed={data_seed}.{RESET}")
        return _SPLIT_PARTS_CACHE[cache_key]

    if split_mode == "ratio":
        print(
            f"{BLUE}Loading and preprocessing cache-backed ratio OP data "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})...{RESET}"
        )
        _, _, _, train_parts, val_parts, test_parts = split_kms_ratio_cache(
            common_path=None,
            seed=data_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            regime_labels=None,
            samples_per_regime=None,
            verbose=False,
        )
    elif split_mode == "60_20_20" and samples_per_regime is None:
        print(f"{BLUE}Loading and preprocessing full cache-backed 60/20/20 OP data...{RESET}")
        _, _, _, train_parts, val_parts, test_parts = split_kms_60_20_20_cache(
            common_path=None,
            seed=data_seed,
            regime_labels=None,
            samples_per_regime=samples_per_regime,
            verbose=False,
        )
    elif split_mode == "60_20_20":
        print(
            f"{BLUE}Loading and preprocessing cache-backed 60/20/20 OP data "
            f"with samples_per_regime={samples_per_regime}...{RESET}"
        )
        _, _, _, train_parts, val_parts, test_parts = split_kms_60_20_20_cache(
            common_path=None,
            seed=data_seed,
            regime_labels=None,
            samples_per_regime=samples_per_regime,
            verbose=False,
        )
    else:
        raise ValueError("split_mode must be '60_20_20' or 'ratio'.")

    split_parts_map = {
        "train": train_parts,
        "val": val_parts,
        "test": test_parts,
    }
    _SPLIT_PARTS_CACHE[cache_key] = split_parts_map
    return split_parts_map


def _run_op_eval_mobilenet_for_one_dir(
    train_run_dir: Path,
    thresholds: list[float] | tuple[float, ...],
) -> dict[float, dict[str, Path]]:
    """Run split-aware OP evaluation for one MobileNet training run.

    Args:
        train_run_dir (Path): Top-level MobileNet run directory.
            The function reads ``mobilenet_config.json`` from this directory and
            uses it as the only source of variant detection.
            If the metadata is missing or invalid, evaluation fails explicitly.
        thresholds (list[float] | tuple[float, ...]): Threshold values to
            evaluate. Each threshold keeps the existing ``op_eval_t...`` output
            directory. Multiple thresholds reuse one prediction pass per
            split/regime, so extra thresholds add metric work but not inference
            work.

    Returns:
        dict[float, dict[str, Path]]: Mapping from threshold to paths for the
            combined summary CSV and output directory.

    Raises:
        FileNotFoundError: If metadata or model artifacts are missing.
        ValueError: If metadata is invalid, no thresholds are provided, or the
            saved model is incompatible with metadata.

    Examples:
        >>> isinstance(TRAIN_RUN_DIR[0], Path)
        True
    """
    if not thresholds:
        raise ValueError("thresholds must contain at least one threshold.")

    config_payload = _load_mobilenet_config(train_run_dir=train_run_dir)
    model_path, scaler_x_path, scaler_y_path = _get_artifact_paths(train_run_dir=train_run_dir)
    _validate_artifacts(model_path=model_path, scaler_x_path=scaler_x_path, scaler_y_path=scaler_y_path)

    enabled_split_names = _get_enabled_split_names()
    train_regime_labels = config_payload.get("train_regime_labels", [])
    if EVALUATE_ONLY_TRAINED_REGIMES:
        evaluated_regimes = [regime for regime in KMS_REGIME_LABELS if regime in train_regime_labels]
    else:
        evaluated_regimes = list(KMS_REGIME_LABELS)

    outputs_by_threshold: dict[float, dict[str, Path]] = {}
    output_dirs_by_threshold: dict[float, Path] = {}
    missing_thresholds: list[float] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        # Stamp the threshold value into the folder name so that runs with
        # different thresholds never overwrite each other.
        threshold_tag = f"t{threshold_value:.8g}"
        output_dir = train_run_dir / f"op_eval_{threshold_tag}"
        output_dirs_by_threshold[threshold_value] = output_dir
        outputs_by_threshold[threshold_value] = {
            "combined_results_table": output_dir / "op_results_by_split_and_regime.csv",
            "output_dir": output_dir,
        }
        if SKIP_IF_OP_EVAL_EXISTS and _is_op_eval_complete(
            output_dir=output_dir,
            split_names=enabled_split_names,
            regime_labels=evaluated_regimes,
        ):
            print(f"{BLUE}Skipping complete OP evaluation at {output_dir}.{RESET}")
        else:
            missing_thresholds.append(threshold_value)

    if not missing_thresholds:
        return outputs_by_threshold

    variant_name = str(config_payload["variant_name"])
    variant_metadata = get_variant_metadata(variant_name)

    print(
        f"{BLUE}Detected MobileNet variant {variant_name} "
        f"({variant_metadata['backbone_family']}, alpha={variant_metadata['alpha']}).{RESET}"
    )

    num_observed_ports = int(config_payload["num_observed_ports"])
    observed_indices = [int(index) for index in config_payload["observed_indices"]]
    if len(observed_indices) != num_observed_ports:
        raise ValueError(
            "MobileNet metadata observed_indices length does not match "
            f"num_observed_ports: {len(observed_indices)} != {num_observed_ports}."
        )
    total_ports = int(config_payload["total_ports"])
    split_mode = str(config_payload.get("split_mode", "60_20_20"))
    samples_per_regime = config_payload.get("samples_per_regime")
    data_seed = int(config_payload["data_seed"])
    train_ratio = float(config_payload.get("train_ratio", 0.6))
    val_ratio = float(config_payload.get("val_ratio", 0.2))
    test_ratio = float(config_payload.get("test_ratio", 0.2))
    if total_ports != TOTAL_PORTS:
        raise ValueError(
            f"MobileNet metadata total_ports={total_ports} does not match the expected {TOTAL_PORTS}."
        )

    model, scaler_x, scaler_y = _load_model_and_scalers(
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path,
    )

    input_shape = model.input_shape
    if input_shape != (None, 32, 32, 3):
        raise ValueError(
            "Saved model input shape does not match sparse MobileNet metadata: "
            f"model expects {input_shape}, metadata expects (None, 32, 32, 3)."
        )

    scaler_features = int(getattr(scaler_x, "n_features_in_", -1))
    if scaler_features != num_observed_ports:
        raise ValueError(
            "Input scaler feature count does not match MobileNet metadata: "
            f"scaler_X expects {scaler_features}, metadata expects {num_observed_ports}."
        )
    sparse_model = SparsePortImagePredictAdapter(
        model=model,
        observed_indices=np.asarray(observed_indices, dtype=np.int64),
        total_ports=total_ports,
    )

    split_parts_map = _load_split_parts(
        split_mode=split_mode,
        samples_per_regime=samples_per_regime,
        data_seed=data_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    for threshold in missing_thresholds:
        output_dir = output_dirs_by_threshold[threshold]
        output_dir.mkdir(parents=True, exist_ok=True)
        op_eval_config = {
            "train_run_dir": str(train_run_dir),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "scaler_x_path": str(scaler_x_path),
            "scaler_y_path": str(scaler_y_path),
            "variant_name": variant_name,
            "backbone_family": str(config_payload["backbone_family"]),
            "alpha": float(config_payload["alpha"]),
            "input_shape": list(config_payload["input_shape"]),
            "total_ports": int(total_ports),
            "num_observed_ports": int(num_observed_ports),
            "split_mode": split_mode,
            "samples_per_regime": samples_per_regime,
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "unused_ratio": float(config_payload.get("unused_ratio", 0.0)),
            "data_seed": int(data_seed),
            "threshold": float(threshold),
            "snr_linear": float(SNR_LINEAR),
            "batch_size": int(BATCH_SIZE),
            "evaluate_train_split": bool(EVALUATE_TRAIN_SPLIT),
            "evaluate_val_split": bool(EVALUATE_VAL_SPLIT),
            "evaluate_test_split": bool(EVALUATE_TEST_SPLIT),
            "evaluate_only_trained_regimes": bool(EVALUATE_ONLY_TRAINED_REGIMES),
            "enabled_split_names": list(enabled_split_names),
            "regime_labels": evaluated_regimes,
        }
        (output_dir / "op_eval_config.json").write_text(
            json.dumps(op_eval_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"{BLUE}Enabled OP evaluation splits: {', '.join(enabled_split_names)}{RESET}")

    combined_results_frames_by_threshold: dict[float, list[pd.DataFrame]] = {
        threshold: [] for threshold in missing_thresholds
    }

    total_splits = len(enabled_split_names)

    # Evaluate each enabled split independently so the per-split CSVs stay
    # aligned with the existing example layout while still sharing one script.
    for split_idx, split_name in enumerate(enabled_split_names, 1):
        for threshold in missing_thresholds:
            split_output_dir = output_dirs_by_threshold[threshold] / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)

        if EVALUATE_ONLY_TRAINED_REGIMES:
            regimes_to_eval = [
                (regime, split_subset)
                for regime, split_subset in zip(KMS_REGIME_LABELS, split_parts_map[split_name])
                if regime in train_regime_labels
            ]
        else:
            regimes_to_eval = list(zip(KMS_REGIME_LABELS, split_parts_map[split_name]))

        total_regimes = len(regimes_to_eval)
        split_results_frames_by_threshold: dict[float, list[pd.DataFrame]] = {
            threshold: [] for threshold in missing_thresholds
        }

        for regime_idx, (regime, split_subset) in enumerate(regimes_to_eval, 1):
            print(
                f"{BLUE}[Split {split_idx}/{total_splits}] [Regime {regime_idx}/{total_regimes}] "
                f"Evaluating variant={variant_name}, split={split_name}, regime={regime}, "
                f"thresholds={missing_thresholds}{RESET}"
            )
            results_by_threshold, predictions_df = evaluate_op_per_regime_for_thresholds(
                model=sparse_model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                data=split_subset,
                num_observed_ports=num_observed_ports,
                observed_indices=np.asarray(observed_indices, dtype=np.int64),
                total_ports=total_ports,
                thresholds=missing_thresholds,
                snr_linear=SNR_LINEAR,
                batch_size=BATCH_SIZE,
            )
            for threshold in missing_thresholds:
                results_df = results_by_threshold[threshold]
                results_df["split"] = split_name
                results_df["regime"] = regime
                split_results_frames_by_threshold[threshold].append(results_df)

                regime_output_dir = output_dirs_by_threshold[threshold] / split_name / regime
                save_op_outputs(
                    results_df=results_df,
                    predictions_df=predictions_df,
                    output_dir=regime_output_dir,
                    max_samples=100,
                )
                print(
                    f"{GREEN}[Split {split_idx}/{total_splits}] [Regime {regime_idx}/{total_regimes}] "
                    f"Saved variant={variant_name}, split={split_name}, regime={regime}, "
                    f"threshold={threshold:.8g} outputs to {regime_output_dir}{RESET}"
                )

        for threshold in missing_thresholds:
            split_results_df = pd.concat(
                split_results_frames_by_threshold[threshold],
                axis=0,
                ignore_index=True,
            )
            split_output_dir = output_dirs_by_threshold[threshold] / split_name
            split_results_path = split_output_dir / "op_results_by_regime.csv"
            split_results_df.to_csv(split_results_path, index=False)
            combined_results_frames_by_threshold[threshold].append(split_results_df)

    for threshold in missing_thresholds:
        combined_results_df = pd.concat(
            combined_results_frames_by_threshold[threshold],
            axis=0,
            ignore_index=True,
        )
        combined_results_path = output_dirs_by_threshold[threshold] / "op_results_by_split_and_regime.csv"
        combined_results_df.to_csv(combined_results_path, index=False)
    print(f"{GREEN}MobileNet Keras OP evaluation finished successfully.{RESET}")

    return outputs_by_threshold


def run_op_eval_mobilenet(
    train_run_dirs: list[Path] | None = None,
    root_search_dir: list[Path] | None = ROOT_SEARCH_DIR,
) -> dict[Path, dict[float, dict[str, Path]]]:
    """Run OP evaluation for each MobileNet Keras run.

    Args:
        train_run_dirs (list[Path] | None): Explicit list of MobileNet run directories
            to evaluate. Used only if root_search_dir is None.
        root_search_dir (list[Path] | None): If provided, automatically discovers
            candidate runs from each direct child directory and validates each
            candidate explicitly. If None, uses train_run_dirs or the
            module-level TRAIN_RUN_DIR list.

    Returns:
        dict[Path, dict[float, dict[str, Path]]]: Mapping from each evaluated
            run directory to threshold-specific output paths returned by
            ``_run_op_eval_mobilenet_for_one_dir``.

    Raises:
        ValueError: If no run directories are provided or found.
        FileNotFoundError: If any listed run is missing required metadata or
            artifacts.
        RuntimeError: Propagates model-loading failures from the per-run
            evaluation helper.

    Examples:
        >>> isinstance(TRAIN_RUN_DIR, list)
        True
    """
    if root_search_dir is not None:
        discovered_dirs = []
        for search_dir in root_search_dir:
            if search_dir.exists():
                for child in search_dir.iterdir():
                    if child.is_dir():
                        discovered_dirs.append(child)
        discovered_dirs.sort()
        dirs_to_evaluate = discovered_dirs
    else:
        dirs_to_evaluate = train_run_dirs if train_run_dirs is not None else TRAIN_RUN_DIR

    if not dirs_to_evaluate:
        raise ValueError("Must provide at least one run directory to evaluate.")

    outputs_by_run: dict[Path, dict[float, dict[str, Path]]] = {}
    total_dirs = len(dirs_to_evaluate)

    # Each run writes into its own output tree, so evaluating them one by one
    # keeps artifact boundaries explicit and preserves the existing layout.
    for dir_idx, train_run_dir in enumerate(dirs_to_evaluate, 1):
        print(
            f"\n{BLUE}[Run {dir_idx}/{total_dirs}] [Thresholds {THRESHOLD}] "
            f"Running MobileNet Keras OP evaluation for {train_run_dir}{RESET}"
        )
        outputs_by_run[train_run_dir] = _run_op_eval_mobilenet_for_one_dir(
            train_run_dir=train_run_dir,
            thresholds=THRESHOLD,
        )

    return outputs_by_run


if __name__ == "__main__":
    run_op_eval_mobilenet()
