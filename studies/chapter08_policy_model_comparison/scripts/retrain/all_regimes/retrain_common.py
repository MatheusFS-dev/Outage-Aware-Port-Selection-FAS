"""Shared training workflow for NAS-selected port-reading experiments."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import gc
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/retrain/all_regimes/retrain_common.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter06_mobilenet_nas.scripts.nas.safe_selector_nas import (
    build_nas_selector_models,
)
from mobilenet.keras.sparse_input import build_sparse_port_images
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    OUTAGE_THRESHOLD,
)
from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import build_uni_indices
from utils.data import KMS_REGIME_LABELS, split_kms_60_20_20_cache
from utils.qttf import (
    build_quantile_balanced_calibration_data,
    convert_ptq_model_to_strict_int8_tflite,
    test_strict_int8_io,
)


TOTAL_PORTS = 100
IMAGE_SIDE = 32
DATA_SEED = 0
TRAIN_SEEDS = (0, 1, 2, 3, 4)
SUPPORTED_POLICIES = ("uni", "mar", "gc")
OTHER_RETRAIN_POLICIES = ("suni", "swuni", "rag", "rnd")
TRAINING_POLICIES = SUPPORTED_POLICIES + OTHER_RETRAIN_POLICIES
SUPPORTED_BUDGETS = (3, 4, 5, 6, 7)
EPOCHS = 100
BATCH_SIZE = 4096
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-10
USE_JIT_COMPILE = True
PRECISION_POLICY = "float32"
INT8_CALIBRATION_SAMPLE_COUNT = 256
INT8_CALIBRATION_NUM_BINS = 16
INT8_CALIBRATION_SEED = 0
RUN_ROOT = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/retrain/all_regimes/runs'
PATTERN_MANIFEST_PATH = RUN_ROOT / "port_patterns.json"


@dataclass(frozen=True)
class TopModelConfig:
    """Describe one selected all-regime MobileNetV3 NAS architecture.

    Args:
        budget (int): Number of observed ports accepted by the model. The
            supported campaign budgets are 3 through 7.
        trial (int): Zero-based Optuna trial identifier from the corresponding
            all-regime NAS study.
        alpha (float): MobileNetV3 width multiplier selected by NAS.
        truncation_block_name (str): Final MobileNetV3 feature block retained
            by the selected architecture.
        head_depth (int): Number of dense layers in the selector head.
        head_width (int): Units in each dense selector-head layer.
        learning_rate (float): Initial Adam learning rate selected by NAS.
        source_model_path (Path): Original rank-1 Keras artifact used only for
            structural parity checks. Its trained weights are not reused.

    Returns:
        None: Instances store immutable architecture provenance.

    Raises:
        None.
    """

    budget: int
    trial: int
    alpha: float
    truncation_block_name: str
    head_depth: int
    head_width: int
    learning_rate: float
    source_model_path: Path


WEIGHTS_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/models/all_regimes/nas'
TOP_MODEL_CONFIGS = {
    3: TopModelConfig(
        budget=3,
        trial=64,
        alpha=0.8,
        truncation_block_name="v3_block_4_add",
        head_depth=1,
        head_width=144,
        learning_rate=0.00014092436678173658,
        source_model_path=WEIGHTS_ROOT / "3_ports" / "top_1_trial_64.keras",
    ),
    4: TopModelConfig(
        budget=4,
        trial=94,
        alpha=0.7,
        truncation_block_name="v3_block_1_project_gn",
        head_depth=1,
        head_width=64,
        learning_rate=0.00011044731521262944,
        source_model_path=WEIGHTS_ROOT / "4_ports" / "top_1_trial_94.keras",
    ),
    5: TopModelConfig(
        budget=5,
        trial=46,
        alpha=0.3,
        truncation_block_name="v3_block_3_project_gn",
        head_depth=1,
        head_width=32,
        learning_rate=0.001657193428459794,
        source_model_path=WEIGHTS_ROOT / "5_ports" / "top_1_trial_46.keras",
    ),
    6: TopModelConfig(
        budget=6,
        trial=84,
        alpha=0.7,
        truncation_block_name="v3_block_3_project_gn",
        head_depth=1,
        head_width=80,
        learning_rate=0.00021976735511661943,
        source_model_path=WEIGHTS_ROOT / "6_ports" / "top_1_trial_84.keras",
    ),
    7: TopModelConfig(
        budget=7,
        trial=38,
        alpha=0.6,
        truncation_block_name="v3_block_3_project_gn",
        head_depth=1,
        head_width=32,
        learning_rate=0.0001318104936913322,
        source_model_path=WEIGHTS_ROOT / "7_ports" / "top_1_trial_38.keras",
    ),
}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object through an atomic sibling-file replacement.

    Args:
        path (Path): Final JSON path. Existing content is replaced only after
            the complete temporary document has been written.
        payload (dict[str, Any]): JSON-serializable object to persist.

    Returns:
        None: The function writes the document as a side effect.

    Raises:
        OSError: If the directory, temporary file, or atomic replacement cannot
            be created.
        TypeError: If ``payload`` contains a non-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object from disk.

    Args:
        path (Path): Existing JSON document to read.

    Returns:
        dict[str, Any]: Parsed top-level JSON object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
        ValueError: If the top-level JSON value is not an object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def payload_digest(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible object.

    Args:
        payload (dict[str, Any]): JSON-compatible object whose key ordering
            should not affect the digest.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.

    Raises:
        TypeError: If ``payload`` is not JSON serializable.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_policy_budget(policy: str, budget: int) -> None:
    """Reject unsupported policy names or observation budgets.

    Args:
        policy (str): Reading-policy identifier. Accepted values are the three
            dissertation policies and the four isolated other-retrain policies.
        budget (int): Requested observed-port count. Accepted values are 3
            through 7.

    Returns:
        None: Valid inputs pass without mutation.

    Raises:
        ValueError: If either input is outside the controlled campaign matrix.
    """
    if policy not in TRAINING_POLICIES:
        raise ValueError(
            f"Unsupported policy {policy!r}; expected one of {TRAINING_POLICIES}."
        )
    if budget not in SUPPORTED_BUDGETS:
        raise ValueError(
            f"Unsupported budget {budget}; expected one of {SUPPORTED_BUDGETS}."
        )


def compute_pooled_mar_statistics(
    train_parts: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled threshold-exceedance scores and mean values by port.

    The computation streams over regime partitions so it does not allocate a
    Boolean matrix for the full pooled training split.

    Args:
        train_parts (tuple[np.ndarray, ...]): Per-regime 60% training matrices.
            Every matrix must contain 100 port columns and at least one row.

    Returns:
        tuple[np.ndarray, np.ndarray]: Exceedance frequencies and pooled mean
        values, each with shape ``(100,)``.

    Raises:
        ValueError: If no partitions are supplied or a partition has an invalid
            shape.

    Examples:
        >>> parts = (np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),)
        >>> scores, means = compute_pooled_mar_statistics(parts)
        Traceback (most recent call last):
        ...
        ValueError: Every training partition must have shape (n, 100).
    """
    if not train_parts:
        raise ValueError("At least one training partition is required.")

    success_counts = np.zeros(TOTAL_PORTS, dtype=np.int64)
    value_sums = np.zeros(TOTAL_PORTS, dtype=np.float64)
    total_rows = 0
    for part in train_parts:
        if part.ndim != 2 or part.shape[0] == 0 or part.shape[1] != TOTAL_PORTS:
            raise ValueError("Every training partition must have shape (n, 100).")
        success_counts += np.count_nonzero(part > OUTAGE_THRESHOLD, axis=0)
        value_sums += np.sum(part, axis=0, dtype=np.float64)
        total_rows += len(part)

    return success_counts / total_rows, value_sums / total_rows


def build_pooled_mar_order(
    exceedance_scores: np.ndarray,
    design_means: np.ndarray,
) -> np.ndarray:
    """Rank ports using the SBrT marginal-policy ordering and tie-breaks.

    Args:
        exceedance_scores (np.ndarray): Pooled per-port threshold-exceedance
            frequencies with shape ``(100,)``.
        design_means (np.ndarray): Pooled per-port mean values with shape
            ``(100,)``. Higher means break equal exceedance scores.

    Returns:
        np.ndarray: Complete zero-based port order from most to least preferred.

    Raises:
        ValueError: If either vector does not have exactly 100 entries.
    """
    if exceedance_scores.shape != (TOTAL_PORTS,):
        raise ValueError("exceedance_scores must have shape (100,).")
    if design_means.shape != (TOTAL_PORTS,):
        raise ValueError("design_means must have shape (100,).")
    indices = np.arange(TOTAL_PORTS)
    return np.lexsort((indices, -design_means, -exceedance_scores)).astype(int)


def build_pooled_gc_order(
    train_parts: tuple[np.ndarray, ...],
    exceedance_scores: np.ndarray,
    design_means: np.ndarray,
    max_budget: int,
) -> np.ndarray:
    """Build pooled greedy coverage while streaming regime partitions.

    Args:
        train_parts (tuple[np.ndarray, ...]): Per-regime 60% training matrices
            used as the pooled policy-design data.
        exceedance_scores (np.ndarray): MAR exceedance frequencies used as the
            first deterministic tie-break.
        design_means (np.ndarray): Pooled mean values used as the second
            deterministic tie-break.
        max_budget (int): Number of greedy selections to produce. The campaign
            uses seven so every requested budget is a prefix of one order.

    Returns:
        np.ndarray: Zero-based greedy port order of length ``max_budget``.

    Raises:
        ValueError: If ``max_budget`` is invalid or the score shapes are not
            aligned with 100 ports.
    """
    if max_budget < 1 or max_budget > TOTAL_PORTS:
        raise ValueError("max_budget must satisfy 1 <= max_budget <= 100.")
    if exceedance_scores.shape != (TOTAL_PORTS,):
        raise ValueError("exceedance_scores must have shape (100,).")
    if design_means.shape != (TOTAL_PORTS,):
        raise ValueError("design_means must have shape (100,).")

    mar_order = build_pooled_mar_order(exceedance_scores, design_means)
    selected = np.zeros(TOTAL_PORTS, dtype=bool)
    covered_parts = [np.zeros(len(part), dtype=bool) for part in train_parts]
    order: list[int] = []

    while len(order) < max_budget:
        cover_counts = np.zeros(TOTAL_PORTS, dtype=np.int64)
        any_uncovered = False
        for part, covered in zip(train_parts, covered_parts, strict=True):
            uncovered = ~covered
            if not np.any(uncovered):
                continue
            any_uncovered = True
            cover_counts += np.count_nonzero(
                part[uncovered] > OUTAGE_THRESHOLD,
                axis=0,
            )

        if not any_uncovered:
            order.extend(
                int(index)
                for index in mar_order
                if not selected[index]
            )
            break

        cover_counts[selected] = -1
        best_count = int(np.max(cover_counts))
        candidates = np.flatnonzero(cover_counts == best_count)
        candidate_order = np.lexsort(
            (
                candidates,
                -design_means[candidates],
                -exceedance_scores[candidates],
            )
        )
        best_port = int(candidates[candidate_order[0]])
        order.append(best_port)
        selected[best_port] = True
        for part, covered in zip(train_parts, covered_parts, strict=True):
            covered |= part[:, best_port] > OUTAGE_THRESHOLD

    return np.asarray(order[:max_budget], dtype=int)


def build_pattern_manifest(
    train_parts: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """Construct all 15 fixed pooled-training reading patterns.

    Args:
        train_parts (tuple[np.ndarray, ...]): The 14 per-regime 60% training
            partitions created with data seed zero. Validation and test rows
            must not be supplied.

    Returns:
        dict[str, Any]: Versioned manifest containing UNI, MAR, and GC indices
        for budgets 3 through 7 in zero-based and one-based form.

    Raises:
        ValueError: If the partitions are incomplete or a generated pattern is
            duplicated, out of range, or has the wrong size.

    Examples:
        This production function expects the complete 100-port partitions:

        >>> callable(build_pattern_manifest)
        True
    """
    if len(train_parts) != len(KMS_REGIME_LABELS):
        raise ValueError(
            f"Expected {len(KMS_REGIME_LABELS)} training partitions, "
            f"received {len(train_parts)}."
        )

    scores, means = compute_pooled_mar_statistics(train_parts)
    mar_order = build_pooled_mar_order(scores, means)
    gc_order = build_pooled_gc_order(
        train_parts=train_parts,
        exceedance_scores=scores,
        design_means=means,
        max_budget=max(SUPPORTED_BUDGETS),
    )
    patterns: dict[str, dict[str, dict[str, list[int]]]] = {}
    for policy in SUPPORTED_POLICIES:
        patterns[policy] = {}
        for budget in SUPPORTED_BUDGETS:
            if policy == "uni":
                indices = build_uni_indices(TOTAL_PORTS, budget)
            elif policy == "mar":
                indices = mar_order[:budget]
            else:
                indices = gc_order[:budget]
            sorted_indices = np.sort(np.asarray(indices, dtype=int))
            if len(np.unique(sorted_indices)) != budget:
                raise ValueError(f"{policy.upper()} N={budget} contains duplicate ports.")
            if sorted_indices[0] < 0 or sorted_indices[-1] >= TOTAL_PORTS:
                raise ValueError(f"{policy.upper()} N={budget} is out of range.")
            patterns[policy][str(budget)] = {
                "selection_order_0based": [int(value) for value in indices],
                "sorted_pattern_0based": [int(value) for value in sorted_indices],
                "sorted_pattern_1based": [int(value) + 1 for value in sorted_indices],
            }

    return {
        "schema_version": 1,
        "data_seed": DATA_SEED,
        "split_mode": "60_20_20",
        "policy_design_scope": "pooled_training_rows_across_14_regimes",
        "threshold": OUTAGE_THRESHOLD,
        "total_ports": TOTAL_PORTS,
        "regime_labels": list(KMS_REGIME_LABELS),
        "patterns": patterns,
    }


def load_or_create_pattern_manifest(
    train_parts: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """Load the shared pattern manifest or create it from training rows.

    Args:
        train_parts (tuple[np.ndarray, ...]): Per-regime training partitions
            used only when the manifest does not yet exist.

    Returns:
        dict[str, Any]: Validated manifest for the controlled campaign.

    Raises:
        ValueError: If an existing manifest uses incompatible threshold, split,
            seed, port count, regime labels, schema, or pattern structure.
        OSError: If a new manifest cannot be written atomically.
    """
    if PATTERN_MANIFEST_PATH.exists():
        manifest = read_json(PATTERN_MANIFEST_PATH)
    else:
        manifest = build_pattern_manifest(train_parts)
        write_json_atomic(PATTERN_MANIFEST_PATH, manifest)

    expected_metadata = {
        "schema_version": 1,
        "data_seed": DATA_SEED,
        "split_mode": "60_20_20",
        "policy_design_scope": "pooled_training_rows_across_14_regimes",
        "threshold": OUTAGE_THRESHOLD,
        "total_ports": TOTAL_PORTS,
        "regime_labels": list(KMS_REGIME_LABELS),
    }
    actual_metadata = {
        key: manifest.get(key)
        for key in expected_metadata
    }
    if actual_metadata != expected_metadata:
        raise ValueError(
            f"Incompatible pattern manifest at {PATTERN_MANIFEST_PATH}."
        )
    for policy in SUPPORTED_POLICIES:
        for budget in SUPPORTED_BUDGETS:
            pattern = manifest.get("patterns", {}).get(policy, {}).get(str(budget))
            if not isinstance(pattern, dict):
                raise ValueError(f"Pattern manifest is missing {policy} N={budget}.")
            indices = pattern.get("sorted_pattern_0based")
            if not isinstance(indices, list) or len(indices) != budget:
                raise ValueError(f"Pattern manifest has invalid {policy} N={budget}.")
    return manifest


def get_pattern_indices(
    manifest: dict[str, Any],
    policy: str,
    budget: int,
) -> np.ndarray:
    """Extract one sorted zero-based reading pattern from a manifest.

    Args:
        manifest (dict[str, Any]): Validated pattern manifest returned by
            :func:`load_or_create_pattern_manifest`.
        policy (str): One of ``uni``, ``mar``, or ``gc``.
        budget (int): Number of observed ports, from 3 through 7.

    Returns:
        np.ndarray: Strictly increasing int64 port indices of length ``budget``.

    Raises:
        ValueError: If the requested case is unsupported or its stored indices
            are malformed.
    """
    validate_policy_budget(policy, budget)
    values = manifest["patterns"][policy][str(budget)]["sorted_pattern_0based"]
    indices = np.asarray(values, dtype=np.int64)
    if indices.shape != (budget,) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"Stored pattern for {policy} N={budget} is invalid.")
    return indices


def build_training_data_for_indices(
    y_values: np.ndarray,
    observed_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build outage-aware selector inputs and equal-mass targets.

    Args:
        y_values (np.ndarray): Full channel matrix with shape ``(n, 100)``.
        observed_indices (np.ndarray): Sorted fixed observed-port indices. The
            selected policy and budget determine its length.

    Returns:
        tuple[np.ndarray, np.ndarray]: Compact observed inputs and 100-class
        equal-mass targets for recoverable samples.

    Raises:
        ValueError: If inputs are malformed or no recoverable samples exist.
    """
    if y_values.ndim != 2 or y_values.shape[1] != TOTAL_PORTS:
        raise ValueError("y_values must have shape (n, 100).")
    if observed_indices.ndim != 1 or len(observed_indices) == 0:
        raise ValueError("observed_indices must be a non-empty vector.")
    if np.any(np.diff(observed_indices) <= 0):
        raise ValueError("observed_indices must be strictly increasing.")

    observed_values = y_values[:, observed_indices]
    observed_failed = np.max(observed_values, axis=1) <= OUTAGE_THRESHOLD
    unobserved_good = y_values > OUTAGE_THRESHOLD
    unobserved_good[:, observed_indices] = False
    recoverable = observed_failed & np.any(unobserved_good, axis=1)
    if not np.any(recoverable):
        raise ValueError("No recoverable samples exist for this reading pattern.")

    inputs = observed_values[recoverable].astype(np.float32)
    targets = unobserved_good[recoverable].astype(np.float32)
    targets /= np.sum(targets, axis=1, keepdims=True)
    return inputs, targets


def resolve_steps_per_execution(
    train_size: int,
    validation_size: int,
    batch_size: int,
) -> int:
    """Resolve a safe automatic Keras steps-per-execution value.

    Args:
        train_size (int): Number of recoverable training samples.
        validation_size (int): Number of recoverable validation samples.
        batch_size (int): Positive batch size used by both loops.

    Returns:
        int: Positive execution grouping no larger than either loop's number of
        batches. The automatic target is capped at 64 batches.

    Raises:
        ValueError: If any size is non-positive.
    """
    if train_size <= 0 or validation_size <= 0 or batch_size <= 0:
        raise ValueError("Dataset sizes and batch_size must be positive.")
    train_batches = int(np.ceil(train_size / batch_size))
    validation_batches = int(np.ceil(validation_size / batch_size))
    return max(1, min(64, train_batches, validation_batches))


def build_selected_models(
    config: TopModelConfig,
    observed_indices: np.ndarray,
    log_scale: float,
    train_seed: int,
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build one selected NAS architecture with fresh initialized weights.

    Args:
        config (TopModelConfig): Budget-specific rank-1 NAS configuration.
        observed_indices (np.ndarray): Fixed reading-pattern indices used by
            embedded input, mask, and observed-port guard layers.
        log_scale (float): Positive pooled-training normalization scale.
        train_seed (int): Weight initialization and training seed. Different
            seeds change learned weights but not architecture.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]: Explorer,
        compact decision, and sparse-protocol models sharing fresh weights.

    Raises:
        ValueError: If the NAS builder rejects a configuration or input.
    """
    return build_nas_selector_models(
        observed_indices=observed_indices,
        log_scale=log_scale,
        alpha=config.alpha,
        truncation_block_name=config.truncation_block_name,
        head_depth=config.head_depth,
        head_width=config.head_width,
        train_seed=train_seed,
        variant_name="mobv3_alpha02",
    )


def validate_architecture_parity(
    config: TopModelConfig,
    observed_indices: np.ndarray,
    log_scale: float,
) -> None:
    """Verify rebuilt parameter counts and weight shapes against rank 1.

    Args:
        config (TopModelConfig): Selected NAS architecture and source artifact.
        observed_indices (np.ndarray): New policy-specific indices for the same
            observation budget.
        log_scale (float): Positive normalization scale used to build the new
            model. This changes constants but not trainable structure.

    Returns:
        None: Matching architectures pass without retaining either model.

    Raises:
        FileNotFoundError: If the source top-1 Keras artifact is missing.
        ValueError: If parameter counts or ordered weight shapes differ.
    """
    if not config.source_model_path.is_file():
        raise FileNotFoundError(
            f"Missing top-1 source model: {config.source_model_path}"
        )
    _, _, rebuilt = build_selected_models(
        config=config,
        observed_indices=observed_indices,
        log_scale=log_scale,
        train_seed=0,
    )
    source = tf.keras.models.load_model(config.source_model_path, compile=False)
    rebuilt_shapes = [tuple(weight.shape) for weight in rebuilt.weights]
    source_shapes = [tuple(weight.shape) for weight in source.weights]
    if rebuilt.count_params() != source.count_params():
        raise ValueError(
            f"Parameter-count mismatch for N={config.budget}: "
            f"rebuilt={rebuilt.count_params()}, source={source.count_params()}."
        )
    if rebuilt_shapes != source_shapes:
        raise ValueError(f"Ordered weight-shape mismatch for N={config.budget}.")
    del rebuilt
    del source
    tf.keras.backend.clear_session()
    gc.collect()


def build_run_contract(
    policy: str,
    config: TopModelConfig,
    observed_indices: np.ndarray,
    train_seed: int,
    log_scale: float,
) -> dict[str, Any]:
    """Build immutable metadata used to validate restart compatibility.

    Args:
        policy (str): Reading policy for this case.
        config (TopModelConfig): Selected NAS architecture configuration.
        observed_indices (np.ndarray): Sorted fixed port indices.
        train_seed (int): Weight initialization and training seed.
        log_scale (float): Pooled-training normalization constant.

    Returns:
        dict[str, Any]: JSON-compatible immutable run contract.

    Raises:
        ValueError: If the policy or budget is unsupported.
    """
    validate_policy_budget(policy, config.budget)
    architecture = asdict(config)
    architecture["source_model_path"] = str(config.source_model_path)
    return {
        "schema_version": 2,
        "variant_name": "mobv3_alpha02",
        "method_name": "outage_aware_conditional_port_selection",
        "task_type": "safe_port_selector",
        "policy": policy,
        "budget": config.budget,
        "num_observed_ports": config.budget,
        "total_ports": TOTAL_PORTS,
        "observed_indices": [int(value) for value in observed_indices],
        "unobserved_indices": [
            int(value)
            for value in np.setdiff1d(
                np.arange(TOTAL_PORTS, dtype=np.int64),
                observed_indices,
            )
        ],
        "architecture": architecture,
        "threshold": OUTAGE_THRESHOLD,
        "objective": "categorical_crossentropy_equal_mass_acceptable_unobserved_ports",
        "data_seed": DATA_SEED,
        "train_seed": train_seed,
        "split_mode": "60_20_20",
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "samples_per_regime": None,
        "train_regime_labels": list(KMS_REGIME_LABELS),
        "input_shape": [IMAGE_SIDE, IMAGE_SIDE, 3],
        "compact_selector_input_shape": [config.budget],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "precision_policy": PRECISION_POLICY,
        "jit_compile": USE_JIT_COMPILE,
        "log_scale": log_scale,
        "quantization": {
            "method": "post_training_static_range",
            "supported_ops": "TFLITE_BUILTINS_INT8",
            "inference_input_type": "int8",
            "inference_output_type": "int8",
            "calibration_sample_count": INT8_CALIBRATION_SAMPLE_COUNT,
            "calibration_num_bins": INT8_CALIBRATION_NUM_BINS,
            "calibration_seed": INT8_CALIBRATION_SEED,
        },
    }


def case_run_dir(
    policy: str,
    budget: int,
    train_seed: int,
    run_root: Path | None = None,
) -> Path:
    """Return the stable artifact directory for one training case.

    Args:
        policy (str): Reading-policy identifier.
        budget (int): Observed-port budget.
        train_seed (int): Training seed used to isolate replicates.
        run_root (Path | None): Optional artifact root. ``None`` uses the
            dissertation-policy retraining root for backward compatibility.

    Returns:
        Path: Case directory below the ignored retraining run root.

    Raises:
        ValueError: If the policy, budget, or seed is invalid.
    """
    validate_policy_budget(policy, budget)
    if train_seed not in TRAIN_SEEDS:
        raise ValueError(f"Unsupported training seed {train_seed}.")
    output_root = RUN_ROOT if run_root is None else run_root
    return (
        output_root
        / f"seed_{train_seed}"
        / f"train_nas_top1_{policy}_{budget}_ports"
    )


def required_case_artifacts(run_dir: Path) -> tuple[Path, ...]:
    """List final artifacts required before a case can be skipped.

    Args:
        run_dir (Path): Top-level directory for one policy-budget-seed case.

    Returns:
        tuple[Path, ...]: Required non-empty final file paths.

    Raises:
        None.
    """
    training_dir = run_dir / "model_training"
    return (
        training_dir / "args" / "run_contract.json",
        training_dir / "args" / "mobilenet_config.json",
        training_dir / "args" / "model_stats.txt",
        training_dir / "model" / "model.keras",
        training_dir / "model" / "compact_decision_model.keras",
        training_dir / "model" / "explorer_model.keras",
        training_dir / "tflite" / "model_int8.tflite",
        training_dir / "scaler" / "scaler_X.pkl",
        training_dir / "scaler" / "scaler_y.pkl",
        training_dir / "history" / "history.csv",
        training_dir / "history" / "selection_history.csv",
        training_dir / "checkpoint" / "best.weights.h5",
    )


def is_complete_training_case(
    run_dir: Path,
    expected_contract: dict[str, Any] | None = None,
) -> bool:
    """Return whether a run has a matching marker and every final artifact.

    Args:
        run_dir (Path): Case directory to inspect.
        expected_contract (dict[str, Any] | None): Optional current contract.
            When provided, its digest must match both stored metadata files.

    Returns:
        bool: ``True`` only for a fully materialized compatible run.

    Raises:
        ValueError: If a completion marker exists but conflicts with the stored
            or expected run contract.
    """
    marker_path = run_dir / "completion.json"
    if not marker_path.is_file():
        return False
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_case_artifacts(run_dir)):
        return False

    marker = read_json(marker_path)
    stored_contract = read_json(
        run_dir / "model_training" / "args" / "run_contract.json"
    )
    stored_digest = payload_digest(stored_contract)
    if marker.get("status") != "complete" or marker.get("contract_sha256") != stored_digest:
        raise ValueError(f"Invalid completion metadata in {run_dir}.")
    if expected_contract is not None and payload_digest(expected_contract) != stored_digest:
        raise ValueError(f"Completed run contract differs at {run_dir}.")
    return True


def prepare_case_directories(
    run_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Path]:
    """Create or validate the artifact tree for a resumable case.

    Args:
        run_dir (Path): Stable case directory.
        contract (dict[str, Any]): Immutable compatibility contract for the
            requested policy, architecture, seed, and training protocol.

    Returns:
        dict[str, Path]: Named artifact directories used by training.

    Raises:
        ValueError: If an existing contract differs or partial artifacts exist
            without a contract.
        OSError: If directories or the initial contract cannot be created.
    """
    paths = {
        "args": run_dir / "model_training" / "args",
        "model": run_dir / "model_training" / "model",
        "tflite": run_dir / "model_training" / "tflite",
        "scaler": run_dir / "model_training" / "scaler",
        "history": run_dir / "model_training" / "history",
        "checkpoint": run_dir / "model_training" / "checkpoint",
        "backup": run_dir / "model_training" / "checkpoint" / "latest",
    }
    contract_path = paths["args"] / "run_contract.json"
    if run_dir.exists() and not contract_path.exists():
        partial_files = [path for path in run_dir.rglob("*") if path.is_file()]
        if partial_files:
            raise ValueError(
                f"Partial artifacts exist without a run contract in {run_dir}."
            )
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError(f"Run contract mismatch in {run_dir}.")
    else:
        write_json_atomic(contract_path, contract)
    return paths


def save_identity_scalers(
    scaler_dir: Path,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
) -> None:
    """Persist explicit identity transformers for evaluation compatibility.

    Args:
        scaler_dir (Path): Existing output directory for both pickle files.
        train_inputs (np.ndarray): Compact training inputs used to fit the
            input identity transformer on at most 1024 samples.
        train_targets (np.ndarray): Training targets used to fit the target
            identity transformer on at most 1024 samples.

    Returns:
        None: Two pickle artifacts are written as a side effect.

    Raises:
        OSError: If either output file cannot be written.
        ValueError: If scikit-learn rejects the provided arrays.
    """
    scaler_x = FunctionTransformer(validate=True)
    scaler_y = FunctionTransformer(validate=True)
    scaler_x.fit(train_inputs[: min(1024, len(train_inputs))])
    scaler_y.fit(train_targets[: min(1024, len(train_targets))])
    with open(scaler_dir / "scaler_X.pkl", "wb") as file_obj:
        pickle.dump(scaler_x, file_obj)
    with open(scaler_dir / "scaler_y.pkl", "wb") as file_obj:
        pickle.dump(scaler_y, file_obj)


def build_int8_calibration_data(
    y_train: np.ndarray,
    observed_indices: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Build the NAS-equivalent train-only strict-INT8 calibration set.

    Args:
        y_train (np.ndarray): Pooled training matrix with shape ``(n, 100)``.
            Validation and test rows must not be included.
        observed_indices (np.ndarray): Sorted zero-based observed-port indices
            for the current policy and budget.

    Returns:
        tuple[np.ndarray, list[int]]: Sparse ``32 x 32 x 3`` representative
        images and the number of selected samples in each of the 16 target
        quantile bins.

    Raises:
        ValueError: If the training matrix or observed indices are invalid, or
            the shared NAS calibration selector rejects the data.

    Examples:
        Production calls use the complete pooled training split:

        >>> callable(build_int8_calibration_data)
        True
    """
    if y_train.ndim != 2 or y_train.shape[0] == 0:
        raise ValueError("y_train must be a non-empty two-dimensional matrix.")
    if y_train.shape[1] != TOTAL_PORTS:
        raise ValueError("y_train must have 100 port columns.")
    if (
        observed_indices.ndim != 1
        or len(observed_indices) == 0
        or len(np.unique(observed_indices)) != len(observed_indices)
        or np.any(observed_indices < 0)
        or np.any(observed_indices >= TOTAL_PORTS)
    ):
        raise ValueError("observed_indices must contain unique ports in [0, 99].")

    observed_values = y_train[:, observed_indices].astype(np.float32)
    calibration_values, calibration_counts = (
        build_quantile_balanced_calibration_data(
            observed_values,
            y_train,
            sample_count=INT8_CALIBRATION_SAMPLE_COUNT,
            num_bins=INT8_CALIBRATION_NUM_BINS,
            seed=INT8_CALIBRATION_SEED,
        )
    )
    calibration_data = build_sparse_port_images(
        observed_values=calibration_values,
        observed_indices=observed_indices.astype(int),
        total_ports=TOTAL_PORTS,
        image_height=IMAGE_SIDE,
        image_width=IMAGE_SIDE,
    )
    return calibration_data, calibration_counts


def save_strict_int8_tflite(
    model: tf.keras.Model,
    calibration_data: np.ndarray,
    output_path: Path,
) -> int:
    """Convert, validate, and atomically save one strict-INT8 TFLite model.

    Args:
        model (tf.keras.Model): Trained sparse-protocol model with floating-point
            input and output tensors before post-training quantization.
        calibration_data (np.ndarray): Train-only representative sparse images
            accepted by the model input.
        output_path (Path): Final `.tflite` path. A sibling temporary file is
            replaced only after conversion and strict endpoint validation pass.

    Returns:
        int: Serialized TFLite model size in bytes.

    Raises:
        AssertionError: If the converted model does not expose INT8 input and
            output tensors.
        OSError: If the destination directory or model file cannot be written.
        Exception: Propagates TensorFlow Lite conversion and allocation errors.

    Examples:
        Production passes a trained protocol model and train-only calibration:

        >>> callable(save_strict_int8_tflite)
        True
    """
    int8_tflite_model = convert_ptq_model_to_strict_int8_tflite(
        model,
        calibration_data,
    )
    test_strict_int8_io(int8_tflite_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_bytes(int8_tflite_model)
    temporary_path.replace(output_path)
    return len(int8_tflite_model)


def compute_validation_snapshot(
    decision_model: tf.keras.Model,
    y_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Compute final validation outage and selection-diversity diagnostics.

    Args:
        decision_model (tf.keras.Model): Compact decision-score model whose
            output argmax selects a physical port.
        y_values (np.ndarray): Full validation channel matrix with 100 columns.
        observed_indices (np.ndarray): Fixed observed-port indices used by the
            model and observed-only reference.
        batch_size (int): Positive prediction batch size.

    Returns:
        dict[str, float]: Model-only outage, observed-only outage, and number of
        distinct model-selected ports.

    Raises:
        ValueError: If data or index shapes are invalid.
    """
    if y_values.ndim != 2 or y_values.shape[1] != TOTAL_PORTS:
        raise ValueError("y_values must have shape (n, 100).")
    observed_values = y_values[:, observed_indices].astype(np.float32)
    scores = decision_model.predict(observed_values, batch_size=batch_size, verbose=0)
    selected_ports = np.argmax(scores, axis=1)
    rows = np.arange(len(y_values))
    return {
        "model_only_op": float(
            np.mean(y_values[rows, selected_ports] <= OUTAGE_THRESHOLD)
        ),
        "observed_only_op": float(
            np.mean(np.max(observed_values, axis=1) <= OUTAGE_THRESHOLD)
        ),
        "distinct_selected_ports": int(np.unique(selected_ports).size),
    }


def read_history(history_path: Path) -> pd.DataFrame:
    """Read a non-empty Keras CSV history with required validation loss.

    Args:
        history_path (Path): CSVLogger output from the current case.

    Returns:
        pd.DataFrame: Parsed history ordered as written by Keras.

    Raises:
        FileNotFoundError: If the history does not exist.
        ValueError: If it is empty or lacks ``epoch`` or ``val_loss``.
    """
    history = pd.read_csv(history_path)
    if history.empty or not {"epoch", "val_loss"}.issubset(history.columns):
        raise ValueError(f"Invalid training history at {history_path}.")
    return history


def train_one_case(
    policy: str,
    config: TopModelConfig,
    observed_indices: np.ndarray,
    log_scale: float,
    train_seed: int,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    y_validation: np.ndarray,
    calibration_data: np.ndarray,
    calibration_counts: list[int],
    run_root: Path | None = None,
) -> None:
    """Train or resume one policy-budget-seed selected architecture.

    Args:
        policy (str): Reading-policy identifier.
        config (TopModelConfig): Budget-specific selected NAS architecture.
        observed_indices (np.ndarray): Fixed policy-specific port indices.
        log_scale (float): Pooled-training normalization constant.
        train_seed (int): Seed controlling model initialization and shuffling.
        train_inputs (np.ndarray): Recoverable compact training inputs.
        train_targets (np.ndarray): Equal-mass 100-class training targets.
        validation_inputs (np.ndarray): Recoverable compact validation inputs.
        validation_targets (np.ndarray): Equal-mass validation targets.
        y_validation (np.ndarray): Regime-balanced validation subset used for
            final OP diagnostics after best-weight restoration.
        calibration_data (np.ndarray): Train-only sparse images used for
            strict-INT8 post-training quantization.
        calibration_counts (list[int]): Representative-sample counts for the
            16 target-quantile bins used by the NAS calibration procedure.
        run_root (Path | None): Optional artifact root used to isolate a
            retraining campaign. ``None`` preserves the established root.

    Returns:
        None: Artifacts and the final atomic completion marker are written under
        the stable case directory.

    Raises:
        ValueError: If stored state is incompatible or training artifacts are
            inconsistent.
        OSError: If an artifact cannot be written.
        Exception: Propagates TensorFlow training and serialization failures so
            the global runner stops without marking the case complete.

    Examples:
        Production calls are dispatched through a fixed policy-budget script:

        >>> callable(train_one_case)
        True
    """
    run_dir = case_run_dir(
        policy,
        config.budget,
        train_seed,
        run_root=run_root,
    )
    contract = build_run_contract(
        policy=policy,
        config=config,
        observed_indices=observed_indices,
        train_seed=train_seed,
        log_scale=log_scale,
    )
    if is_complete_training_case(run_dir, contract):
        print(f"[SKIP] Complete run: {run_dir}")
        return

    paths = prepare_case_directories(run_dir, contract)
    history_path = paths["history"] / "history.csv"
    best_weights_path = paths["checkpoint"] / "best.weights.h5"
    if history_path.exists() != best_weights_path.exists():
        raise ValueError(
            f"History and best checkpoint must either both exist or both be absent: {run_dir}"
        )

    tf.keras.mixed_precision.set_global_policy(PRECISION_POLICY)
    tf.keras.utils.set_random_seed(train_seed)
    explorer_model, decision_model, protocol_model = build_selected_models(
        config=config,
        observed_indices=observed_indices,
        log_scale=log_scale,
        train_seed=train_seed,
    )
    steps_per_execution = resolve_steps_per_execution(
        train_size=len(train_inputs),
        validation_size=len(validation_inputs),
        batch_size=BATCH_SIZE,
    )
    explorer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="target_argmax_accuracy")
        ],
        jit_compile=USE_JIT_COMPILE,
        steps_per_execution=steps_per_execution,
    )

    initial_best = None
    if history_path.exists():
        initial_best = float(read_history(history_path)["val_loss"].min())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=REDUCE_LR_PATIENCE,
            factor=REDUCE_LR_FACTOR,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=initial_best,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_path, append=history_path.exists()),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=paths["backup"],
            save_freq="epoch",
            delete_checkpoint=False,
        ),
    ]
    print(
        f"[RUN] policy={policy}, budget={config.budget}, seed={train_seed}, "
        f"run_dir={run_dir}"
    )
    explorer_model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    if not best_weights_path.is_file():
        raise ValueError(f"Training produced no best checkpoint in {run_dir}.")
    explorer_model.load_weights(best_weights_path)

    history = read_history(history_path)
    selected_row = history.loc[history["val_loss"].idxmin()]
    snapshot = compute_validation_snapshot(
        decision_model=decision_model,
        y_values=y_validation,
        observed_indices=observed_indices,
        batch_size=BATCH_SIZE,
    )
    paths["model"].mkdir(parents=True, exist_ok=True)
    protocol_model.save(paths["model"] / "model.keras")
    decision_model.save(paths["model"] / "compact_decision_model.keras")
    explorer_model.save(paths["model"] / "explorer_model.keras")
    int8_tflite_size_bytes = save_strict_int8_tflite(
        protocol_model,
        calibration_data,
        paths["tflite"] / "model_int8.tflite",
    )
    save_identity_scalers(paths["scaler"], train_inputs, train_targets)

    selection_history = pd.DataFrame(
        [
            {
                "selected_epoch": int(selected_row["epoch"]) + 1,
                "best_val_loss": float(selected_row["val_loss"]),
                **snapshot,
            }
        ]
    )
    selection_history.to_csv(
        paths["history"] / "selection_history.csv",
        index=False,
    )
    final_config = {
        **contract,
        "contract_sha256": payload_digest(contract),
        "epochs_completed": int(len(history)),
        "selected_epoch": int(selected_row["epoch"]) + 1,
        "best_val_loss": float(selected_row["val_loss"]),
        "train_recoverable_samples": int(len(train_inputs)),
        "validation_recoverable_samples": int(len(validation_inputs)),
        "validation_diagnostic": snapshot,
        "int8_calibration_counts": [int(value) for value in calibration_counts],
        "int8_tflite_size_bytes": int(int8_tflite_size_bytes),
        "output_semantics": "decision_scores",
        "op_evaluation": "separate_pass_required",
    }
    write_json_atomic(paths["args"] / "mobilenet_config.json", final_config)
    (paths["args"] / "model_stats.txt").write_text(
        "\n".join(
            (
                f"policy={policy}",
                f"budget={config.budget}",
                f"train_seed={train_seed}",
                f"parameters={protocol_model.count_params()}",
                f"epochs_completed={len(history)}",
                f"selected_epoch={int(selected_row['epoch']) + 1}",
                f"best_val_loss={float(selected_row['val_loss'])}",
                f"int8_tflite_size_bytes={int8_tflite_size_bytes}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in required_case_artifacts(run_dir)
    ):
        raise ValueError(f"Final artifact set is incomplete in {run_dir}.")
    write_json_atomic(
        run_dir / "completion.json",
        {
            "status": "complete",
            "contract_sha256": payload_digest(contract),
        },
    )
    print(f"[DONE] Completed run: {run_dir}")


def run_policy_budget_campaign(policy: str, budget: int) -> None:
    """Run all five training seeds for one policy-budget script.

    Args:
        policy (str): One of ``uni``, ``mar``, or ``gc``. It changes only the
            fixed observed-port pattern, not the architecture or loss.
        budget (int): Observation budget from 3 through 7. It selects both the
            rank-1 NAS architecture and the number of observed inputs.

    Returns:
        None: Five seed-specific runs are completed or skipped.

    Raises:
        ValueError: If the requested case or stored state is invalid.
        FileNotFoundError: If data or the selected source model is missing.
        Exception: Propagates training failures without continuing to later
            seeds in this script.

    Examples:
        Run the UNI three-port campaign from its dedicated script:

        >>> callable(run_policy_budget_campaign)
        True
    """
    validate_policy_budget(policy, budget)
    config = TOP_MODEL_CONFIGS[budget]
    (
        y_train,
        y_validation,
        y_test,
        train_parts,
        validation_parts,
        test_parts,
    ) = split_kms_60_20_20_cache(
        common_path=None,
        seed=DATA_SEED,
        regime_labels=KMS_REGIME_LABELS,
        samples_per_regime=None,
        verbose=True,
    )
    del y_test
    del test_parts
    manifest = load_or_create_pattern_manifest(train_parts)
    observed_indices = get_pattern_indices(manifest, policy, budget)
    del train_parts

    train_inputs, train_targets = build_training_data_for_indices(
        y_train,
        observed_indices,
    )
    calibration_data, calibration_counts = build_int8_calibration_data(
        y_train,
        observed_indices,
    )
    validation_inputs, validation_targets = build_training_data_for_indices(
        y_validation,
        observed_indices,
    )
    diagnostic_validation = np.concatenate(
        [part[: min(5_000, len(part))] for part in validation_parts],
        axis=0,
    ).astype(np.float32)
    del validation_parts
    scale_stride = max(1, len(y_train) // 100_000)
    log_scale = float(np.quantile(np.log1p(y_train[::scale_stride]), 0.999))
    validate_architecture_parity(config, observed_indices, log_scale)

    try:
        for train_seed in TRAIN_SEEDS:
            train_one_case(
                policy=policy,
                config=config,
                observed_indices=observed_indices,
                log_scale=log_scale,
                train_seed=train_seed,
                train_inputs=train_inputs,
                train_targets=train_targets,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                y_validation=diagnostic_validation,
                calibration_data=calibration_data,
                calibration_counts=calibration_counts,
            )
            tf.keras.backend.clear_session()
            gc.collect()
    finally:
        del y_train
        del y_validation
        del train_inputs
        del train_targets
        del calibration_data
        del validation_inputs
        del validation_targets
        del diagnostic_validation
        gc.collect()
