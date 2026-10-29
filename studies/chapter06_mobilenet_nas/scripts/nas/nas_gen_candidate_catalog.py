"""
Generate MobileNetV1 INT8 Candidate Catalog.

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev

This script builds an architecture feasibility catalog for MobileNetV1-based
safe port selectors. It does not train any model.

For each observed-port budget, alpha value, and MobileNetV1 block-level
truncation point, the script:

1. Builds the full untrained candidate model.
2. Converts it to strict INT8 TFLite using training-only calibration data.
3. Estimates peak live TFLite tensor RAM from the interpreter op schedule.
4. Writes one row to a full catalog CSV.
5. Writes a filtered feasible catalog CSV for candidates under MAX_RAM_BYTES.

No argparse is used. Configure the constants below and call
generate_mobilenetv1_int8_candidate_catalog() from your existing runner,
interactive session, or notebook.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Sys Path Configuration                            #
# ———————————————————————————————————————————————————————————————————————————— #
import sys
from pathlib import Path

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/nas/nas_gen_candidate_catalog.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Setup and Configuration                           #
# ———————————————————————————————————————————————————————————————————————————— #
import csv
import gc
import json
import os
import traceback
from collections.abc import Callable, Iterable
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from mobilenet.keras.sparse_input import build_sparse_port_images
from studies.chapter06_mobilenet_nas.scripts.nas.safe_selector_nas import (
    build_nas_selector_models,
)
from utils.data import (
    KMS_REGIME_LABELS,
    get_observed_ports,
    split_kms_60_20_20_cache,
)

tf.get_logger().setLevel("ERROR")

# %%—————————————————————————————————————————————————————————————————————————— #
#                                Run Parameters                                #
# ———————————————————————————————————————————————————————————————————————————— #
TOTAL_FEATURES = 100
OBSERVED_PORTS_LIST = (3, 4, 5, 6, 7)

# Use KMS_REGIME_LABELS for all-regime catalog.
# Use a single label tuple, e.g. ("kappa0_mu1_m0",), for per-regime catalog.
TRAIN_REGIME_LABELS = KMS_REGIME_LABELS

DATA_SEED = 0
INT8_CALIBRATION_SAMPLE_COUNT = 256
INT8_CALIBRATION_NUM_BINS = 16
INT8_CALIBRATION_SEED = 0

IMAGE_SIDE = 32
IMAGE_CHANNELS = 3

ALPHA_MIN = 0.05
ALPHA_MAX = 1.00
ALPHA_STEP = 0.05

MOBILENETV1_TRUNCATION_BLOCKS = (
    "conv_pw_1_relu",
    "conv_pw_2_relu",
    "conv_pw_3_relu",
    "conv_pw_4_relu",
    "conv_pw_5_relu",
    "conv_pw_6_relu",
    "conv_pw_7_relu",
    "conv_pw_8_relu",
    "conv_pw_9_relu",
    "conv_pw_10_relu",
    "conv_pw_11_relu",
    "conv_pw_12_relu",
    "conv_pw_13_relu",
)

# Conservative screening head. NAS can still sample smaller heads later.
SCREEN_HEAD_DEPTH = 3
SCREEN_HEAD_WIDTH = 256

MAX_RAM_BYTES = 160 * 1024

RUN_DIR = Path("runs/mobilenetv1_int8_candidate_catalog")
FULL_CATALOG_CSV = RUN_DIR / "mobilenetv1_int8_candidate_catalog.csv"
FEASIBLE_CATALOG_CSV = RUN_DIR / "mobilenetv1_int8_feasible_candidate_catalog.csv"

REBUILD_CATALOG = True


# %%—————————————————————————————————————————————————————————————————————————— #
#                               Logging Utilities                              #
# ———————————————————————————————————————————————————————————————————————————— #
def log_print(function_name: str, message: str) -> None:
    """Print a standardized log line.

    Args:
        function_name (str): Function or block name emitting the message.
        message (str): Human-readable message body.

    Returns:
        None: Writes one formatted line to stdout.

    Raises:
        OSError: If stdout cannot be written.

    Examples:
        >>> log_print("catalog", "Starting candidate generation")
    """
    print(f"[{_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/nas/nas_gen_candidate_catalog.py'.stem}:{function_name}] {message}")


# %%—————————————————————————————————————————————————————————————————————————— #
#                              Calibration Helpers                             #
# ———————————————————————————————————————————————————————————————————————————— #
def build_alpha_values(
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
) -> tuple[float, ...]:
    """Build a stable rounded alpha grid.

    Args:
        alpha_min (float): First alpha value.
        alpha_max (float): Last alpha value, inclusive within rounding.
        alpha_step (float): Step between alpha values.

    Returns:
        tuple[float, ...]: Rounded alpha values.

    Raises:
        ValueError: If alpha bounds or step are invalid.

    Examples:
        >>> build_alpha_values(0.05, 0.15, 0.05)
        (0.05, 0.1, 0.15)
    """
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive.")
    if alpha_max < alpha_min:
        raise ValueError("alpha_max must be greater than or equal to alpha_min.")
    if alpha_step <= 0.0:
        raise ValueError("alpha_step must be positive.")

    n_steps = int(round((alpha_max - alpha_min) / alpha_step)) + 1
    return tuple(round(alpha_min + index * alpha_step, 2) for index in range(n_steps))


def allocate_group_sample_counts(
    total_sample_count: int,
    n_groups: int,
) -> tuple[int, ...]:
    """Allocate a total calibration budget across groups.

    Args:
        total_sample_count (int): Total number of samples to allocate.
        n_groups (int): Number of groups, such as KMS regimes.

    Returns:
        tuple[int, ...]: Per-group sample counts whose sum is
            ``total_sample_count``.

    Raises:
        ValueError: If ``total_sample_count`` or ``n_groups`` is invalid.

    Examples:
        >>> allocate_group_sample_counts(5, 2)
        (3, 2)
    """
    if total_sample_count < 1:
        raise ValueError("total_sample_count must be at least 1.")
    if n_groups < 1:
        raise ValueError("n_groups must be at least 1.")

    base_count = total_sample_count // n_groups
    remainder = total_sample_count % n_groups

    counts = []
    for group_index in range(n_groups):
        count = base_count + (1 if group_index < remainder else 0)
        counts.append(count)

    return tuple(counts)


def select_quantile_balanced_indices(
    y_values: np.ndarray,
    sample_count: int,
    num_bins: int,
    seed: int,
) -> np.ndarray:
    """Select row indices balanced across within-regime SNR quantiles.

    Args:
        y_values (np.ndarray): Full-port target matrix with shape
            ``(n_samples, total_features)``.
        sample_count (int): Target number of selected rows.
        num_bins (int): Number of quantile bins.
        seed (int): Random seed used for reproducible selection.

    Returns:
        np.ndarray: Selected row indices with dtype ``int64``.

    Raises:
        ValueError: If ``y_values`` is not a non-empty 2D matrix, or if
            ``sample_count`` or ``num_bins`` is invalid.

    Examples:
        >>> y = np.arange(1000, dtype=np.float32).reshape(10, 100)
        >>> idx = select_quantile_balanced_indices(y, 5, 2, 0)
        >>> idx.shape[0]
        5
    """
    if y_values.ndim != 2:
        raise ValueError("y_values must be a 2D matrix.")
    if y_values.shape[0] == 0:
        raise ValueError("y_values must contain at least one sample.")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    rng = np.random.default_rng(seed)
    n_samples = y_values.shape[0]
    target_count = min(sample_count, n_samples)

    selection_score = np.max(y_values, axis=1)
    quantile_edges = np.quantile(
        selection_score,
        np.linspace(0.0, 1.0, num_bins + 1),
    )

    selected_indices = []
    base_per_bin = max(target_count // num_bins, 1)

    for bin_index in range(num_bins):
        left_edge = quantile_edges[bin_index]
        right_edge = quantile_edges[bin_index + 1]

        if bin_index == num_bins - 1:
            bin_mask = (selection_score >= left_edge) & (selection_score <= right_edge)
        else:
            bin_mask = (selection_score >= left_edge) & (selection_score < right_edge)

        bin_indices = np.flatnonzero(bin_mask)
        if bin_indices.size == 0:
            continue

        n_select = min(base_per_bin, bin_indices.size)
        selected = rng.choice(bin_indices, size=n_select, replace=False)
        selected_indices.extend(selected.tolist())

    selected_set = set(selected_indices)
    remaining_slots = target_count - len(selected_indices)

    if remaining_slots > 0:
        remaining_indices = np.array(
            [index for index in range(n_samples) if index not in selected_set],
            dtype=np.int64,
        )
        if remaining_indices.size > 0:
            n_fill = min(remaining_slots, remaining_indices.size)
            fill_indices = rng.choice(remaining_indices, size=n_fill, replace=False)
            selected_indices.extend(fill_indices.tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices)

    return selected_indices


def build_catalog_calibration_data(
    train_parts: tuple[np.ndarray, ...],
    regime_labels: tuple[str, ...],
    num_observed_ports: int,
    total_features: int,
    total_calibration_samples: int,
    num_bins: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build calibration data for catalog INT8 conversion.

    For all-regime studies, this function allocates the total calibration
    budget across regimes. Within each regime, it samples rows balanced across
    quantiles of ``max(y_train_part, axis=1)``.

    For per-regime studies, pass one regime part and one regime label.

    Args:
        train_parts (tuple[np.ndarray, ...]): Per-regime training matrices.
        regime_labels (tuple[str, ...]): Labels aligned with ``train_parts``.
        num_observed_ports (int): Number of observed ports used as model input.
        total_features (int): Total number of full-port target features.
        total_calibration_samples (int): Total calibration budget across all
            selected regimes.
        num_bins (int): Number of quantile bins per regime.
        seed (int): Random seed for reproducible sampling.

    Returns:
        tuple[np.ndarray, dict[str, int]]: Calibration matrix and per-regime
            selected sample counts.

    Raises:
        ValueError: If inputs are inconsistent or empty.

    Examples:
        >>> calibration, counts = build_catalog_calibration_data(
        ...     train_parts=train_parts,
        ...     regime_labels=KMS_REGIME_LABELS,
        ...     num_observed_ports=3,
        ...     total_features=100,
        ...     total_calibration_samples=256,
        ...     num_bins=16,
        ...     seed=0,
        ... )
        >>> calibration.shape[1]
        3
    """
    if len(train_parts) == 0:
        raise ValueError("train_parts must contain at least one part.")
    if len(train_parts) != len(regime_labels):
        raise ValueError(
            "train_parts and regime_labels must have the same length. "
            f"Got {len(train_parts)} and {len(regime_labels)}."
        )

    sample_counts = allocate_group_sample_counts(
        total_sample_count=total_calibration_samples,
        n_groups=len(train_parts),
    )

    calibration_parts = []
    calibration_counts = {}

    for part_index, (regime_label, train_part, part_sample_count) in enumerate(
        zip(regime_labels, train_parts, sample_counts, strict=True)
    ):
        if train_part.ndim != 2:
            raise ValueError(f"Train part for {regime_label} must be 2D.")
        if train_part.shape[1] != total_features:
            raise ValueError(
                f"Train part for {regime_label} must have {total_features} columns. "
                f"Got {train_part.shape[1]}."
            )

        observed_part, observed_indices = get_observed_ports(
            train_part,
            num_observed_ports,
            total_features,
        )
        selected_indices = select_quantile_balanced_indices(
            y_values=train_part,
            sample_count=part_sample_count,
            num_bins=num_bins,
            seed=seed + part_index,
        )

        # Select compact rows before sparse projection so calibration memory is
        # bounded by the representative budget, not the full training part.
        calibration_parts.append(
            build_sparse_port_images(
                observed_values=observed_part[selected_indices],
                observed_indices=observed_indices.astype(int),
                total_ports=total_features,
                image_height=IMAGE_SIDE,
                image_width=IMAGE_SIDE,
            )
        )
        calibration_counts[regime_label] = int(selected_indices.shape[0])

    calibration_data = np.concatenate(calibration_parts, axis=0).astype(np.float32)
    return calibration_data, calibration_counts


def make_representative_dataset(
    calibration_data: np.ndarray,
) -> Callable[[], Iterable[list[np.ndarray]]]:
    """Create a TFLite representative dataset generator.

    Args:
        calibration_data (np.ndarray): Scaled sparse calibration image batch
            with shape ``(n_samples, 32, 32, 3)``.

    Returns:
        Callable[[], Iterable[list[np.ndarray]]]: Representative dataset
            generator factory compatible with TFLite conversion.

    Raises:
        ValueError: If ``calibration_data`` is not a non-empty 4D image batch.

    Examples:
        >>> representative_dataset = make_representative_dataset(calibration_data)
        >>> callable(representative_dataset)
        True
    """
    if calibration_data.ndim != 4:
        raise ValueError("calibration_data must be a 4D sparse image batch.")
    if calibration_data.shape[0] == 0:
        raise ValueError("calibration_data must contain at least one sample.")

    calibration = calibration_data.astype(np.float32, copy=False)

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        """Yield one-sample batches for strict INT8 calibration.

        Args:
            None.

        Yields:
            list[np.ndarray]: One raw sparse input with batch dimension.

        Raises:
            None.
        """
        for sample in calibration:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    return representative_dataset


# %%—————————————————————————————————————————————————————————————————————————— #
#                                Model Helpers                                 #
# ———————————————————————————————————————————————————————————————————————————— #
def build_mobilenetv1_candidate_model(
    num_observed_ports: int,
    total_features: int,
    alpha: float,
    image_side: int,
    truncation_block_name: str,
    head_depth: int,
    head_width: int,
    observed_indices: np.ndarray,
    log_scale: float,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Build a GroupNorm safe-selector candidate for catalog screening.

    Args:
        num_observed_ports (int): Number of observed FAS ports used as input.
        total_features (int): Number of output decision scores.
        alpha (float): MobileNetV1 width multiplier.
        image_side (int): Side length of the sparse square image.
        truncation_block_name (str): MobileNetV1 block output layer name used
            as the truncation point.
        head_depth (int): Number of dense hidden layers after global pooling.
        head_width (int): Number of units in each dense hidden layer.
        observed_indices (np.ndarray): Physical indices of compact inputs.
        log_scale (float): Shared internal `log1p` scale.

    Returns:
        tuple[tf.keras.Model, dict[str, Any]]: Candidate model and metadata.

    Raises:
        ValueError: If parameters are invalid or the truncation layer does not
            exist in the MobileNetV1 instance.

    """
    if num_observed_ports < 1:
        raise ValueError("num_observed_ports must be at least 1.")
    if total_features < 1:
        raise ValueError("total_features must be at least 1.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if image_side < 32:
        raise ValueError("image_side must be at least 32 for MobileNetV1.")
    if head_depth < 1:
        raise ValueError("head_depth must be positive.")
    if head_width < 1:
        raise ValueError("head_width must be at least 1.")

    explorer_model, _, protocol_model = build_nas_selector_models(
        observed_indices=observed_indices,
        log_scale=log_scale,
        alpha=alpha,
        truncation_block_name=truncation_block_name,
        head_depth=head_depth,
        head_width=head_width,
        train_seed=DATA_SEED,
    )

    metadata = {
        "truncation_layer_name": truncation_block_name,
        "truncated_output_shape": list(
            explorer_model.get_layer("global_average_pool").input.shape
        ),
        "model_input_shape": list(protocol_model.input_shape),
        "model_output_shape": list(protocol_model.output_shape),
    }
    return protocol_model, metadata


def convert_model_to_strict_int8_tflite(
    model: tf.keras.Model,
    calibration_data: np.ndarray,
) -> bytes:
    """Convert a Keras model to strict INT8 TFLite.

    Args:
        model (tf.keras.Model): Keras model to convert.
        calibration_data (np.ndarray): Scaled representative input matrix.

    Returns:
        bytes: Serialized INT8 TFLite flatbuffer.

    Raises:
        RuntimeError: If conversion fails.
        ValueError: If calibration data is invalid.

    Examples:
        >>> tflite_model = convert_model_to_strict_int8_tflite(model, calibration_data)
        >>> isinstance(tflite_model, bytes)
        True
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(calibration_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter.convert()
    except Exception as exc:
        raise RuntimeError("Strict INT8 TFLite conversion failed.") from exc


def assert_strict_int8_io(
    tflite_model: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Verify that a TFLite model uses INT8 input and output tensors.

    Args:
        tflite_model (bytes): Serialized TFLite model.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: Input and output tensor shapes.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If the model input or output dtype is not INT8.

    Examples:
        >>> input_shape, output_shape = assert_strict_int8_io(tflite_model)
        >>> isinstance(input_shape, tuple)
        True
    """
    if not tflite_model:
        raise ValueError("tflite_model must not be empty.")

    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] != np.int8:
        raise RuntimeError(f"Expected INT8 input, got {input_detail['dtype']}.")
    if output_detail["dtype"] != np.int8:
        raise RuntimeError(f"Expected INT8 output, got {output_detail['dtype']}.")

    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])
    return input_shape, output_shape


def estimate_tflite_peak_ram_bytes(
    tflite_model: bytes,
) -> int:
    """Estimate peak live RAM from a TFLite execution schedule.

    Args:
        tflite_model (bytes): Serialized TFLite model.

    Returns:
        int: Estimated peak live tensor memory in bytes.

    Raises:
        ValueError: If ``tflite_model`` is empty.
        RuntimeError: If the interpreter does not expose operator details.

    Examples:
        >>> peak_ram = estimate_tflite_peak_ram_bytes(tflite_model)
        >>> peak_ram > 0
        True
    """
    if not tflite_model:
        raise ValueError("tflite_model must not be empty.")

    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()

    tensor_size_by_index = {}
    for detail in interpreter.get_tensor_details():
        shape = np.asarray(detail.get("shape", []), dtype=np.int64)
        safe_shape = np.where(shape < 0, 0, shape)
        element_count = int(np.prod(safe_shape, dtype=np.int64)) if safe_shape.size else 0

        if element_count <= 0:
            tensor_size_by_index[int(detail["index"])] = 0
            continue

        dtype = np.dtype(detail["dtype"])
        tensor_size_by_index[int(detail["index"])] = element_count * int(dtype.itemsize)

    if not hasattr(interpreter, "_get_ops_details"):
        raise RuntimeError("TFLite interpreter does not expose operator schedule details.")

    peak_bytes = 0
    for op_detail in interpreter._get_ops_details():
        input_indices = [int(index) for index in op_detail.get("inputs", []) if int(index) >= 0]
        output_indices = [int(index) for index in op_detail.get("outputs", []) if int(index) >= 0]
        touched_indices = set(input_indices + output_indices)
        live_bytes = sum(tensor_size_by_index.get(index, 0) for index in touched_indices)
        peak_bytes = max(peak_bytes, int(live_bytes))

    return int(peak_bytes)


# %%—————————————————————————————————————————————————————————————————————————— #
#                                CSV Utilities                                 #
# ———————————————————————————————————————————————————————————————————————————— #
def get_catalog_columns() -> tuple[str, ...]:
    """Return the canonical candidate catalog column order.

    Args:
        None.

    Returns:
        tuple[str, ...]: CSV column names.

    Raises:
        None.

    Examples:
        >>> "candidate_id" in get_catalog_columns()
        True
    """
    return (
        "candidate_id",
        "observed_ports_budget",
        "alpha",
        "image_side",
        "truncation_block_name",
        "truncation_layer_name",
        "head_depth_for_screening",
        "head_width_for_screening",
        "projection_units",
        "truncated_output_shape",
        "model_input_shape",
        "model_output_shape",
        "num_params",
        "tflite_input_shape",
        "tflite_output_shape",
        "tflite_int8_model_size_bytes",
        "tflite_int8_peak_ram_bytes",
        "max_ram_bytes",
        "fits_ram_budget",
        "calibration_sample_count",
        "calibration_counts_by_regime",
        "status",
        "failure_reason",
    )


def initialize_catalog_csv(
    csv_path: Path,
    rebuild: bool,
) -> None:
    """Initialize a candidate catalog CSV.

    Args:
        csv_path (Path): Destination CSV path.
        rebuild (bool): If ``True``, an existing CSV is deleted and recreated.

    Returns:
        None: Creates parent directories and writes the CSV header if needed.

    Raises:
        OSError: If the CSV path cannot be created or written.

    Examples:
        >>> initialize_catalog_csv(Path("runs/demo/catalog.csv"), True)
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild and csv_path.exists():
        csv_path.unlink()

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=get_catalog_columns())
            writer.writeheader()


def append_catalog_row(
    csv_path: Path,
    row: dict[str, Any],
) -> None:
    """Append one row to the candidate catalog CSV.

    Args:
        csv_path (Path): Destination CSV path.
        row (dict[str, Any]): Candidate catalog row.

    Returns:
        None: Appends one CSV row.

    Raises:
        OSError: If the CSV cannot be opened or written.

    Examples:
        >>> append_catalog_row(Path("runs/demo/catalog.csv"), {"candidate_id": "demo"})
    """
    columns = get_catalog_columns()
    normalized_row = {column: row.get(column, "") for column in columns}

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writerow(normalized_row)


def write_feasible_catalog(
    full_catalog_csv: Path,
    feasible_catalog_csv: Path,
) -> pd.DataFrame:
    """Write a filtered feasible candidate catalog.

    Args:
        full_catalog_csv (Path): Full candidate catalog CSV path.
        feasible_catalog_csv (Path): Output path for feasible candidates.

    Returns:
        pd.DataFrame: Filtered feasible candidate table.

    Raises:
        FileNotFoundError: If ``full_catalog_csv`` does not exist.
        OSError: If the feasible CSV cannot be written.

    Examples:
        >>> feasible_df = write_feasible_catalog(full_csv, feasible_csv)
        >>> isinstance(feasible_df, pd.DataFrame)
        True
    """
    if not full_catalog_csv.exists():
        raise FileNotFoundError(f"Full catalog CSV not found: {full_catalog_csv}")

    catalog_df = pd.read_csv(full_catalog_csv)
    feasible_df = catalog_df[
        (catalog_df["status"] == "complete") &
        (catalog_df["fits_ram_budget"] == True)
    ].copy()

    feasible_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
    feasible_df.to_csv(feasible_catalog_csv, index=False)
    return feasible_df


# %%—————————————————————————————————————————————————————————————————————————— #
#                             Catalog Generation                               #
# ———————————————————————————————————————————————————————————————————————————— #
def prepare_budget_calibration_data(
    y_train: np.ndarray,
    train_parts: tuple[np.ndarray, ...],
    regime_labels: tuple[str, ...],
    num_observed_ports: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Prepare raw calibration data and selector metadata for one budget.

    Args:
        y_train (np.ndarray): Concatenated training target matrix.
        train_parts (tuple[np.ndarray, ...]): Per-regime training matrices.
        regime_labels (tuple[str, ...]): Regime labels aligned with
            ``train_parts``.
        num_observed_ports (int): Number of observed ports for this catalog
            slice.

    Returns:
        tuple[np.ndarray, dict[str, int], dict[str, Any]]: Calibration matrix,
            counts by regime, and fixed selector metadata.

    Raises:
        ValueError: If observed-port extraction or scaling inputs are invalid.

    """
    _, observed_indices = get_observed_ports(
        y_train,
        num_observed_ports,
        TOTAL_FEATURES,
    )
    scale_stride = max(1, len(y_train) // 100_000)
    log_scale = float(
        np.quantile(np.log1p(y_train[::scale_stride]), 0.999)
    )

    calibration_data, calibration_counts = build_catalog_calibration_data(
        train_parts=train_parts,
        regime_labels=regime_labels,
        num_observed_ports=num_observed_ports,
        total_features=TOTAL_FEATURES,
        total_calibration_samples=INT8_CALIBRATION_SAMPLE_COUNT,
        num_bins=INT8_CALIBRATION_NUM_BINS,
        seed=INT8_CALIBRATION_SEED,
    )

    selector_metadata = {
        "observed_indices": observed_indices.astype(np.int64),
        "log_scale": log_scale,
    }
    return calibration_data, calibration_counts, selector_metadata


def build_candidate_id(
    num_observed_ports: int,
    alpha: float,
    image_side: int,
    truncation_block_name: str,
    head_depth: int,
    head_width: int,
) -> str:
    """Build a stable candidate identifier.

    Args:
        num_observed_ports (int): Observed-port budget.
        alpha (float): MobileNetV1 width multiplier.
        image_side (int): Latent image side length.
        truncation_block_name (str): MobileNetV1 truncation block name.
        head_depth (int): Screening head depth.
        head_width (int): Screening head width.

    Returns:
        str: Stable candidate identifier.

    Raises:
        None.

    Examples:
        >>> build_candidate_id(3, 0.1, 32, "conv_pw_3_relu", 3, 256)
        'n03_alpha0p10_img32_conv_pw_3_relu_h3_w256'
    """
    alpha_key = f"{alpha:.2f}".replace(".", "p")
    return (
        f"n{num_observed_ports:02d}_"
        f"alpha{alpha_key}_"
        f"img{image_side}_"
        f"{truncation_block_name}_"
        f"h{head_depth}_"
        f"w{head_width}"
    )


def evaluate_catalog_candidate(
    num_observed_ports: int,
    alpha: float,
    image_side: int,
    truncation_block_name: str,
    calibration_data: np.ndarray,
    calibration_counts: dict[str, int],
    selector_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build, quantize, and measure one MobileNetV1 catalog candidate.

    Args:
        num_observed_ports (int): Observed-port budget.
        alpha (float): MobileNetV1 width multiplier.
        image_side (int): Latent image side length.
        truncation_block_name (str): Block-level MobileNetV1 truncation point.
        calibration_data (np.ndarray): Raw training-only calibration images.
        calibration_counts (dict[str, int]): Calibration counts by regime.
        selector_metadata (dict[str, Any]): Observed/candidate indices, rates,
            and shared normalization scale.

    Returns:
        dict[str, Any]: Candidate catalog row.

    Raises:
        None: Conversion/build failures are caught and encoded in the returned
            row as ``status="failed"``.

    """
    candidate_id = build_candidate_id(
        num_observed_ports=num_observed_ports,
        alpha=alpha,
        image_side=image_side,
        truncation_block_name=truncation_block_name,
        head_depth=SCREEN_HEAD_DEPTH,
        head_width=SCREEN_HEAD_WIDTH,
    )

    base_row = {
        "candidate_id": candidate_id,
        "observed_ports_budget": int(num_observed_ports),
        "alpha": float(alpha),
        "image_side": int(image_side),
        "truncation_block_name": truncation_block_name,
        "truncation_layer_name": "",
        "head_depth_for_screening": int(SCREEN_HEAD_DEPTH),
        "head_width_for_screening": int(SCREEN_HEAD_WIDTH),
        "projection_units": int(image_side * image_side * IMAGE_CHANNELS),
        "truncated_output_shape": "",
        "model_input_shape": "",
        "model_output_shape": "",
        "num_params": "",
        "tflite_input_shape": "",
        "tflite_output_shape": "",
        "tflite_int8_model_size_bytes": "",
        "tflite_int8_peak_ram_bytes": "",
        "max_ram_bytes": int(MAX_RAM_BYTES),
        "fits_ram_budget": False,
        "calibration_sample_count": int(calibration_data.shape[0]),
        "calibration_counts_by_regime": json.dumps(calibration_counts, sort_keys=True),
        "status": "failed",
        "failure_reason": "",
    }

    try:
        tf.keras.backend.clear_session()

        model, metadata = build_mobilenetv1_candidate_model(
            num_observed_ports=num_observed_ports,
            total_features=TOTAL_FEATURES,
            alpha=alpha,
            image_side=image_side,
            truncation_block_name=truncation_block_name,
            head_depth=SCREEN_HEAD_DEPTH,
            head_width=SCREEN_HEAD_WIDTH,
            observed_indices=selector_metadata["observed_indices"],
            log_scale=float(selector_metadata["log_scale"]),
        )

        tflite_model = convert_model_to_strict_int8_tflite(
            model=model,
            calibration_data=calibration_data,
        )
        tflite_input_shape, tflite_output_shape = assert_strict_int8_io(tflite_model)
        peak_ram_bytes = estimate_tflite_peak_ram_bytes(tflite_model)
        model_size_bytes = len(tflite_model)
        fits_ram_budget = peak_ram_bytes <= MAX_RAM_BYTES

        base_row.update(
            {
                "truncation_layer_name": metadata["truncation_layer_name"],
                "truncated_output_shape": json.dumps(metadata["truncated_output_shape"]),
                "model_input_shape": json.dumps(metadata["model_input_shape"]),
                "model_output_shape": json.dumps(metadata["model_output_shape"]),
                "num_params": int(model.count_params()),
                "tflite_input_shape": json.dumps(tflite_input_shape),
                "tflite_output_shape": json.dumps(tflite_output_shape),
                "tflite_int8_model_size_bytes": int(model_size_bytes),
                "tflite_int8_peak_ram_bytes": int(peak_ram_bytes),
                "fits_ram_budget": bool(fits_ram_budget),
                "status": "complete",
                "failure_reason": "",
            }
        )
    except Exception as exc:
        base_row["status"] = "failed"
        base_row["failure_reason"] = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=5)}"
        )
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

    return base_row


def generate_mobilenetv1_int8_candidate_catalog() -> pd.DataFrame:
    """Generate the MobileNetV1 INT8 candidate catalog.

    Args:
        None.

    Returns:
        pd.DataFrame: Feasible candidate catalog table.

    Raises:
        FileNotFoundError: If the KMS runtime cache is missing.
        OSError: If catalog files cannot be written.

    Examples:
        >>> feasible_df = generate_mobilenetv1_int8_candidate_catalog()
        >>> isinstance(feasible_df, pd.DataFrame)
        True
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    initialize_catalog_csv(FULL_CATALOG_CSV, rebuild=REBUILD_CATALOG)

    alpha_values = build_alpha_values(
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        alpha_step=ALPHA_STEP,
    )

    log_print("catalog", f"Output directory: {RUN_DIR}")
    log_print("catalog", f"Full catalog CSV: {FULL_CATALOG_CSV}")
    log_print("catalog", f"Feasible catalog CSV: {FEASIBLE_CATALOG_CSV}")
    log_print("catalog", f"Alpha values: {alpha_values}")
    log_print("catalog", f"Observed-port budgets: {OBSERVED_PORTS_LIST}")
    log_print("catalog", f"Truncation blocks: {len(MOBILENETV1_TRUNCATION_BLOCKS)}")
    log_print("catalog", f"Regime labels: {TRAIN_REGIME_LABELS}")

    (
        y_train,
        y_val,
        y_test,
        train_parts,
        val_parts,
        test_parts,
    ) = split_kms_60_20_20_cache(
        seed=DATA_SEED,
        regime_labels=TRAIN_REGIME_LABELS,
        verbose=True,
    )
    del y_val
    del y_test
    del val_parts
    del test_parts

    total_candidates = (
        len(OBSERVED_PORTS_LIST)
        * len(alpha_values)
        * len(MOBILENETV1_TRUNCATION_BLOCKS)
    )
    processed_candidates = 0

    for num_observed_ports in OBSERVED_PORTS_LIST:
        log_print(
            "catalog",
            f"Preparing calibration for N={num_observed_ports}",
        )

        (
            calibration_data,
            calibration_counts,
            selector_metadata,
        ) = prepare_budget_calibration_data(
            y_train=y_train,
            train_parts=train_parts,
            regime_labels=TRAIN_REGIME_LABELS,
            num_observed_ports=num_observed_ports,
        )

        log_print(
            "catalog",
            (
                f"N={num_observed_ports}: calibration shape={calibration_data.shape}, "
                f"counts={calibration_counts}"
            ),
        )

        for alpha in alpha_values:
            for truncation_block_name in MOBILENETV1_TRUNCATION_BLOCKS:
                processed_candidates += 1
                log_print(
                    "catalog",
                    (
                        f"[{processed_candidates}/{total_candidates}] "
                        f"N={num_observed_ports}, alpha={alpha:.2f}, "
                        f"block={truncation_block_name}"
                    ),
                )

                row = evaluate_catalog_candidate(
                    num_observed_ports=num_observed_ports,
                    alpha=alpha,
                    image_side=IMAGE_SIDE,
                    truncation_block_name=truncation_block_name,
                    calibration_data=calibration_data,
                    calibration_counts=calibration_counts,
                    selector_metadata=selector_metadata,
                )
                append_catalog_row(FULL_CATALOG_CSV, row)

    feasible_df = write_feasible_catalog(
        full_catalog_csv=FULL_CATALOG_CSV,
        feasible_catalog_csv=FEASIBLE_CATALOG_CSV,
    )

    log_print("catalog", f"Feasible candidates: {len(feasible_df)}")
    log_print("catalog", "Catalog generation finished.")

    return feasible_df


if __name__ == "__main__":
    generate_mobilenetv1_int8_candidate_catalog()
