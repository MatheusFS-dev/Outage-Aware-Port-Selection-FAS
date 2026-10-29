"""Generate untrained seven-port MobileNet TFLite benchmark artifacts."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import gc
import sys
from pathlib import Path

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/generate_7_port_benchmark_models.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from mobilenet.keras.sparse_input import build_sparse_port_images
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    IMAGE_SIDE,
    TOTAL_PORTS,
    VARIANT_METADATA,
    build_models,
    build_protocol_decision_model,
)
from utils.data import KMS_REGIME_LABELS, get_observed_ports, split_kms_60_20_20_cache
from utils.qttf import (
    build_quantile_balanced_calibration_data,
    convert_ptq_model_to_strict_int8_tflite,
    test_strict_int8_io,
)


NUM_OBSERVED_PORTS = 7
BENCHMARK_BATCH_SIZE = 1
MODEL_SEED = 0
CALIBRATION_SAMPLE_COUNT = 256
CALIBRATION_NUM_BINS = 16
CALIBRATION_SEED = 0
OUTPUT_DIR = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/generated_7_port_benchmark_models'


def replace_int8_log_with_float(tflite_model: bytes) -> bytes:
    """Replace the benchmark model's INT8 LOG with a float TFLM-compatible path.

    Args:
        tflite_model (bytes): Strict-INT8 TFLite flatbuffer containing exactly
            one single-input, single-output INT8 ``LOG`` operator in its only
            subgraph.

    Returns:
        bytes: Equivalent TFLite flatbuffer with
        ``DEQUANTIZE -> float LOG -> QUANTIZE`` replacing the original LOG.
        Public input and output tensors remain INT8.

    Raises:
        RuntimeError: If the flatbuffer does not have the expected one-subgraph,
            one-LOG deployment structure or lacks a required opcode.

    Examples:
        >>> callable(replace_int8_log_with_float)
        True
    """
    model = schema.Model.GetRootAsModel(tflite_model, 0)
    model_object = schema.ModelT.InitFromObj(model)
    if len(model_object.subgraphs) != 1:
        raise RuntimeError("Expected exactly one TFLite subgraph.")

    subgraph = model_object.subgraphs[0]
    opcode_indices = {
        code.builtinCode: index
        for index, code in enumerate(model_object.operatorCodes)
    }
    required_opcodes = (
        schema.BuiltinOperator.DEQUANTIZE,
        schema.BuiltinOperator.LOG,
        schema.BuiltinOperator.QUANTIZE,
    )
    missing_opcodes = [
        opcode for opcode in required_opcodes if opcode not in opcode_indices
    ]
    if missing_opcodes:
        raise RuntimeError(f"Missing required TFLite opcodes: {missing_opcodes}.")

    log_indices = [
        index
        for index, operator in enumerate(subgraph.operators)
        if (
            model_object.operatorCodes[operator.opcodeIndex].builtinCode
            == schema.BuiltinOperator.LOG
        )
    ]
    if len(log_indices) != 1:
        raise RuntimeError(f"Expected one LOG operator, found {len(log_indices)}.")

    log_index = log_indices[0]
    log_operator = subgraph.operators[log_index]
    if len(log_operator.inputs) != 1 or len(log_operator.outputs) != 1:
        raise RuntimeError("Expected LOG to have one input and one output.")

    input_index = int(log_operator.inputs[0])
    output_index = int(log_operator.outputs[0])
    input_tensor = subgraph.tensors[input_index]
    output_tensor = subgraph.tensors[output_index]
    if (
        input_tensor.type != schema.TensorType.INT8
        or output_tensor.type != schema.TensorType.INT8
    ):
        raise RuntimeError("Expected LOG to have INT8 input and output tensors.")
    if list(input_tensor.shape) != list(output_tensor.shape):
        raise RuntimeError("Expected LOG input and output shapes to match.")

    float_input = schema.TensorT()
    float_input.shape = list(input_tensor.shape)
    float_input.shapeSignature = list(input_tensor.shape)
    float_input.type = schema.TensorType.FLOAT32
    float_input.buffer = 0
    float_input.name = "tflm_log_float_input"
    float_input.isVariable = False
    subgraph.tensors.append(float_input)
    float_input_index = len(subgraph.tensors) - 1

    float_output = schema.TensorT()
    float_output.shape = list(output_tensor.shape)
    float_output.shapeSignature = list(output_tensor.shape)
    float_output.type = schema.TensorType.FLOAT32
    float_output.buffer = 0
    float_output.name = "tflm_log_float_output"
    float_output.isVariable = False
    subgraph.tensors.append(float_output)
    float_output_index = len(subgraph.tensors) - 1

    dequantize = schema.OperatorT()
    dequantize.opcodeIndex = opcode_indices[schema.BuiltinOperator.DEQUANTIZE]
    dequantize.inputs = np.asarray([input_index], dtype=np.int32)
    dequantize.outputs = np.asarray([float_input_index], dtype=np.int32)

    log_operator.inputs = np.asarray([float_input_index], dtype=np.int32)
    log_operator.outputs = np.asarray([float_output_index], dtype=np.int32)

    quantize = schema.OperatorT()
    quantize.opcodeIndex = opcode_indices[schema.BuiltinOperator.QUANTIZE]
    quantize.inputs = np.asarray([float_output_index], dtype=np.int32)
    quantize.outputs = np.asarray([output_index], dtype=np.int32)

    subgraph.operators = (
        subgraph.operators[:log_index]
        + [dequantize, log_operator, quantize]
        + subgraph.operators[log_index + 1 :]
    )
    builder = flatbuffers.Builder(0)
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def build_calibration_images(
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build deterministic train-only INT8 calibration images for seven ports.

    Args:
        y_train (np.ndarray): Full 60% training split with shape ``(n, 100)``.
            It must contain at least one row and no validation or test samples.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: A tuple containing representative
        sparse images with shape ``(m, 32, 32, 3)``, the seven zero-based
        physical port indices used by every generated model, and the training
        normalization log scale used by every generated model.

    Raises:
        ValueError: If ``y_train`` is invalid or cannot supply calibration data.

    Examples:
        >>> calibration_images, observed_indices, _ = build_calibration_images(
        ...     np.ones((32, 100), dtype=np.float32)
        ... )
        >>> calibration_images.shape[1:]
        (32, 32, 3)
        >>> observed_indices.tolist()
        [0, 16, 33, 49, 66, 82, 99]
    """
    observed_values, observed_indices = get_observed_ports(
        y_train,
        NUM_OBSERVED_PORTS,
        TOTAL_PORTS,
    )
    scale_stride = max(1, len(y_train) // 100_000)
    log_scale = float(np.quantile(np.log1p(y_train[::scale_stride]), 0.999))
    calibration_values, _ = build_quantile_balanced_calibration_data(
        observed_values.astype(np.float32),
        y_train,
        sample_count=CALIBRATION_SAMPLE_COUNT,
        num_bins=CALIBRATION_NUM_BINS,
        seed=CALIBRATION_SEED,
    )
    calibration_images = build_sparse_port_images(
        observed_values=calibration_values,
        observed_indices=observed_indices,
        total_ports=TOTAL_PORTS,
        image_height=IMAGE_SIDE,
        image_width=IMAGE_SIDE,
    )
    return calibration_images, observed_indices, log_scale


def save_untrained_variant(
    variant_name: str,
    observed_indices: np.ndarray,
    calibration_images: np.ndarray,
    log_scale: float,
) -> Path:
    """Convert one untrained seven-port MobileNet protocol model to INT8 TFLite.

    Args:
        variant_name (str): Supported safe-selector variant name. Accepted
            values are ``mobv1_alpha02``, ``mobv2_alpha02``, and
            ``mobv3_alpha02``; each selects its corresponding alpha-0.2
            backbone while retaining the same seven-port protocol input.
        observed_indices (np.ndarray): Seven sorted zero-based physical port
            indices. They determine the protocol model's extraction and output
            masks and must match the calibration image encoding.
        calibration_images (np.ndarray): Train-only representative sparse
            images with shape ``(m, 32, 32, 3)``. They calibrate post-training
            INT8 ranges but do not update model weights.
        log_scale (float): Positive 99.9th-percentile ``log1p`` scale computed
            from the training split. It matches the existing training model's
            internal input normalization.

    Returns:
        Path: Saved strict-INT8 TFLite artifact for ``variant_name``.

    Raises:
        ValueError: If ``variant_name`` is unsupported or inputs are invalid.
        AssertionError: If conversion does not expose INT8 input and output.
        OSError: If the artifact cannot be written.
        Exception: Propagates TensorFlow model-building or conversion failures.
    """
    if variant_name not in VARIANT_METADATA:
        raise ValueError(
            f"Unsupported benchmark variant {variant_name!r}. "
            f"Expected one of {sorted(VARIANT_METADATA)}."
        )

    # Resetting the seed per variant makes untrained initial weights reproducible.
    tf.keras.utils.set_random_seed(MODEL_SEED)
    _, compact_decision_model = build_models(
        observed_indices=observed_indices,
        log_scale=log_scale,
        train_seed=MODEL_SEED,
        variant_name=variant_name,
        fixed_batch_size=BENCHMARK_BATCH_SIZE,
    )
    protocol_model = build_protocol_decision_model(
        compact_decision_model=compact_decision_model,
        observed_indices=observed_indices,
        variant_name=variant_name,
        fixed_batch_size=BENCHMARK_BATCH_SIZE,
    )
    tflite_model = convert_ptq_model_to_strict_int8_tflite(
        protocol_model,
        calibration_images,
    )
    tflite_model = replace_int8_log_with_float(tflite_model)
    test_strict_int8_io(tflite_model)

    output_path = OUTPUT_DIR / f"{variant_name}_7_ports_int8.tflite"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tflite.tmp")
    temporary_path.write_bytes(tflite_model)
    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    """Generate strict-INT8 benchmark models without executing any training.

    Args:
        None.

    Returns:
        None: Writes one seven-port TFLite artifact for each MobileNet variant
        to ``generated_7_port_benchmark_models`` and prints each path.

    Raises:
        ValueError: If the configured training split cannot provide valid
            calibration images.
        OSError: If an artifact destination cannot be created or written.
        Exception: Propagates TensorFlow data-loading, model-building, and
            TFLite conversion failures.
    """
    y_train, _, _, _, _, _ = split_kms_60_20_20_cache(
        common_path=None,
        seed=MODEL_SEED,
        regime_labels=KMS_REGIME_LABELS,
        samples_per_regime=None,
        verbose=True,
    )
    try:
        calibration_images, observed_indices, log_scale = build_calibration_images(
            y_train
        )
    finally:
        del y_train
        gc.collect()

    print(f"observed_indices={observed_indices.tolist()}")
    for variant_name in VARIANT_METADATA:
        try:
            output_path = save_untrained_variant(
                variant_name,
                observed_indices,
                calibration_images,
                log_scale,
            )
            print(f"generated_model={output_path}")
        finally:
            tf.keras.backend.clear_session()
            gc.collect()


if __name__ == "__main__":
    main()
