"""
TensorFlow quantization helpers.

This module keeps PTQ, QAT, TFLite validation, RAM estimation, and INT8
evaluation logic out of NAS scripts so trial objectives can stay focused on
study flow.

As of May 16, 2026, ai-edge-litert is Google's standalone Python package for LiteRT, the successor/evolution of TensorFlow Lite for on-device inference. It is meant for running .tflite models on edge platforms with small binary size, low latency, and hardware acceleration support. PyPI describes it as the official solution for mobile and embedded ML inference.

However, ai-edge-litert does not currently work well with training. Thus, this module still uses TensorFlow's TFLiteConverter and TFLite Interpreter for quantization and evaluation.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import contextlib
import os
import sys
import warnings
from typing import Callable, Iterable, Iterator

import numpy as np
import tensorflow as tf

# %%—————————————————————————————————————————————————————————————————————————— #
#                            stdout/stderr Supressor                           #
# ———————————————————————————————————————————————————————————————————————————— #


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True) -> Iterator[None]:
    """Suppress Python and native stdout/stderr inside a narrow code block.

    Args:
        enabled (bool): Whether stdout and stderr should be redirected to
            ``os.devnull`` while the context is active. If ``False``, the
            context does nothing.

    Yields:
        None: Control returns to the wrapped block while output redirection is
            active.

    Raises:
        None: Exceptions from the wrapped block are not swallowed. Output
            streams are restored before the exception propagates.
    """
    if not enabled:
        yield
        return

    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def make_tflite_interpreter(tflite_model: bytes) -> tf.lite.Interpreter:
    """Create a TFLite interpreter without default delegates.

    Args:
        tflite_model (bytes): Serialized TFLite flatbuffer to inspect or run.

    Returns:
        tf.lite.Interpreter: Unallocated interpreter using only built-in ops.

    Raises:
        Exception: Propagates TensorFlow Lite interpreter construction errors.
    """
    return tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_delegates=[],
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


# %%—————————————————————————————————————————————————————————————————————————— #
#                       Representative Calibration Data                        #
# ———————————————————————————————————————————————————————————————————————————— #
def make_representative_dataset(calibration_data: np.ndarray) -> Callable[[], Iterable[list[np.ndarray]]]:
    """Build a strict INT8 representative dataset factory.

    Args:
        calibration_data (np.ndarray): Scaled input samples with shape
            ``(n_samples, n_observed_ports)``. The array must contain at least
            one row and is cast to ``float32`` because TFLite calibration
            expects floating-point model inputs before quantization.

    Returns:
        Callable[[], Iterable[list[np.ndarray]]]: Zero-argument factory that
            yields one-sample batches in the format required by
            ``TFLiteConverter.representative_dataset``.

    Raises:
        ValueError: If ``calibration_data`` has zero rows.

    Examples:
        >>> samples = np.asarray([[0.0, 1.0]], dtype=np.float32)
        >>> representative_dataset = make_representative_dataset(samples)
        >>> next(representative_dataset())[0].shape
        (1, 2)
    """
    if calibration_data.shape[0] == 0:
        raise ValueError("calibration_data must have at least one sample.")

    calibration = calibration_data.astype(np.float32)

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        """Yield one-sample batches for TFLite calibration.

        Args:
            None: The generator closes over validated calibration samples.

        Yields:
            list[np.ndarray]: Single-element list containing one sample with a
                preserved batch dimension.

        Raises:
            None.
        """
        for sample_index in range(calibration.shape[0]):
            yield [calibration[sample_index : sample_index + 1]]

    return representative_dataset


def build_quantile_balanced_calibration_data(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_count: int,
    num_bins: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Select train-only calibration samples balanced by target quantiles.

    Args:
        X_train_scaled (np.ndarray): Train-only scaled input matrix used for
            TFLite representative calibration. It must contain the same number
            of rows as ``y_train``.
        y_train (np.ndarray): Unscaled train target matrix. Samples are binned
            by ``np.max(y_train, axis=1)`` so calibration covers weak and
            strong best-port cases without using validation or test rows.
        sample_count (int): Maximum number of calibration samples to select.
            Must be positive. Larger values can improve calibration coverage
            but increase TFLite conversion time.
        num_bins (int): Number of quantile bins. Must be positive. More bins
            increase target coverage granularity but can leave fewer rows per
            bin.
        seed (int): NumPy RNG seed used to shuffle candidates within each bin.
            Reusing the same seed keeps calibration deterministic.

    Returns:
        tuple[np.ndarray, list[int]]: Selected ``float32`` calibration matrix
            and a per-bin count list with length ``num_bins``.

    Raises:
        ValueError: If arrays are empty, lengths differ, or configuration
            values are not positive.

    Examples:
        >>> x = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
        >>> y = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
        >>> data, counts = build_quantile_balanced_calibration_data(
        ...     x, y, sample_count=2, num_bins=2, seed=0
        ... )
        >>> data.shape[0]
        2
    """
    if X_train_scaled.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Training data must be non-empty for INT8 calibration.")
    if X_train_scaled.shape[0] != y_train.shape[0]:
        raise ValueError("X_train_scaled and y_train must contain the same number of rows.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    rng = np.random.default_rng(seed)
    target_scores = np.max(y_train, axis=1)
    quantile_edges = np.quantile(target_scores, np.linspace(0.0, 1.0, num_bins + 1))
    selected_indices: list[int] = []
    calibration_counts: list[int] = []
    base_quota = sample_count // num_bins
    remainder = sample_count % num_bins

    for bin_index in range(num_bins):
        left_edge = quantile_edges[bin_index]
        right_edge = quantile_edges[bin_index + 1]
        if bin_index == num_bins - 1:
            bin_mask = (target_scores >= left_edge) & (target_scores <= right_edge)
        else:
            bin_mask = (target_scores >= left_edge) & (target_scores < right_edge)

        bin_indices = np.flatnonzero(bin_mask)
        rng.shuffle(bin_indices)

        # Distribute the fixed sample budget evenly so each target quantile can
        # contribute calibration examples before any fill rows are added.
        bin_quota = base_quota + (1 if bin_index < remainder else 0)
        selected_from_bin = bin_indices[: min(bin_quota, bin_indices.size)].tolist()
        selected_indices.extend(selected_from_bin)
        calibration_counts.append(len(selected_from_bin))

    if len(selected_indices) < min(sample_count, X_train_scaled.shape[0]):
        missing_count = min(sample_count, X_train_scaled.shape[0]) - len(selected_indices)
        remaining_indices = np.setdiff1d(np.arange(X_train_scaled.shape[0]), np.asarray(selected_indices))
        rng.shuffle(remaining_indices)
        selected_indices.extend(remaining_indices[:missing_count].tolist())

        # Assign fill rows to their real quantile bins so logged calibration
        # counts still describe the selected data.
        for index in selected_indices[-missing_count:]:
            bin_index = int(np.searchsorted(quantile_edges[1:-1], target_scores[index], side="right"))
            calibration_counts[bin_index] += 1

    return X_train_scaled[np.asarray(selected_indices, dtype=np.int64)].astype(np.float32), calibration_counts


# %%—————————————————————————————————————————————————————————————————————————— #
#                              TFLite Conversion                               #
# ———————————————————————————————————————————————————————————————————————————— #
def convert_ptq_model_to_strict_int8_tflite(
    model: tf.keras.Model,
    calibration_data: np.ndarray,
    *,
    suppress_output: bool = True,
) -> bytes:
    """Convert a Keras model to strict INT8 TFLite with PTQ calibration.

    Args:
        model (tf.keras.Model): Keras model to convert. This helper is used for
            untrained RAM pre-screening and trained INT8 objective evaluation.
        calibration_data (np.ndarray): Train-only scaled samples used to
            calibrate activation ranges. The array must be accepted by
            ``make_representative_dataset``.

    Returns:
        bytes: Serialized strict INT8 TFLite flatbuffer.

    Raises:
        Exception: Propagates TensorFlow conversion failures so the caller can
            mark the trial invalid with fixed penalty objectives.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calibration_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with suppress_native_output(suppress_output):
        return converter.convert()


def convert_qat_model_to_strict_int8_tflite(qat_model: tf.keras.Model) -> bytes:
    """Convert a trained QAT model to strict INT8 TFLite bytes.

    Args:
        qat_model (tf.keras.Model): Model produced by
            ``tfmot.quantization.keras.quantize_model`` after QAT fine-tuning.
            The model must contain QAT quantization wrappers; no PTQ fallback or
            representative calibration data is used in this path.

    Returns:
        bytes: Serialized strict INT8 TFLite flatbuffer.

    Raises:
        Exception: Propagates TensorFlow conversion failures so the trial can
            return the fixed invalid-objective tuple.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


# %%—————————————————————————————————————————————————————————————————————————— #
#                          Quantization Trial Tests                            #
# ———————————————————————————————————————————————————————————————————————————— #
def test_strict_int8_io(
    tflite_model: bytes,
    *,
    suppress_output: bool = True,
    suppress_warning: bool = True,
) -> None:
    """Test that a TFLite model exposes INT8 input and output tensors.

    Args:
        tflite_model (bytes): Serialized TFLite model produced by a strict INT8
            converter path. Empty bytes are invalid and should fail the trial.
        suppress_output (bool): Whether to suppress native stdout/stderr output.
        suppress_warning (bool): Whether to suppress the tf.lite.Interpreter
            deprecation warning. Defaults to True.

    Returns:
        None: The function completes only when both public model endpoints are
            INT8 tensors.

    Raises:
        AssertionError: If the model input or output dtype is not INT8.
        Exception: Propagates interpreter allocation failures.
    """
    with suppress_native_output(suppress_output):
        with warnings.catch_warnings():
            if suppress_warning:
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*tf.lite.Interpreter is deprecated.*",
                )
            interpreter = make_tflite_interpreter(tflite_model)
            interpreter.allocate_tensors()

    input_dtype = interpreter.get_input_details()[0]["dtype"]
    output_dtype = interpreter.get_output_details()[0]["dtype"]

    if input_dtype != np.int8:
        raise AssertionError(f"Expected INT8 input tensor, got {input_dtype}.")
    if output_dtype != np.int8:
        raise AssertionError(f"Expected INT8 output tensor, got {output_dtype}.")


def test_no_nested_keras_models(model: tf.keras.Model) -> None:
    """Test that a model has no nested Keras ``Model`` layers before QAT.

    Args:
        model (tf.keras.Model): Model to inspect before TFMOT QAT wrapping.
            Nested Keras models are unsupported by the current whole-model QAT
            path and must fail the trial explicitly.

    Returns:
        None: The function completes only when every internal layer is a layer,
            not another Keras model.

    Raises:
        AssertionError: If any inner layer is itself a Keras ``Model``.
    """
    nested_model_layers = [
        layer.name for layer in model.layers if isinstance(layer, tf.keras.Model) and layer is not model
    ]

    if nested_model_layers:
        raise AssertionError(
            "TFMOT QAT does not support nested Keras Model layers. "
            f"Nested model layers found: {nested_model_layers}"
        )


# %%—————————————————————————————————————————————————————————————————————————— #
#                            TFLite RAM Estimation                             #
# ———————————————————————————————————————————————————————————————————————————— #
def estimate_tflite_peak_ram_bytes(
    tflite_model: bytes,
    *,
    suppress_output: bool = True,
    suppress_warning: bool = True,
) -> int:
    """Estimate peak live RAM from an INT8 TFLite operator schedule.

    Args:
        tflite_model (bytes): Serialized strict INT8 TFLite flatbuffer. The
            model is allocated in a local interpreter to inspect tensor sizes
            and per-operator input/output references.
        suppress_output (bool): Whether to suppress native stdout/stderr output.
        suppress_warning (bool): Whether to suppress the tf.lite.Interpreter
            deprecation warning. Defaults to True.

    Returns:
        int: Approximate peak live tensor memory in bytes.

    Raises:
        RuntimeError: If the TFLite interpreter does not expose operator
            schedule details.
        Exception: Propagates interpreter allocation failures.
    """
    with suppress_native_output(suppress_output):
        with warnings.catch_warnings():
            if suppress_warning:
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*tf.lite.Interpreter is deprecated.*",
                )
            interpreter = make_tflite_interpreter(tflite_model)
            interpreter.allocate_tensors()

    tensor_size_by_index: dict[int, int] = {}
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
        input_indices = [int(idx) for idx in op_detail.get("inputs", []) if int(idx) >= 0]
        output_indices = [int(idx) for idx in op_detail.get("outputs", []) if int(idx) >= 0]
        touched_indices = set(input_indices + output_indices)
        live_bytes = sum(tensor_size_by_index.get(idx, 0) for idx in touched_indices)
        peak_bytes = max(peak_bytes, int(live_bytes))

    return int(peak_bytes)


# %%—————————————————————————————————————————————————————————————————————————— #
#                              TFLite Evaluation                               #
# ———————————————————————————————————————————————————————————————————————————— #
def evaluate_tflite_model_from_bytes(
    tflite_model: bytes,
    X_val_scaled: np.ndarray,
    y_val_scaled: np.ndarray,
    *,
    suppress_output: bool = True,
    suppress_warning: bool = True,
) -> float:
    """Evaluate strict INT8 TFLite validation MSE in scaled target space.

    Args:
        tflite_model (bytes): Serialized strict INT8 TFLite flatbuffer.
        X_val_scaled (np.ndarray): Validation inputs already transformed by the
            train-fitted input ``MinMaxScaler``.
        y_val_scaled (np.ndarray): Validation targets already transformed by
            the train-fitted target ``MinMaxScaler``. MSE is computed directly
            against this scaled target matrix.
        suppress_output (bool): Whether to suppress native stdout/stderr output.
        suppress_warning (bool): Whether to suppress the tf.lite.Interpreter
            deprecation warning. Defaults to True.

    Returns:
        float: Mean squared error between dequantized TFLite predictions and
            scaled validation targets.

    Raises:
        ValueError: If the TFLite model does not expose quantized input or
            output scales.
        Exception: Propagates interpreter allocation and invocation failures.
    """
    with suppress_native_output(suppress_output):
        with warnings.catch_warnings():
            if suppress_warning:
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*tf.lite.Interpreter is deprecated.*",
                )
            interpreter = make_tflite_interpreter(tflite_model)
            interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]

    if input_scale == 0 or output_scale == 0:
        raise ValueError("Strict INT8 TFLite model must expose non-zero input and output scales.")

    predictions = []
    for sample in X_val_scaled.astype(np.float32):
        # Match deployment math by quantizing each scaled validation sample with
        # the interpreter-provided input scale and zero point.
        quantized_input = np.round(sample / input_scale + input_zero_point)
        quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized_input.reshape(input_detail["shape"]))
        interpreter.invoke()

        quantized_output = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        output = (quantized_output - output_zero_point) * output_scale
        predictions.append(output.reshape(-1))

    y_pred_scaled = np.asarray(predictions, dtype=np.float32)
    return float(np.mean(np.square(y_pred_scaled - y_val_scaled.astype(np.float32))))
