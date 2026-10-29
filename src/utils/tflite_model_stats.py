"""
Profile strict INT8 TFLite models and persist metrics on Optuna trials.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

# %%—————————————————————————————————————————————————————————————————————————— #
#                                    Imports                                   #
# ———————————————————————————————————————————————————————————————————————————— #
from typing import Any, Iterable, Optional

import os
import time
import warnings

import numpy as np
import optuna
import tensorflow as tf

from utils.qttf import estimate_tflite_peak_ram_bytes, make_tflite_interpreter


# %%—————————————————————————————————————————————————————————————————————————— #
#                               Helper functions                               #
# ———————————————————————————————————————————————————————————————————————————— #
def format_tflite_bytes(value: int | None) -> str:
    """Format a byte count as a human-readable string.

    Args:
        value (int | None): Number of bytes to format.

    Returns:
        str: Human-readable byte string.

    Raises:
        None.
    """
    if value is None:
        return "N/A"

    units = ("B", "KiB", "MiB", "GiB")
    scaled_value = float(value)
    unit_index = 0

    while scaled_value >= 1024.0 and unit_index < len(units) - 1:
        scaled_value /= 1024.0
        unit_index += 1

    return f"{value} B ({scaled_value:.3f} {units[unit_index]})"


def format_tflite_scientific_with_unit(value: float, unit: str) -> str:
    """Format a scalar in scientific notation with a trailing unit.

    Args:
        value (float): Scalar value to format.
        unit (str): Unit suffix appended after the formatted quantity.

    Returns:
        str: Display string using scientific notation, for example,
            ``1.3368×10^-3 s``.

    Raises:
        None.
    """
    mantissa, exponent_text = f"{value:.4e}".split("e")
    exponent = int(exponent_text)
    return f"{mantissa}×10^{exponent} {unit}"


def format_tflite_count(value: int, noun: str) -> str:
    """Format a count with a singular display noun.

    Args:
        value (int): Count to display.
        noun (str): Unit label appended to the count, such as ``"tensor"`` or
            ``"operator"``.

    Returns:
        str: Display string such as ``"12 tensors"``.

    Raises:
        None.
    """
    suffix = noun if value == 1 else f"{noun}s"
    return f"{value:,} {suffix}"


def get_tflite_model_bytes(tflite_model: str | bytes) -> bytes:
    """Load TFLite model bytes from a path or an in-memory buffer.

    Args:
        tflite_model (str | bytes): TFLite model path or serialized TFLite
            FlatBuffer bytes.

    Returns:
        bytes: Serialized TFLite model bytes.

    Raises:
        TypeError: If ``tflite_model`` is neither ``str`` nor ``bytes``.
        FileNotFoundError: If ``tflite_model`` is a path that does not exist.
        OSError: If the model file cannot be read.
    """
    if isinstance(tflite_model, bytes):
        return tflite_model

    if isinstance(tflite_model, str):
        if not os.path.exists(tflite_model):
            raise FileNotFoundError(f"TFLite model file not found: {tflite_model}")
        with open(tflite_model, "rb") as model_file:
            return model_file.read()

    raise TypeError("tflite_model must be a file path string or bytes.")


def make_tflite_dummy_input(input_detail: dict, batch_size: int) -> np.ndarray:
    """Create one dummy tensor compatible with a TFLite model input.

    Args:
        input_detail (dict): TFLite input metadata dictionary returned by
            ``interpreter.get_input_details()[0]``.
        batch_size (int): Requested batch size for latency profiling.

    Returns:
        np.ndarray: Dummy tensor matching the model input dtype and shape.

    Raises:
        ValueError: If ``batch_size`` is less than ``1`` or the input dtype is
            unsupported.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    input_shape = np.asarray(input_detail["shape"], dtype=np.int64)
    input_shape[0] = batch_size
    input_dtype = np.dtype(input_detail["dtype"])

    if input_dtype == np.dtype(np.int8):
        return np.zeros(input_shape, dtype=np.int8)
    if input_dtype == np.dtype(np.uint8):
        return np.zeros(input_shape, dtype=np.uint8)
    if input_dtype == np.dtype(np.float32):
        return np.zeros(input_shape, dtype=np.float32)

    raise ValueError(f"Unsupported TFLite input dtype for profiling: {input_dtype}")


def resize_tflite_batch_if_needed(
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    batch_size: int,
) -> None:
    """Resize the TFLite model input batch dimension when needed.

    Args:
        interpreter (tf.lite.Interpreter): TFLite interpreter.
        input_detail (dict): Input detail dictionary from the interpreter.
        batch_size (int): Desired batch size.

    Returns:
        None: The interpreter input tensor is resized in-place if necessary.

    Raises:
        ValueError: If ``batch_size`` is less than ``1``.
        Exception: Propagates TensorFlow Lite resize failures.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    input_shape = np.asarray(input_detail["shape"], dtype=np.int64)

    if int(input_shape[0]) == batch_size:
        return

    input_shape[0] = batch_size
    interpreter.resize_tensor_input(
        input_detail["index"],
        input_shape.astype(np.int32),
        strict=False,
    )


def measure_tflite_latency_seconds(
    tflite_model_bytes: bytes,
    batch_size: int,
    test_runs: int,
    *,
    suppress_warning: bool = True,
) -> dict[str, float]:
    """Measure TFLite interpreter latency using a dummy input tensor.

    Args:
        tflite_model_bytes (bytes): Serialized TFLite model.
        batch_size (int): Batch size used for dummy input inference.
        test_runs (int): Number of timed invocations.
        suppress_warning (bool): Whether to suppress the tf.lite.Interpreter
            deprecation warning. Defaults to True.

    Returns:
        dict[str, float]: Latency statistics in seconds and milliseconds.

    Raises:
        ValueError: If ``test_runs`` is less than ``1``.
        Exception: Propagates TensorFlow Lite allocation and invocation
            failures.
    """
    if test_runs < 1:
        raise ValueError("test_runs must be at least 1.")

    with warnings.catch_warnings():
        if suppress_warning:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*tf.lite.Interpreter is deprecated.*",
            )
        interpreter = make_tflite_interpreter(tflite_model_bytes)

    input_detail = interpreter.get_input_details()[0]
    resize_tflite_batch_if_needed(
        interpreter=interpreter,
        input_detail=input_detail,
        batch_size=batch_size,
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    dummy_input = make_tflite_dummy_input(
        input_detail=input_detail,
        batch_size=batch_size,
    )

    interpreter.set_tensor(input_detail["index"], dummy_input)

    # Warm-up is excluded because the first invocation can include interpreter
    # setup overhead that does not represent steady-state inference latency.
    interpreter.invoke()

    latencies = []
    for _ in range(test_runs):
        start_time = time.perf_counter()
        interpreter.invoke()
        latencies.append(time.perf_counter() - start_time)

    latency_array = np.asarray(latencies, dtype=np.float64)

    return {
        "mean_seconds": float(np.mean(latency_array)),
        "std_seconds": float(np.std(latency_array)),
        "min_seconds": float(np.min(latency_array)),
        "max_seconds": float(np.max(latency_array)),
        "mean_ms": float(np.mean(latency_array) * 1000.0),
        "std_ms": float(np.std(latency_array) * 1000.0),
        "min_ms": float(np.min(latency_array) * 1000.0),
        "max_ms": float(np.max(latency_array) * 1000.0),
    }


def summarize_tflite_interpreter(
    tflite_model_bytes: bytes,
    *,
    suppress_warning: bool = True,
) -> dict[str, Any]:
    """Collect structural TFLite interpreter metadata.

    Args:
        tflite_model_bytes (bytes): Serialized TFLite model.
        suppress_warning (bool): Whether to suppress the tf.lite.Interpreter
            deprecation warning. Defaults to True.

    Returns:
        dict[str, Any]: Dictionary with tensor, operator, input, and output
            metadata.

    Raises:
        Exception: Propagates TensorFlow Lite allocation failures.
    """
    with warnings.catch_warnings():
        if suppress_warning:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*tf.lite.Interpreter is deprecated.*",
            )
        interpreter = make_tflite_interpreter(tflite_model_bytes)
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()

    op_details = []
    if hasattr(interpreter, "_get_ops_details"):
        op_details = interpreter._get_ops_details()

    return {
        "input_details": [
            {
                "name": str(detail["name"]),
                "shape": detail["shape"].astype(int).tolist(),
                "dtype": str(np.dtype(detail["dtype"])),
                "quantization": [
                    float(detail["quantization"][0]),
                    int(detail["quantization"][1]),
                ],
            }
            for detail in input_details
        ],
        "output_details": [
            {
                "name": str(detail["name"]),
                "shape": detail["shape"].astype(int).tolist(),
                "dtype": str(np.dtype(detail["dtype"])),
                "quantization": [
                    float(detail["quantization"][0]),
                    int(detail["quantization"][1]),
                ],
            }
            for detail in output_details
        ],
        "tensor_count": int(len(tensor_details)),
        "operator_count": int(len(op_details)),
        "operators": [
            {
                "index": int(index),
                "op_name": str(detail.get("op_name", "UNKNOWN")),
                "inputs": [int(value) for value in detail.get("inputs", [])],
                "outputs": [int(value) for value in detail.get("outputs", [])],
            }
            for index, detail in enumerate(op_details)
        ],
    }


# %%—————————————————————————————————————————————————————————————————————————— #
#                                Public helper                                 #
# ———————————————————————————————————————————————————————————————————————————— #
def set_user_attr_tflite_model_stats(
    trial: optuna.Trial,
    tflite_model: str | bytes,
    *,
    batch_size: int = 1,
    stats_to_measure: Iterable[str] = (
        "model_size",
        "peak_ram_bytes",
        "tensor_count",
        "operator_count",
        "input_details",
        "output_details",
        "inference_latency",
    ),
    test_runs: int = 10,
    verbose: int = 0,
    extra_attrs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Profile a TFLite model and persist statistics on an Optuna trial.

    Args:
        trial (optuna.Trial): Optuna trial that will receive the TFLite
            statistics as user attributes.
        tflite_model (str | bytes): Path to a ``.tflite`` file or serialized
            TFLite model bytes.
        batch_size (int): Batch size used when creating dummy inputs for
            latency profiling.
        stats_to_measure (Iterable[str]): Metric names to compute. Supported
            values are ``"model_size"``, ``"peak_ram_bytes"``,
            ``"tensor_count"``, ``"operator_count"``, ``"input_details"``,
            ``"output_details"``, and ``"inference_latency"``.
        test_runs (int): Number of timed TFLite invocations for latency
            profiling.
        verbose (int): Verbosity level. If greater than zero, a compact report
            is printed.
        extra_attrs (Optional[dict[str, Any]]): Additional attributes stored
            with the TFLite statistics.

    Returns:
        dict[str, Any]: TFLite statistics dictionary.

    Raises:
        ValueError: If ``batch_size`` or ``test_runs`` is invalid, or if an
            unsupported metric name is requested.
        TypeError: If ``tflite_model`` is neither a path nor bytes.
        FileNotFoundError: If ``tflite_model`` is a missing path.
        Exception: Propagates TensorFlow Lite interpreter failures.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if test_runs < 1:
        raise ValueError("test_runs must be at least 1.")

    supported_stats = {
        "model_size",
        "peak_ram_bytes",
        "tensor_count",
        "operator_count",
        "input_details",
        "output_details",
        "inference_latency",
    }
    requested_stats = set(stats_to_measure)
    unsupported_stats = requested_stats - supported_stats

    if unsupported_stats:
        raise ValueError(f"Unsupported TFLite stats requested: {sorted(unsupported_stats)}.")

    extra_attrs = extra_attrs or {}
    tflite_model_bytes = get_tflite_model_bytes(tflite_model)
    interpreter_summary = summarize_tflite_interpreter(tflite_model_bytes)

    stats: dict[str, Any] = {}

    if "model_size" in requested_stats:
        stats["model_size"] = int(len(tflite_model_bytes))
        stats["model_size_display"] = format_tflite_bytes(stats["model_size"])

    if "peak_ram_bytes" in requested_stats:
        stats["peak_ram_bytes"] = int(estimate_tflite_peak_ram_bytes(tflite_model_bytes))
        stats["peak_ram_bytes_display"] = format_tflite_bytes(stats["peak_ram_bytes"])

    if "tensor_count" in requested_stats:
        stats["tensor_count"] = int(interpreter_summary["tensor_count"])
        stats["tensor_count_display"] = format_tflite_count(stats["tensor_count"], "tensor")

    if "operator_count" in requested_stats:
        stats["operator_count"] = int(interpreter_summary["operator_count"])
        stats["operator_count_display"] = format_tflite_count(stats["operator_count"], "operator")

    if "input_details" in requested_stats:
        stats["input_details"] = interpreter_summary["input_details"]

    if "output_details" in requested_stats:
        stats["output_details"] = interpreter_summary["output_details"]

    if "inference_latency" in requested_stats:
        stats["inference_latency"] = measure_tflite_latency_seconds(
            tflite_model_bytes=tflite_model_bytes,
            batch_size=batch_size,
            test_runs=test_runs,
        )
        stats["inference_latency_mean_display"] = format_tflite_scientific_with_unit(
            stats["inference_latency"]["mean_seconds"],
            "s",
        )
        stats["inference_latency_peak_display"] = format_tflite_scientific_with_unit(
            stats["inference_latency"]["max_seconds"],
            "s",
        )

    if extra_attrs:
        stats["extra_attrs"] = extra_attrs

    trial.set_user_attr("tflite_model_stats", stats)

    if "model_size" in stats:
        trial.set_user_attr("tflite_model_size", stats["model_size_display"])

    if "peak_ram_bytes" in stats:
        trial.set_user_attr("tflite_peak_ram_bytes", stats["peak_ram_bytes_display"])

    if "tensor_count" in stats:
        trial.set_user_attr("tflite_tensor_count", stats["tensor_count_display"])

    if "operator_count" in stats:
        trial.set_user_attr("tflite_operator_count", stats["operator_count_display"])

    if "input_details" in stats:
        trial.set_user_attr("tflite_input_details", stats["input_details"])

    if "output_details" in stats:
        trial.set_user_attr("tflite_output_details", stats["output_details"])

    if "inference_latency" in stats:
        trial.set_user_attr("tflite_inference_latency", stats["inference_latency"])
        trial.set_user_attr("tflite_inference_latency_mean", stats["inference_latency_mean_display"])
        trial.set_user_attr("tflite_inference_latency_peak", stats["inference_latency_peak_display"])

    if extra_attrs:
        trial.set_user_attr("tflite_model_stats_extra_attrs", extra_attrs)

    return stats
