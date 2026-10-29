"""Compare INT8 MobileNet selectors and the WCNPS 2025 CNNs on CPU."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import csv
import json
import multiprocessing
import os
import platform
import sys
import threading
import time
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/train/benchmark_cpu_architectures.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
sys.path.insert(0, str(_COMPANION_ROOT / 'src'))

from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    TOTAL_PORTS,
    VARIANT_METADATA,
    build_models,
)
from utils.data import get_observed_ports
from utils.qttf import convert_ptq_model_to_strict_int8_tflite


# WCNPS 2025, Table I. Batch size and learning rate are training-only settings.
OBSERVATION_BUDGETS = (3, 4, 5, 6, 7)
WCNPS_2025_CONFIGS = {
    3: {
        "filters": (192, 384),
        "kernels": (3, 3),
        "pools": (1, 1),
        "activations": ("relu", "relu"),
    },
    4: {
        "filters": (160, 256),
        "kernels": (4, 5),
        "pools": (1, 2),
        "activations": ("relu", "relu"),
    },
    5: {
        "filters": (512,),
        "kernels": (5,),
        "pools": (4,),
        "activations": ("swish",),
    },
    6: {
        "filters": (192, 160, 96),
        "kernels": (5, 1, 2),
        "pools": (1, 1, 1),
        "activations": ("leaky_relu", "relu", "elu"),
    },
    7: {
        "filters": (416, 160, 224),
        "kernels": (5, 4, 5),
        "pools": (1, 1, 1),
        "activations": ("relu", "swish", "swish"),
    },
}
WARMUP = 25
BLOCKS = 100
INVOCATIONS_PER_BLOCK = 10
CALIBRATION_SAMPLES = 256
OUTPUT_DIR = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/cpu_architecture_screen_results'
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.001


def build_wcnps_2025_cnn(observed_ports: int) -> tf.keras.Model:
    """Build the published CNN for one observation budget.

    Args:
        observed_ports (int): Number of observed ports. Must be from three to
            seven, inclusive.

    Returns:
        tf.keras.Model: Untrained CNN mapping observed values to 100 scores.

    Raises:
        ValueError: If the observation budget has no published configuration.
    """
    if observed_ports not in WCNPS_2025_CONFIGS:
        raise ValueError(
            f"No WCNPS 2025 CNN configuration for {observed_ports} observed ports."
        )
    config = WCNPS_2025_CONFIGS[observed_ports]
    inputs = tf.keras.Input(shape=(observed_ports,), name="observed_snr")
    x = tf.keras.layers.Reshape((observed_ports, 1))(inputs)
    for index, (width, kernel, pool, activation) in enumerate(
        zip(
            config["filters"],
            config["kernels"],
            config["pools"],
            config["activations"],
        )
    ):
        x = tf.keras.layers.Conv1D(
            width, kernel, padding="same", name=f"conv_{index}"
        )(x)
        if activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(name=f"activation_{index}")(x)
        else:
            x = tf.keras.layers.Activation(activation, name=f"activation_{index}")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool, padding="same", name=f"pool_{index}"
        )(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(TOTAL_PORTS, name="predicted_ports")(x)
    return tf.keras.Model(
        inputs,
        outputs,
        name=f"wcnps_2025_cnn_{observed_ports}",
    )


def create_reference_tflite_interpreter(
    tflite_model: bytes,
) -> tf.lite.Interpreter:
    """Create a single-thread TFLite interpreter without default delegates.

    Args:
        tflite_model (bytes): Serialized TFLite flatbuffer to execute.

    Returns:
        tf.lite.Interpreter: Allocated reference-kernel interpreter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*tf.lite.Interpreter is deprecated.*",
        )
        interpreter = tf.lite.Interpreter(
            model_content=tflite_model,
            num_threads=1,
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
    interpreter.allocate_tensors()
    return interpreter


def make_calibration_values(observed_ports: int) -> np.ndarray:
    """Build deterministic nonnegative values for graph-only INT8 calibration.

    Args:
        observed_ports (int): Number of values accepted by the model.

    Returns:
        np.ndarray: Calibration matrix with one row per calibration sample.
    """
    return np.linspace(
        0.0,
        10.0,
        CALIBRATION_SAMPLES * observed_ports,
        dtype=np.float32,
    ).reshape(CALIBRATION_SAMPLES, observed_ports)


def parse_proc_status_rss_kib(status_text: str) -> int:
    """Extract the current resident set size from Linux process status text.

    Args:
        status_text (str): Contents of ``/proc/<pid>/status``.

    Returns:
        int: Current resident set size in KiB.

    Raises:
        ValueError: If the status text does not contain a ``VmRSS`` field.
    """
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise ValueError("Linux process status does not contain a VmRSS field.")


def read_current_rss_kib() -> int:
    """Read the current process resident set size on Linux.

    Returns:
        int: Current resident set size in KiB.
    """
    return parse_proc_status_rss_kib(
        Path("/proc/self/status").read_text(encoding="utf-8")
    )


def summarize_peak_rss(baseline_rss_kib: int, samples_kib: list[int]) -> dict[str, int]:
    """Summarize absolute and incremental resident-memory peaks.

    Args:
        baseline_rss_kib (int): RSS immediately before interpreter creation.
        samples_kib (list[int]): RSS samples collected during inference setup
            and execution.

    Returns:
        dict[str, int]: Baseline, absolute peak, and peak increase in KiB.

    Raises:
        ValueError: If no memory samples were collected.
    """
    if not samples_kib:
        raise ValueError("At least one RSS sample is required.")
    peak_rss_kib = max(samples_kib)
    return {
        "baseline_rss_kib": baseline_rss_kib,
        "peak_rss_kib": peak_rss_kib,
        "peak_incremental_rss_kib": max(0, peak_rss_kib - baseline_rss_kib),
    }


def quantize_fixed_input(
    interpreter: tf.lite.Interpreter,
    values: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Quantize a fixed input row for an allocated TFLite interpreter.

    Args:
        interpreter (tf.lite.Interpreter): Allocated strict-INT8 interpreter.
        values (np.ndarray): Float32 input row matching the model input shape.

    Returns:
        tuple[dict, np.ndarray]: Input tensor metadata and quantized values.

    Raises:
        ValueError: If the model has an invalid input quantization scale.
    """
    input_detail = interpreter.get_input_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    if input_scale <= 0.0:
        raise ValueError(f"Invalid TFLite input scale: {input_scale}.")
    fixed_input = np.clip(
        np.rint(values / input_scale + input_zero_point),
        -128,
        127,
    ).astype(np.int8)
    return input_detail, fixed_input


def _memory_benchmark_worker(
    tflite_model: bytes,
    values: np.ndarray,
    connection,
) -> None:
    """Measure one interpreter in an otherwise fresh spawned process."""
    tracking_started = threading.Event()
    tracking_stopped = threading.Event()
    samples_kib = []

    def sample_rss() -> None:
        tracking_started.wait()
        while not tracking_stopped.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
            samples_kib.append(read_current_rss_kib())

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    baseline_rss_kib = read_current_rss_kib()
    samples_kib.append(baseline_rss_kib)
    tracking_started.set()
    try:
        interpreter = create_reference_tflite_interpreter(tflite_model)
        input_detail, fixed_input = quantize_fixed_input(interpreter, values)
        interpreter.set_tensor(input_detail["index"], fixed_input)
        for _ in range(WARMUP + BLOCKS * INVOCATIONS_PER_BLOCK):
            interpreter.invoke()
    finally:
        tracking_stopped.set()
        sampler.join()
        samples_kib.append(read_current_rss_kib())
    connection.send(summarize_peak_rss(baseline_rss_kib, samples_kib))
    connection.close()


def benchmark_peak_memory(tflite_model: bytes, values: np.ndarray) -> dict[str, int]:
    """Measure peak TFLite inference RSS in a fresh spawned process.

    Args:
        tflite_model (bytes): Strict-INT8 model to load and invoke.
        values (np.ndarray): One float32 input row with the model's input shape.

    Returns:
        dict[str, int]: Baseline, absolute peak, and incremental peak RSS in
        KiB. The baseline is captured after imports and argument transfer but
        before interpreter creation.

    Raises:
        RuntimeError: If the memory worker fails or does not return a result.
    """
    context = multiprocessing.get_context("spawn")
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_memory_benchmark_worker,
        args=(tflite_model, values, sending_connection),
    )
    process.start()
    sending_connection.close()
    if not receiving_connection.poll(300):
        process.terminate()
        process.join()
        raise RuntimeError("Memory benchmark worker timed out.")
    result = receiving_connection.recv()
    receiving_connection.close()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Memory benchmark worker exited with code {process.exitcode}."
        )
    return result


def benchmark(tflite_model: bytes, values: np.ndarray) -> dict[str, float]:
    """Measure reference-kernel TFLite latency on a fixed batch-one input.

    Args:
        tflite_model (bytes): Strict-INT8 model to invoke on CPU.
        values (np.ndarray): One float32 input row with the model's input shape.

    Returns:
        dict[str, float]: Median block mean and block distribution in ms.

    Raises:
        ValueError: If the model has an invalid input quantization scale.
    """
    interpreter = create_reference_tflite_interpreter(tflite_model)
    input_detail, fixed_input = quantize_fixed_input(interpreter, values)
    interpreter.set_tensor(input_detail["index"], fixed_input)
    for _ in range(WARMUP):
        interpreter.invoke()
    block_means = []
    for _ in range(BLOCKS):
        start = time.perf_counter_ns()
        for _ in range(INVOCATIONS_PER_BLOCK):
            interpreter.invoke()
        block_means.append(
            (time.perf_counter_ns() - start) / INVOCATIONS_PER_BLOCK / 1e6
        )
    return {
        "median_block_mean_ms": float(np.median(block_means)),
        "mean_block_mean_ms": float(np.mean(block_means)),
        "std_block_mean_ms": float(np.std(block_means)),
        "p95_block_mean_ms": float(np.percentile(block_means, 95)),
        "min_block_mean_ms": float(np.min(block_means)),
        "max_block_mean_ms": float(np.max(block_means)),
    }


def main() -> None:
    """Run four INT8 architectures at five budgets and save provenance."""
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "tensorflow": tf.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpu": Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].splitlines()[0].strip(" :\t") if Path("/proc/cpuinfo").exists() else platform.processor(),
        "threads": 1,
        "batch_size": 1,
        "warmup": WARMUP,
        "blocks": BLOCKS,
        "invocations_per_block": INVOCATIONS_PER_BLOCK,
        "backend": "strict INT8 TFLite CPU reference kernels, one thread, no GPU, default delegates disabled",
        "weights": "random initialization, no training",
        "observation_budgets": OBSERVATION_BUDGETS,
        "input": "fixed zero-valued vector quantized with each model input scale",
        "calibration": "256 deterministic synthetic nonnegative rows spanning 0 to 10; conversion only, no accuracy claim",
        "cnn_source": "WCNPS 2025 Table I budget-specific configurations, Conv1D same padding, max pooling same padding, flatten, linear 100-output head",
        "mobilenet_source": "safe_selector.build_models, alpha 0.2 GroupNorm variants, compact decision model",
        "model_size": "serialized strict-INT8 TFLite flatbuffer bytes",
        "memory": "fresh spawned process per case; Linux VmRSS sampled every 1 ms from immediately before interpreter creation through allocation, warm-up, and 1000 timed-equivalent invocations",
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    for budget in OBSERVATION_BUDGETS:
        indices = get_observed_ports(
            np.zeros((1, TOTAL_PORTS), dtype=np.float32), budget, TOTAL_PORTS
        )[1]
        fixed_input = np.zeros((1, budget), dtype=np.float32)
        for name in (*VARIANT_METADATA, "wcnps_2025_cnn"):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(0)
            if name == "wcnps_2025_cnn":
                model = build_wcnps_2025_cnn(budget)
                config = WCNPS_2025_CONFIGS[budget]
            else:
                _, model = build_models(
                    observed_indices=indices,
                    log_scale=1.0,
                    train_seed=0,
                    variant_name=name,
                    fixed_batch_size=1,
                )
                config = VARIANT_METADATA[name]
            calibration_values = make_calibration_values(budget)
            tflite_model = convert_ptq_model_to_strict_int8_tflite(
                model,
                calibration_values,
            )
            row = {
                "architecture": name,
                "observed_ports": budget,
                "observed_indices": json.dumps(indices.tolist()),
                "configuration": json.dumps(config),
                "parameters": model.count_params(),
                "model_size_bytes": len(tflite_model),
                **benchmark_peak_memory(tflite_model, fixed_input),
                **benchmark(tflite_model, fixed_input),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
