"""Run held-out OP evaluation for completed all-regimes retraining cases."""

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

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/op_eval_retrain.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mobilenet.keras.sparse_input import SparsePortImagePredictAdapter, iter_sparse_port_image_batches
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    ExpandedPortImage,
    SafeDecisionScores,
    SparseImageObservedValues,
)
from utils.constants import BLUE, GREEN, RESET
from utils.data import load_raw_kms_cache
from utils.op_eval import evaluate_op_per_regime_for_thresholds, save_op_outputs

ROOT_SEARCH_DIR: Path | None = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/results/retrain_all_regimes'
RETRAIN_RUN_DIR: list[Path] = []
TOTAL_PORTS = 100
THRESHOLD = [1.1885398626327515]
SNR_LINEAR = 1.0
BATCH_SIZE = 4096
TFLITE_BATCH_SIZE = 256
SKIP_IF_OP_EVAL_EXISTS = True

_SAFE_SELECTOR_CUSTOM_OBJECTS = {
    "ExpandedPortImage": ExpandedPortImage,
    "SafeDecisionScores": SafeDecisionScores,
    "SparseImageObservedValues": SparseImageObservedValues,
}


class TFLitePredictAdapter:
    """Expose a strict-INT8 sparse MobileNet model through ``predict``."""

    def __init__(self, model_path: Path, observed_indices: np.ndarray) -> None:
        """Load one TFLite model and validate its inference contract.

        Args:
            model_path (Path): Non-empty strict-INT8 `.tflite` artifact whose
                input and output tensors must have non-zero quantization scales.
            observed_indices (np.ndarray): Zero-based physical port indices used
                to expand compact scaled inputs into sparse 32x32x3 images.

        Returns:
            None: The initialized adapter stores one allocated interpreter.

        Raises:
            FileNotFoundError: If ``model_path`` is missing.
            ValueError: If TFLite tensor types, shape, or quantization are unsupported.
            Exception: If TensorFlow Lite cannot allocate the model tensors.
        """
        if not _is_non_empty_file(model_path):
            raise FileNotFoundError(f"Expected non-empty TFLite model at {model_path}.")

        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_delegates=[],
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        self.input_detail = self.interpreter.get_input_details()[0]
        input_shape = np.asarray(self.input_detail["shape"], dtype=np.int64).tolist()
        if input_shape != [1, 32, 32, 3]:
            raise ValueError(f"Unsupported TFLite input shape: {input_shape}.")
        input_signature = np.asarray(
            self.input_detail["shape_signature"],
            dtype=np.int64,
        ).tolist()
        if input_signature != [-1, 32, 32, 3]:
            raise ValueError(f"Unsupported TFLite input shape signature: {input_signature}.")

        # The serialized deployment artifact retains batch size one, while its
        # dynamic signature permits equivalent batched offline evaluation.
        self.inference_batch_size = TFLITE_BATCH_SIZE
        self.interpreter.resize_tensor_input(
            self.input_detail["index"],
            [self.inference_batch_size, 32, 32, 3],
            strict=True,
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        input_scale, _ = self.input_detail["quantization"]
        output_scale, _ = self.output_detail["quantization"]
        if input_scale == 0 or output_scale == 0:
            raise ValueError("Strict INT8 TFLite model must expose non-zero input and output scales.")
        supported_dtypes = (np.dtype(np.int8), np.dtype(np.uint8), np.dtype(np.float32))
        if np.dtype(self.input_detail["dtype"]) not in supported_dtypes:
            raise ValueError(f"Unsupported TFLite input dtype: {self.input_detail['dtype']}.")
        if np.dtype(self.output_detail["dtype"]) not in supported_dtypes:
            raise ValueError(f"Unsupported TFLite output dtype: {self.output_detail['dtype']}.")

    def predict(self, x: np.ndarray, batch_size: int = 64, verbose: int = 0) -> np.ndarray:
        """Predict scaled target values from compact observed-port inputs.

        Args:
            x (np.ndarray): Two-dimensional scaled input matrix with shape
                ``(samples, observed_ports)``.
            batch_size (int): Number of compact rows expanded into sparse images
                requested by the shared evaluator. Strict-INT8 inference uses
                the fixed no-delegate batch size of 256 independently to limit
                temporary tensor memory while retaining batched throughput.
                Changing this compatibility value does not change TFLite
                outputs, allocation, logging, or internal batch size.
            verbose (int): Compatibility parameter required by ``utils.op_eval``.
                Its value has no logging side effect in this adapter.

        Returns:
            np.ndarray: Float32 predictions in scaled target space with shape
            ``(samples, 100)``.

        Raises:
            ValueError: If ``x`` is not two-dimensional or ``batch_size`` is
                not positive.
            Exception: If TFLite invocation fails.
        """
        del verbose
        if x.ndim != 2:
            raise ValueError("TFLite OP input must have shape (samples, observed_ports).")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions = []
        for sparse_batch in iter_sparse_port_image_batches(
            observed_values=x.astype(np.float32),
            observed_indices=self.observed_indices,
            batch_size=self.inference_batch_size,
            total_ports=TOTAL_PORTS,
        ):
            valid_count = len(sparse_batch)
            if valid_count < self.inference_batch_size:
                padded_batch = np.zeros(
                    (self.inference_batch_size, 32, 32, 3),
                    dtype=np.float32,
                )
                padded_batch[:valid_count] = sparse_batch
            else:
                padded_batch = sparse_batch
            self.interpreter.set_tensor(
                self.input_detail["index"],
                self._quantize_input(padded_batch),
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_detail["index"])
            predictions.append(self._dequantize_output(output)[:valid_count])
        return np.concatenate(predictions, axis=0).astype(np.float32)

    def _quantize_input(self, input_sample: np.ndarray) -> np.ndarray:
        """Convert a sparse-image batch to the interpreter representation.

        Args:
            input_sample (np.ndarray): Float32 sparse images with shape
                ``(256, 32, 32, 3)``.

        Returns:
            np.ndarray: Input tensor in the model's declared dtype.

        Raises:
            ValueError: If the TFLite input dtype is unsupported.
        """
        input_dtype = np.dtype(self.input_detail["dtype"])
        if input_dtype == np.dtype(np.float32):
            return input_sample.astype(np.float32)
        input_scale, input_zero_point = self.input_detail["quantization"]
        quantized_input = np.round(input_sample / input_scale + input_zero_point)
        if input_dtype == np.dtype(np.int8):
            quantized_input = np.clip(quantized_input, -128, 127)
        elif input_dtype == np.dtype(np.uint8):
            quantized_input = np.clip(quantized_input, 0, 255)
        else:
            raise ValueError(f"Unsupported TFLite input dtype: {input_dtype}.")
        return quantized_input.astype(input_dtype)

    def _dequantize_output(self, output_sample: np.ndarray) -> np.ndarray:
        """Convert one raw TFLite output batch to scaled float values.

        Args:
            output_sample (np.ndarray): Raw output tensor with shape
                ``(256, 100)`` returned by the allocated interpreter.

        Returns:
            np.ndarray: Float32 values in the target scaler's feature space.

        Raises:
            ValueError: If the TFLite output dtype is unsupported.
        """
        output_dtype = np.dtype(self.output_detail["dtype"])
        if output_dtype == np.dtype(np.float32):
            return output_sample.astype(np.float32)
        if output_dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
            raise ValueError(f"Unsupported TFLite output dtype: {output_dtype}.")
        output_scale, output_zero_point = self.output_detail["quantization"]
        return (output_sample.astype(np.float32) - output_zero_point) * output_scale


def _is_non_empty_file(path: Path) -> bool:
    """Return whether a path is a regular file containing at least one byte.

    Args:
        path (Path): Artifact or output path to inspect.

    Returns:
        bool: ``True`` only for non-empty regular files.

    Raises:
        OSError: If filesystem metadata cannot be read.
    """
    return path.is_file() and path.stat().st_size > 0


def _load_retrain_config(run_dir: Path) -> dict[str, Any]:
    """Load and validate the saved metadata for one retraining case.

    Args:
        run_dir (Path): Completed retraining-case directory containing
            ``model_training/args/mobilenet_config.json``.

    Returns:
        dict[str, Any]: Validated run metadata used to reconstruct the held-out
        split and sparse observed-port inputs.

    Raises:
        FileNotFoundError: If required metadata is missing.
        ValueError: If metadata is empty, invalid JSON, or violates the retrain
            evaluation contract.
    """
    config_path = run_dir / "model_training" / "args" / "mobilenet_config.json"
    if not _is_non_empty_file(config_path):
        raise FileNotFoundError(f"Expected non-empty retrain metadata at {config_path}.")
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {config_path}: {error}") from error

    required_keys = {
        "data_seed",
        "input_shape",
        "num_observed_ports",
        "observed_indices",
        "op_evaluation",
        "split_mode",
        "total_ports",
        "train_regime_labels",
    }
    missing_keys = sorted(required_keys.difference(config_payload))
    if missing_keys:
        raise ValueError(f"Retrain metadata at {config_path} is missing keys: {missing_keys}.")
    if config_payload["input_shape"] != [32, 32, 3]:
        raise ValueError(f"Unexpected retrain input_shape: {config_payload['input_shape']!r}.")
    if int(config_payload["total_ports"]) != TOTAL_PORTS:
        raise ValueError(f"Retrain total_ports must be {TOTAL_PORTS}.")
    if config_payload["split_mode"] != "60_20_20":
        raise ValueError("Retrain OP evaluation only supports split_mode '60_20_20'.")
    if config_payload["op_evaluation"] != "separate_pass_required":
        raise ValueError("Retrain metadata does not require a separate OP evaluation pass.")

    num_observed_ports = int(config_payload["num_observed_ports"])
    observed_indices = [int(index) for index in config_payload["observed_indices"]]
    if len(observed_indices) != num_observed_ports:
        raise ValueError("observed_indices length does not match num_observed_ports.")
    if len(set(observed_indices)) != len(observed_indices):
        raise ValueError("observed_indices must be unique.")
    if any(index < 0 or index >= TOTAL_PORTS for index in observed_indices):
        raise ValueError(f"observed_indices must be within [0, {TOTAL_PORTS}).")
    if not config_payload["train_regime_labels"]:
        raise ValueError("train_regime_labels must contain at least one regime.")
    return config_payload


def _get_artifact_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return FP32, strict-INT8, and scaler artifact paths for one case.

    Args:
        run_dir (Path): Completed retraining-case directory.

    Returns:
        tuple[Path, Path, Path, Path]: FP32 Keras model, strict-INT8 TFLite
        model, input scaler, and target scaler paths in that order.

    Raises:
        None.
    """
    artifact_root = run_dir / "model_training"
    return (
        artifact_root / "model" / "model.keras",
        artifact_root / "tflite" / "model_int8.tflite",
        artifact_root / "scaler" / "scaler_X.pkl",
        artifact_root / "scaler" / "scaler_y.pkl",
    )


def _load_models_and_scalers(
    keras_path: Path,
    tflite_path: Path,
    scaler_x_path: Path,
    scaler_y_path: Path,
    observed_indices: np.ndarray,
) -> tuple[object, TFLitePredictAdapter, object, object]:
    """Load the two deployed artifacts and fitted identity scalers.

    Args:
        keras_path (Path): Non-empty FP32 Keras model path.
        tflite_path (Path): Non-empty strict-INT8 TFLite model path.
        scaler_x_path (Path): Non-empty fitted input-scaler pickle path.
        scaler_y_path (Path): Non-empty fitted target-scaler pickle path.
        observed_indices (np.ndarray): Zero-based sparse image locations.

    Returns:
        tuple[object, TFLitePredictAdapter, object, object]: Keras model, TFLite
        adapter, input scaler, and target scaler.

    Raises:
        FileNotFoundError: If any model or scaler artifact is missing or empty.
        Exception: If TensorFlow or pickle cannot deserialize an artifact.
    """
    for artifact_path in (keras_path, tflite_path, scaler_x_path, scaler_y_path):
        if not _is_non_empty_file(artifact_path):
            raise FileNotFoundError(f"Expected non-empty retrain artifact at {artifact_path}.")
    keras_model = tf.keras.models.load_model(
        str(keras_path),
        custom_objects=_SAFE_SELECTOR_CUSTOM_OBJECTS,
        compile=False,
    )
    with scaler_x_path.open("rb") as file_obj:
        scaler_x = pickle.load(file_obj)
    with scaler_y_path.open("rb") as file_obj:
        scaler_y = pickle.load(file_obj)
    return keras_model, TFLitePredictAdapter(tflite_path, observed_indices), scaler_x, scaler_y


def _load_test_parts(config_payload: dict[str, Any]) -> tuple[list[str], tuple[object, ...]]:
    """Reconstruct the held-out 60/20/20 data aligned with retrain metadata.

    Args:
        config_payload (dict[str, Any]): Validated retrain metadata containing
            ``data_seed`` and ordered ``train_regime_labels``.

    Returns:
        tuple[list[str], tuple[object, ...]]: Ordered regime labels and their
        aligned held-out test subsets.

    Raises:
        FileNotFoundError: If the KMS data cache cannot be loaded.
        ValueError: If the cached regime count does not match the saved metadata.
        OSError: If a cache file cannot be memory-mapped.
    """
    regime_labels = list(config_payload["train_regime_labels"])
    cached_regimes = load_raw_kms_cache(
        common_path=None,
        regime_labels=tuple(regime_labels),
    )
    rng = np.random.default_rng(int(config_payload["data_seed"]))
    test_parts = []
    for cached_regime in cached_regimes:
        # Reproduce the training splitter's per-regime permutation exactly,
        # while materializing only the held-out 20% needed by this evaluator.
        sample_count = len(cached_regime)
        train_count = int(np.floor(sample_count * 0.6))
        val_count = int(np.floor(sample_count * 0.2))
        permutation = rng.permutation(sample_count)
        test_indices = permutation[train_count + val_count :]
        test_parts.append(np.asarray(cached_regime[test_indices], dtype=np.float32))
    test_parts = tuple(test_parts)
    if len(test_parts) != len(regime_labels):
        raise ValueError(
            "Held-out split part count does not match retrain train_regime_labels: "
            f"{len(test_parts)} != {len(regime_labels)}."
        )
    return regime_labels, test_parts


def _is_op_eval_complete(output_dir: Path, regime_labels: list[str]) -> bool:
    """Return whether every requested retrain OP output is present.

    Args:
        output_dir (Path): Threshold-specific output directory for one retrain case.
        regime_labels (list[str]): Held-out regimes expected for both FP32 and
            strict-INT8 artifacts.

    Returns:
        bool: ``True`` only when metadata, combined results, and every per-regime
        metrics and sampled-prediction CSV are non-empty.

    Raises:
        OSError: If filesystem metadata cannot be read.
    """
    required_paths = [
        output_dir / "op_eval_config.json",
        output_dir / "op_results_by_artifact_split_and_regime.csv",
    ]
    for artifact_type in ("fp32_keras", "int8_tflite"):
        for regime_label in regime_labels:
            regime_dir = output_dir / artifact_type / "test" / regime_label
            required_paths.extend(
                (
                    regime_dir / "op_results_by_regime.csv",
                    regime_dir / "model_predictions_by_sample.csv",
                )
            )
    return all(_is_non_empty_file(path) for path in required_paths)


def _evaluate_artifact(
    model: object,
    artifact_type: str,
    output_dir: Path,
    regime_labels: list[str],
    test_parts: tuple[object, ...],
    scaler_x: object,
    scaler_y: object,
    num_observed_ports: int,
    observed_indices: np.ndarray,
    threshold: float,
) -> list[pd.DataFrame]:
    """Evaluate one artifact over every held-out regime and write its outputs.

    Args:
        model (object): FP32 Keras adapter or strict-INT8 TFLite adapter exposing
            ``predict`` on compact scaled observed-port values.
        artifact_type (str): Output artifact label, either ``"fp32_keras"`` or
            ``"int8_tflite"``. It determines result labels and output location.
        output_dir (Path): Threshold-specific root for the current retrain case.
        regime_labels (list[str]): Ordered regime labels aligned with ``test_parts``.
        test_parts (tuple[object, ...]): Held-out data arrays aligned with labels.
        scaler_x (object): Train-fitted input transformer.
        scaler_y (object): Train-fitted target transformer.
        num_observed_ports (int): Number of compact observed-port input columns.
        observed_indices (np.ndarray): Policy-specific zero-based physical port
            indices used for model inputs and the observed-only reference.
        threshold (float): OP threshold evaluated for every held-out regime.

    Returns:
        list[pd.DataFrame]: One labelled result frame per held-out regime.

    Raises:
        ValueError: If OP evaluation rejects the model, scalers, or data.
        OSError: If result output paths cannot be written.
    """
    frames = []
    for regime_index, (regime_label, test_part) in enumerate(zip(regime_labels, test_parts), 1):
        print(
            f"{BLUE}[{artifact_type}] [Regime {regime_index}/{len(regime_labels)}] "
            f"Evaluating held-out regime={regime_label}.{RESET}"
        )
        results_by_threshold, predictions_df = evaluate_op_per_regime_for_thresholds(
            model=model,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            data=test_part,
            num_observed_ports=num_observed_ports,
            observed_indices=observed_indices,
            total_ports=TOTAL_PORTS,
            thresholds=[threshold],
            snr_linear=SNR_LINEAR,
            batch_size=BATCH_SIZE,
        )
        results_df = results_by_threshold[threshold]
        results_df.insert(0, "artifact_type", artifact_type)
        results_df.insert(1, "split", "test")
        results_df.insert(2, "regime", regime_label)
        save_op_outputs(
            results_df=results_df,
            predictions_df=predictions_df,
            output_dir=output_dir / artifact_type / "test" / regime_label,
            max_samples=100,
        )
        frames.append(results_df)
    return frames


def run_op_eval_retrain_for_one_dir(
    run_dir: Path,
    thresholds: list[float] | tuple[float, ...] | None = None,
) -> dict[float, Path]:
    """Run held-out FP32 and strict-INT8 OP evaluation for one retrain case.

    Args:
        run_dir (Path): Completed all-regimes retraining-case directory.
        thresholds (list[float] | tuple[float, ...] | None): OP thresholds to
            evaluate. ``None`` uses ``THRESHOLD``. Each value creates a separate
            output directory and is skipped only when every expected result exists.

    Returns:
        dict[float, Path]: Threshold-to-combined-results CSV path mapping.

    Raises:
        FileNotFoundError: If required metadata or trained artifacts are missing.
        ValueError: If the retrain contract, model input, scaler width, or
            threshold list is invalid.
        OSError: If OP outputs cannot be written.
    """
    threshold_values = [float(value) for value in (THRESHOLD if thresholds is None else thresholds)]
    if not threshold_values:
        raise ValueError("thresholds must contain at least one value.")
    if len(set(threshold_values)) != len(threshold_values):
        raise ValueError("thresholds must not contain duplicate values.")

    config_payload = _load_retrain_config(run_dir)
    regime_labels, test_parts = _load_test_parts(config_payload)
    combined_paths = {
        threshold: run_dir / f"op_eval_t{threshold:.8g}" / "op_results_by_artifact_split_and_regime.csv"
        for threshold in threshold_values
    }
    thresholds_to_evaluate = [
        threshold
        for threshold in threshold_values
        if not SKIP_IF_OP_EVAL_EXISTS
        or not _is_op_eval_complete(run_dir / f"op_eval_t{threshold:.8g}", regime_labels)
    ]
    if not thresholds_to_evaluate:
        for threshold in threshold_values:
            print(f"{BLUE}Skipping complete retrain OP evaluation at {combined_paths[threshold].parent}.{RESET}")
        return combined_paths

    observed_indices = np.asarray(config_payload["observed_indices"], dtype=np.int64)
    num_observed_ports = int(config_payload["num_observed_ports"])
    keras_path, tflite_path, scaler_x_path, scaler_y_path = _get_artifact_paths(run_dir)
    keras_model, tflite_model, scaler_x, scaler_y = _load_models_and_scalers(
        keras_path,
        tflite_path,
        scaler_x_path,
        scaler_y_path,
        observed_indices,
    )
    if keras_model.input_shape != (None, 32, 32, 3):
        raise ValueError(f"Unexpected FP32 model input shape: {keras_model.input_shape!r}.")
    if int(getattr(scaler_x, "n_features_in_", -1)) != num_observed_ports:
        raise ValueError("Input scaler feature count does not match num_observed_ports.")

    sparse_keras_model = SparsePortImagePredictAdapter(
        model=keras_model,
        observed_indices=observed_indices,
        total_ports=TOTAL_PORTS,
    )
    for threshold in thresholds_to_evaluate:
        output_dir = combined_paths[threshold].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath("op_eval_config.json").write_text(
            json.dumps(
                {
                    "retrain_run_dir": str(run_dir),
                    "model_path": str(keras_path),
                    "tflite_model_path": str(tflite_path),
                    "scaler_x_path": str(scaler_x_path),
                    "scaler_y_path": str(scaler_y_path),
                    "data_seed": int(config_payload["data_seed"]),
                    "threshold": threshold,
                    "snr_linear": SNR_LINEAR,
                    "batch_size": BATCH_SIZE,
                    "tflite_batch_size": TFLITE_BATCH_SIZE,
                    "observed_indices": observed_indices.tolist(),
                    "enabled_split_names": ["test"],
                    "regime_labels": regime_labels,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        frames = _evaluate_artifact(
            sparse_keras_model,
            "fp32_keras",
            output_dir,
            regime_labels,
            test_parts,
            scaler_x,
            scaler_y,
            num_observed_ports,
            observed_indices,
            threshold,
        )
        frames.extend(
            _evaluate_artifact(
                tflite_model,
                "int8_tflite",
                output_dir,
                regime_labels,
                test_parts,
                scaler_x,
                scaler_y,
                num_observed_ports,
                observed_indices,
                threshold,
            )
        )
        pd.concat(frames, axis=0, ignore_index=True).to_csv(combined_paths[threshold], index=False)
    return combined_paths


def run_op_eval_retrain(
    retrain_run_dirs: list[Path] | None = None,
    root_search_dir: Path | None = ROOT_SEARCH_DIR,
) -> dict[Path, dict[float, Path]]:
    """Run held-out OP evaluation for every discovered or explicit retrain case.

    Args:
        retrain_run_dirs (list[Path] | None): Explicit completed retraining-case
            directories, used only when ``root_search_dir`` is ``None``.
        root_search_dir (Path | None): Retrain results root containing ``seed_*``
            directories and ``train_nas_top1_*_ports`` cases. When supplied,
            only completed cases matching that layout are evaluated.

    Returns:
        dict[Path, dict[float, Path]]: Per-case mappings from each evaluated
        threshold to its combined held-out results CSV.

    Raises:
        ValueError: If no completed cases are provided or discovered.
        FileNotFoundError: If a discovered completed case lacks required artifacts.
        OSError: If results cannot be written.
    """
    if root_search_dir is None:
        run_dirs = RETRAIN_RUN_DIR if retrain_run_dirs is None else retrain_run_dirs
    elif not root_search_dir.is_dir():
        raise FileNotFoundError(f"Expected retrain results directory at {root_search_dir}.")
    else:
        run_dirs = sorted(
            run_dir
            for seed_dir in root_search_dir.glob("seed_*")
            if seed_dir.is_dir()
            for run_dir in seed_dir.glob("train_nas_top1_*_ports")
            if run_dir.is_dir() and _is_non_empty_file(run_dir / "completion.json")
        )
    if not run_dirs:
        raise ValueError("Must provide at least one completed retrain run directory.")

    outputs_by_run = {}
    for run_index, run_dir in enumerate(run_dirs, 1):
        print(f"{BLUE}[Run {run_index}/{len(run_dirs)}] Evaluating {run_dir}.{RESET}")
        outputs_by_run[run_dir] = run_op_eval_retrain_for_one_dir(run_dir)
    print(f"{GREEN}Retrain held-out OP evaluation finished successfully.{RESET}")
    return outputs_by_run


if __name__ == "__main__":
    run_op_eval_retrain()
