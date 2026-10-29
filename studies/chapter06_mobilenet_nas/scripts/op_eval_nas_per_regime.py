"""Run OP evaluation for ranked NAS PTQ MobileNetV1 per-regime results."""

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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/op_eval_nas_per_regime.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mobilenet.keras.sparse_input import SparsePortImagePredictAdapter, iter_sparse_port_image_batches
from utils.constants import BLUE, GREEN, RESET

# Default to the sibling runs directory so the script can discover completed NAS studies automatically.
ROOT_SEARCH_DIR: Path | None = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/nas_ptq_mobv3_per_regime'
# Keep an explicit run list available for callers that disable directory scanning.
NAS_RUN_DIR: list[Path] = []

TOTAL_PORTS = 100
THRESHOLD = [1.1885398626327515]
TOP_K = 1
SNR_LINEAR = 1.0
BATCH_SIZE = 256

EVALUATE_TRAIN_SPLIT = False
EVALUATE_VAL_SPLIT = True
EVALUATE_TEST_SPLIT = True
SKIP_IF_OP_EVAL_EXISTS = True


@dataclass(frozen=True)
class RankedTrial:
    """Ranked Optuna trial summary discovered from ``top_*_trial.txt``."""

    rank: int
    trial_id: int
    summary_path: Path


@dataclass(frozen=True)
class RankedTrialArtifacts:
    """Artifact paths required to evaluate one ranked NAS trial."""

    ranked_trial: RankedTrial
    keras_model_path: Path
    tflite_model_path: Path


class TFLitePredictAdapter:
    """Expose strict INT8 TFLite inference through a Keras-like API."""

    def __init__(self, model_path: Path, observed_indices: np.ndarray) -> None:
        """Load a TFLite model and prepare it for repeated OP predictions.

        Args:
            model_path (Path): Path to a strict INT8 TFLite flatbuffer. The
                model is loaded once, tensors are allocated once, and the same
                interpreter is reused for every sample passed to ``predict``.
            observed_indices (np.ndarray): Physical observed-port indices used
                to rebuild sparse ``32x32x3`` input images from flat scaled
                observed values passed by ``utils.op_eval``.

        Returns:
            None: Initializes interpreter state on the adapter instance.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ValueError: If the model does not expose supported input/output
                dtypes or has zero quantization scales.
            Exception: Propagates TensorFlow Lite allocation failures.

        Examples:
            >>> isinstance(TFLitePredictAdapter, type)
            True
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Expected TFLite model not found at {model_path}.")

        import tensorflow as tf

        self.model_path = model_path
        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        input_scale, _ = self.input_detail["quantization"]
        output_scale, _ = self.output_detail["quantization"]
        if input_scale == 0 or output_scale == 0:
            raise ValueError("Strict INT8 TFLite model must expose non-zero input and output scales.")

        input_dtype = np.dtype(self.input_detail["dtype"])
        output_dtype = np.dtype(self.output_detail["dtype"])
        supported_dtypes = (np.dtype(np.float32), np.dtype(np.int8), np.dtype(np.uint8))
        if input_dtype not in supported_dtypes:
            raise ValueError(f"Unsupported TFLite input dtype: {input_dtype}.")
        if output_dtype not in supported_dtypes:
            raise ValueError(f"Unsupported TFLite output dtype: {output_dtype}.")

    def predict(self, x: np.ndarray, batch_size: int = 64, verbose: int = 0) -> np.ndarray:
        """Run TFLite predictions using the Keras ``predict`` call shape.

        Args:
            x (np.ndarray): Scaled input matrix with shape
                ``(n_samples, n_observed_ports)``. Values must already be
                transformed by the train-fitted input scaler.
            batch_size (int): Compatibility argument accepted by
                ``utils.op_eval``. This controls how many compact rows are
                converted to sparse images at once. The TFLite interpreter is
                still invoked sample-by-sample because strict INT8 flatbuffers
                in this study expose fixed batch size ``1``.
            verbose (int): Compatibility argument accepted by ``utils.op_eval``.
                ``0`` and non-zero values behave the same because this adapter
                does not print per-sample progress.

        Returns:
            np.ndarray: Dequantized predictions in scaled target space with
            shape ``(n_samples, total_ports)``.

        Raises:
            ValueError: If ``x`` is not a 2D matrix or cannot be converted to
                the sparse image shape exposed by the TFLite model.
            Exception: Propagates interpreter invocation failures.

        Examples:
            >>> hasattr(TFLitePredictAdapter, "predict")
            True
        """
        if x.ndim != 2:
            raise ValueError("TFLite OP input must have shape (n_samples, n_observed_ports).")

        expected_shape = np.asarray(self.input_detail["shape"], dtype=np.int64)
        if expected_shape.tolist() != [1, 32, 32, 3]:
            raise ValueError(f"Unsupported TFLite input shape: {expected_shape.tolist()}.")

        predictions = []
        for sparse_batch in iter_sparse_port_image_batches(
            observed_values=x.astype(np.float32),
            observed_indices=self.observed_indices,
            batch_size=batch_size,
            total_ports=TOTAL_PORTS,
        ):
            for sample in sparse_batch:
                # TFLite calibration stores the quantization contract in the
                # model, so each temporary sparse image is quantized exactly
                # before invocation.
                model_input = self._quantize_input(sample[np.newaxis, ...])
                self.interpreter.set_tensor(self.input_detail["index"], model_input)
                self.interpreter.invoke()
                raw_output = self.interpreter.get_tensor(self.output_detail["index"])
                predictions.append(self._dequantize_output(raw_output).reshape(-1))

        return np.asarray(predictions, dtype=np.float32)

    def _quantize_input(self, input_sample: np.ndarray) -> np.ndarray:
        """Quantize one scaled input sample for the loaded TFLite model.

        Args:
            input_sample (np.ndarray): Scaled float32 input sample with shape
                matching the interpreter input shape.

        Returns:
            np.ndarray: Input sample converted to the exact dtype expected by
            the TFLite interpreter.

        Raises:
            ValueError: If the model input dtype is unsupported.
        """
        input_dtype = np.dtype(self.input_detail["dtype"])
        input_scale, input_zero_point = self.input_detail["quantization"]
        if input_dtype == np.dtype(np.float32):
            return input_sample.astype(np.float32)
        quantized_input = np.round(input_sample / input_scale + input_zero_point)
        if input_dtype == np.dtype(np.int8):
            quantized_input = np.clip(quantized_input, -128, 127)
        elif input_dtype == np.dtype(np.uint8):
            quantized_input = np.clip(quantized_input, 0, 255)
        else:
            raise ValueError(f"Unsupported TFLite input dtype: {input_dtype}.")
        return quantized_input.astype(input_dtype)

    def _dequantize_output(self, output_sample: np.ndarray) -> np.ndarray:
        """Dequantize one raw output sample into scaled target space.

        Args:
            output_sample (np.ndarray): Raw interpreter output tensor.

        Returns:
            np.ndarray: Float32 output in the same scaled target space used by
            the train-fitted target scaler.

        Raises:
            ValueError: If the model output dtype is unsupported.
        """
        output_dtype = np.dtype(self.output_detail["dtype"])
        output_scale, output_zero_point = self.output_detail["quantization"]
        if output_dtype == np.dtype(np.float32):
            return output_sample.astype(np.float32)
        if output_dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
            raise ValueError(f"Unsupported TFLite output dtype: {output_dtype}.")
        return (output_sample.astype(np.float32) - output_zero_point) * output_scale


def _get_nas_config_path(nas_run_dir: Path) -> Path:
    """Return the strict NAS metadata path for one run.

    Args:
        nas_run_dir (Path): Top-level NAS run directory containing an
            ``optuna_study`` directory.

    Returns:
        Path: Path to ``optuna_study/config.json``.

    Raises:
        None.
    """
    return nas_run_dir / "optuna_study" / "config.json"


def _load_nas_config(nas_run_dir: Path) -> dict[str, Any]:
    """Load and validate NAS metadata required for OP evaluation.

    Args:
        nas_run_dir (Path): Top-level NAS run directory. The evaluator does not
            infer regime, split, or artifact configuration from the directory
            name; it trusts only ``optuna_study/config.json``.

    Returns:
        dict[str, Any]: Parsed and validated config payload.

    Raises:
        FileNotFoundError: If ``optuna_study/config.json`` is missing.
        ValueError: If the JSON is malformed, missing required keys, or names
            an unsupported split mode.
    """
    config_path = _get_nas_config_path(nas_run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Expected NAS metadata not found at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as config_file:
        try:
            config_payload = json.load(config_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {config_path}: {error}") from error

    required_keys = {
        "regime_label",
        "observed_ports_budget",
        "observed_indices",
        "total_features",
        "data_seed",
        "split_mode",
        "train_regime_labels",
        "top_k",
        "rank_key",
        "order",
        "artifact_subdirs",
    }
    missing_keys = sorted(required_keys.difference(config_payload))
    if missing_keys:
        raise ValueError(f"NAS metadata at {config_path} is missing required keys: {missing_keys}.")

    if config_payload["split_mode"] != "60_20_20":
        raise ValueError("op_eval_nas_per_regime supports only split_mode='60_20_20'.")
    if int(config_payload["observed_ports_budget"]) <= 0:
        raise ValueError("observed_ports_budget must be positive.")
    if int(config_payload["total_features"]) != TOTAL_PORTS:
        raise ValueError(
            f"NAS metadata total_features={config_payload['total_features']} does not match {TOTAL_PORTS}."
        )
    artifact_subdirs = config_payload["artifact_subdirs"]
    required_subdir_keys = {"args", "model", "tflite", "scaler"}
    if not isinstance(artifact_subdirs, dict):
        raise ValueError("artifact_subdirs must be a dictionary.")
    missing_subdir_keys = sorted(required_subdir_keys.difference(artifact_subdirs))
    if missing_subdir_keys:
        raise ValueError(f"artifact_subdirs is missing required keys: {missing_subdir_keys}.")

    return config_payload


def _parse_ranked_trial_file(summary_path: Path) -> RankedTrial:
    """Parse one ``top_*_trial.txt`` summary into a ranked trial record.

    Args:
        summary_path (Path): Path to an Optuna summary file containing ``Rank``
            and ``Trial ID`` header lines.

    Returns:
        RankedTrial: Parsed rank, trial ID, and original summary path.

    Raises:
        FileNotFoundError: If ``summary_path`` does not exist.
        ValueError: If required fields are missing or disagree with the file
            name rank.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected top-trial summary not found at {summary_path}.")

    rank = None
    trial_id = None
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Rank:"):
            rank = int(line.split(":", 1)[1].strip())
        elif line.startswith("Trial ID:"):
            trial_id = int(line.split(":", 1)[1].strip())
        if rank is not None and trial_id is not None:
            break

    if rank is None or trial_id is None:
        raise ValueError(f"Top-trial summary {summary_path} must contain Rank and Trial ID lines.")

    match = re.fullmatch(r"top_(\d+)_trial\.txt", summary_path.name)
    if match and int(match.group(1)) != rank:
        raise ValueError(f"Top-trial filename rank and file Rank disagree in {summary_path}.")

    return RankedTrial(rank=rank, trial_id=trial_id, summary_path=summary_path)


def _discover_ranked_trials(study_dir: Path, args_subdir: str) -> list[RankedTrial]:
    """Discover and parse all ranked top-trial summaries for a study.

    Args:
        study_dir (Path): Path to the ``optuna_study`` directory.
        args_subdir (str): Relative subdirectory containing
            ``top_*_trial.txt`` files. The value comes from
            ``config.json["artifact_subdirs"]["args"]``.

    Returns:
        list[RankedTrial]: Ranked trials sorted by ascending rank.

    Raises:
        ValueError: If no top-trial summaries are found or duplicate ranks are
            present.
    """
    args_dir = study_dir / args_subdir
    ranked_trials = [
        _parse_ranked_trial_file(summary_path) for summary_path in sorted(args_dir.glob("top_*_trial.txt"))
    ]
    if not ranked_trials:
        raise ValueError(f"No top-trial summaries found in {args_dir}.")

    ranks = [trial.rank for trial in ranked_trials]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"Duplicate top-trial ranks found in {args_dir}.")

    return sorted(ranked_trials, key=lambda trial: trial.rank)


def _resolve_ranked_trial_artifacts(
    study_dir: Path,
    ranked_trial: RankedTrial,
    artifact_subdirs: dict[str, str] | None = None,
) -> RankedTrialArtifacts:
    """Build expected FP32 and INT8 artifact paths for one ranked trial.

    Args:
        study_dir (Path): Path to the ``optuna_study`` directory.
        ranked_trial (RankedTrial): Rank and trial ID parsed from a top-trial
            summary file.
        artifact_subdirs (dict[str, str] | None): Optional subdirectory mapping
            from ``config.json``. If ``None``, the standard NAS layout
            ``model`` and ``tflite`` is used. Providing a mapping changes only
            path resolution; missing files are still rejected by
            ``_validate_ranked_trial_artifacts``.

    Returns:
        RankedTrialArtifacts: Expected `.keras` and `.tflite` paths.

    Raises:
        KeyError: If a provided ``artifact_subdirs`` mapping omits ``model`` or
            ``tflite``.
    """
    subdirs = artifact_subdirs or {"model": "model", "tflite": "tflite"}
    artifact_stem = f"top_{ranked_trial.rank}_trial_{ranked_trial.trial_id}"
    return RankedTrialArtifacts(
        ranked_trial=ranked_trial,
        keras_model_path=study_dir / subdirs["model"] / f"{artifact_stem}.keras",
        tflite_model_path=study_dir / subdirs["tflite"] / f"{artifact_stem}.tflite",
    )


def _validate_ranked_trial_artifacts(artifact_paths: RankedTrialArtifacts) -> None:
    """Validate both FP32 Keras and strict INT8 TFLite artifacts exist.

    Args:
        artifact_paths (RankedTrialArtifacts): Artifact path record returned by
            ``_resolve_ranked_trial_artifacts``.

    Returns:
        None: Validation succeeds only when both artifacts exist.

    Raises:
        FileNotFoundError: If either required artifact is missing.
    """
    if not artifact_paths.keras_model_path.exists():
        raise FileNotFoundError(f"Expected Keras model not found at {artifact_paths.keras_model_path}.")
    if not artifact_paths.tflite_model_path.exists():
        raise FileNotFoundError(f"Expected TFLite model not found at {artifact_paths.tflite_model_path}.")


def _load_scalers(scaler_path: Path) -> tuple[object, object]:
    """Load the train-fitted input and target scalers for a NAS study.

    Args:
        scaler_path (Path): Path to ``scaler_xy.pkl`` containing a dictionary
            with ``x_scaler`` and ``y_scaler`` entries.

    Returns:
        tuple[object, object]: Loaded ``(x_scaler, y_scaler)`` objects.

    Raises:
        FileNotFoundError: If ``scaler_path`` does not exist.
        KeyError: If either scaler key is missing.
        OSError: If the file cannot be read.
        pickle.UnpicklingError: If the pickle payload is invalid.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Expected scaler_xy.pkl not found at {scaler_path}.")

    with open(scaler_path, "rb") as scaler_file:
        scaler_payload = pickle.load(scaler_file)

    if "x_scaler" not in scaler_payload:
        raise KeyError("Scaler file is missing the 'x_scaler' key.")
    if "y_scaler" not in scaler_payload:
        raise KeyError("Scaler file is missing the 'y_scaler' key.")
    return scaler_payload["x_scaler"], scaler_payload["y_scaler"]


def _get_enabled_split_names() -> tuple[str, ...]:
    """Return enabled OP-evaluation split names in fixed order.

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
        raise ValueError("At least one OP split toggle must be enabled.")
    return tuple(enabled_split_names)


def _is_op_eval_complete(
    output_dir: Path,
    ranked_trials: list[RankedTrial],
    split_names: tuple[str, ...],
    regime_labels: list[str],
) -> bool:
    """Return whether all requested NAS OP outputs are complete.

    Args:
        output_dir (Path): Threshold-specific NAS OP output directory.
        ranked_trials (list[RankedTrial]): Top-ranked trials expected in the
            output tree.
        split_names (tuple[str, ...]): Split names enabled for the current run.
        regime_labels (list[str]): Regimes expected in every enabled split.

    Returns:
        bool: ``True`` only when every aggregate and per-regime output file is
            present and non-empty for both model artifact types.

    Raises:
        OSError: If an existing output file cannot be inspected.
    """
    required_paths = [
        output_dir / "op_eval_config.json",
        output_dir / "op_results_by_artifact_trial_split_and_regime.csv",
    ]
    for artifact_type in ("fp32_keras", "int8_tflite"):
        for ranked_trial in ranked_trials:
            trial_tag = f"top_{ranked_trial.rank}_trial_{ranked_trial.trial_id}"
            trial_dir = output_dir / artifact_type / trial_tag
            for split_name in split_names:
                for regime_label in regime_labels:
                    regime_dir = trial_dir / split_name / regime_label
                    required_paths.extend(
                        (
                            regime_dir / "op_results_by_regime.csv",
                            regime_dir / "model_predictions_by_sample.csv",
                        )
                    )

    return all(path.is_file() and path.stat().st_size > 0 for path in required_paths)


def _load_split_parts(config_payload: dict[str, Any]) -> tuple[list[str], dict[str, tuple[object, ...]]]:
    """Load train, validation, and test OP split parts for a NAS config.

    Args:
        config_payload (dict[str, Any]): Validated config loaded by
            ``_load_nas_config``. The function uses ``train_regime_labels`` and
            ``data_seed`` to reconstruct the same 60/20/20 split policy used by
            the per-regime NAS training script.

    Returns:
        tuple[list[str], dict[str, tuple[object, ...]]]: Evaluated regime labels
            and mapping from split name to per-regime arrays.

    Raises:
        FileNotFoundError: If the KMS runtime cache is missing.
        ValueError: If the regime labels or split metadata are invalid.
    """
    from utils.data import split_kms_60_20_20_cache

    regime_labels = list(config_payload["train_regime_labels"])
    if not regime_labels:
        raise ValueError("train_regime_labels must contain at least one regime.")

    print(f"{BLUE}Loading NAS OP 60/20/20 data for regimes={regime_labels}.{RESET}")
    _, _, _, train_parts, val_parts, test_parts = split_kms_60_20_20_cache(
        common_path=None,
        seed=int(config_payload["data_seed"]),
        regime_labels=tuple(regime_labels),
        samples_per_regime=None,
        verbose=False,
    )
    return regime_labels, {
        "train": train_parts,
        "val": val_parts,
        "test": test_parts,
    }


def _load_keras_model(model_path: Path) -> object:
    """Load one ranked FP32 Keras model for OP evaluation.

    Args:
        model_path (Path): Path to a saved `.keras` model artifact.

    Returns:
        object: Loaded Keras model exposing ``predict``.

    Raises:
        FileNotFoundError: If ``model_path`` is missing.
        Exception: Propagates TensorFlow model loading failures.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Expected Keras model not found at {model_path}.")
    import tensorflow as tf
    from studies.chapter06_mobilenet_nas.scripts.train import safe_selector

    del safe_selector

    return tf.keras.models.load_model(str(model_path), compile=False)


def _validate_model_input_width(model: object, expected_width: int, artifact_label: str) -> None:
    """Validate an OP model exposes the sparse MobileNet image input shape.

    Args:
        model (object): Keras model or TFLite adapter being evaluated.
        expected_width (int): Expected number of observed input ports from
            ``config.json``. It is accepted for call-site symmetry; sparse
            MobileNet artifacts expose the fixed image shape instead.
        artifact_label (str): Human-readable artifact label used in error
            messages. ``"fp32_keras"`` and ``"int8_tflite"`` use the same
            validation rule but report different failure contexts.

    Returns:
        None: Validation succeeds when the model input shape matches.

    Raises:
        ValueError: If the model input shape is unsupported or mismatched.
    """
    if isinstance(model, TFLitePredictAdapter):
        input_shape = np.asarray(model.input_detail["shape"], dtype=np.int64)
        expected_shape = [1, 32, 32, 3]
        model_shape = input_shape.tolist()
    else:
        input_shape = getattr(model, "input_shape", None)
        expected_shape = [None, 32, 32, 3]
        model_shape = list(input_shape) if isinstance(input_shape, tuple) else []

    if model_shape != expected_shape:
        raise ValueError(
            f"{artifact_label} input shape does not match sparse MobileNet config: "
            f"model exposes {model_shape}, expected {expected_shape}."
        )


def _run_op_eval_for_artifact(
    model: object,
    artifact_type: str,
    artifact_output_dir: Path,
    ranked_trial: RankedTrial,
    regime_labels: list[str],
    split_parts_map: dict[str, tuple[object, ...]],
    scaler_x: object,
    scaler_y: object,
    thresholds: list[float],
    num_observed_ports: int,
) -> dict[float, list[object]]:
    """Evaluate one model artifact across enabled splits and regimes.

    Args:
        model (object): Keras model or TFLite adapter exposing ``predict``.
        artifact_type (str): Output label for the artifact family. If
            ``"fp32_keras"``, results describe the saved floating-point model.
            If ``"int8_tflite"``, results describe the quantized deployment
            model. The value changes only output labels and directories.
        artifact_output_dir (Path): Threshold-independent output root for this
            artifact and trial.
        ranked_trial (RankedTrial): Rank and trial ID being evaluated.
        regime_labels (list[str]): Regimes aligned with each split-parts tuple.
        split_parts_map (dict[str, tuple[object, ...]]): Mapping from split
            name to per-regime data arrays returned by ``_load_split_parts``.
        scaler_x (object): Train-fitted input scaler.
        scaler_y (object): Train-fitted target scaler.
        thresholds (list[float]): OP thresholds evaluated in one prediction
            pass per split/regime.
        num_observed_ports (int): Number of observed ports expected by the
            model and input scaler.

    Returns:
        dict[float, list[object]]: Mapping from threshold to result DataFrames.
        ``object`` is used to keep pandas as a lazy runtime dependency.

    Raises:
        ValueError: If OP split settings or model outputs are invalid.
        OSError: If CSV outputs cannot be written.
    """
    from utils.op_eval import evaluate_op_per_regime_for_thresholds, save_op_outputs

    frames_by_threshold: dict[float, list[object]] = {float(threshold): [] for threshold in thresholds}
    enabled_split_names = _get_enabled_split_names()
    total_splits = len(enabled_split_names)
    for split_idx, split_name in enumerate(enabled_split_names, 1):
        split_parts = split_parts_map[split_name]
        for regime_idx, (regime_label, split_subset) in enumerate(zip(regime_labels, split_parts), 1):
            print(
                f"{BLUE}[{artifact_type}] [Trial rank {ranked_trial.rank}] "
                f"[Split {split_idx}/{total_splits}] Evaluating regime={regime_label}.{RESET}"
            )
            results_by_threshold, predictions_df = evaluate_op_per_regime_for_thresholds(
                model=model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                data=split_subset,
                num_observed_ports=num_observed_ports,
                total_ports=TOTAL_PORTS,
                thresholds=thresholds,
                snr_linear=SNR_LINEAR,
                batch_size=BATCH_SIZE,
            )
            for threshold in thresholds:
                threshold_value = float(threshold)
                threshold_dir = artifact_output_dir.parent / f"op_eval_t{threshold_value:.8g}"
                trial_tag = f"top_{ranked_trial.rank}_trial_{ranked_trial.trial_id}"
                regime_output_dir = threshold_dir / artifact_type / trial_tag / split_name / regime_label
                results_df = results_by_threshold[threshold_value]
                results_df.insert(0, "artifact_type", artifact_type)
                results_df.insert(1, "rank", int(ranked_trial.rank))
                results_df.insert(2, "trial_id", int(ranked_trial.trial_id))
                results_df.insert(3, "split", split_name)
                results_df.insert(4, "regime", regime_label)
                frames_by_threshold[threshold_value].append(results_df)
                save_op_outputs(
                    results_df=results_df,
                    predictions_df=predictions_df,
                    output_dir=regime_output_dir,
                    max_samples=100,
                )
                print(
                    f"{GREEN}[{artifact_type}] Saved rank={ranked_trial.rank}, "
                    f"split={split_name}, regime={regime_label}, threshold={threshold_value:.8g}.{RESET}"
                )

    return frames_by_threshold


def _write_threshold_config(
    output_dir: Path,
    nas_run_dir: Path,
    config_payload: dict[str, Any],
    thresholds: list[float],
    ranked_trials: list[RankedTrial],
) -> None:
    """Write OP evaluation metadata for one threshold output directory.

    Args:
        output_dir (Path): Threshold-specific output directory.
        nas_run_dir (Path): Top-level NAS run directory being evaluated.
        config_payload (dict[str, Any]): Validated NAS config payload.
        thresholds (list[float]): Thresholds requested for this evaluator run.
        ranked_trials (list[RankedTrial]): Ranked trials discovered for the
            study.

    Returns:
        None: Writes ``op_eval_config.json`` under ``output_dir``.

    Raises:
        OSError: If the output directory or config file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "nas_run_dir": str(nas_run_dir),
        "source_config": str(_get_nas_config_path(nas_run_dir)),
        "thresholds": [float(threshold) for threshold in thresholds],
        "snr_linear": float(SNR_LINEAR),
        "batch_size": int(BATCH_SIZE),
        "config": config_payload,
        "ranked_trials": [
            {
                "rank": int(ranked_trial.rank),
                "trial_id": int(ranked_trial.trial_id),
                "summary_path": str(ranked_trial.summary_path),
            }
            for ranked_trial in ranked_trials
        ],
    }
    (output_dir / "op_eval_config.json").write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_op_eval_nas_for_one_dir(
    nas_run_dir: Path,
    thresholds: list[float] | tuple[float, ...] | None = None,
) -> dict[float, Path]:
    """Run NAS OP evaluation for one completed NAS study directory.

    Args:
        nas_run_dir (Path): Top-level NAS run directory containing
            ``optuna_study/config.json`` and ranked top artifacts.
        thresholds (list[float] | tuple[float, ...] | None): Optional threshold
            list. If ``None``, the evaluator uses the module-level
            ``THRESHOLD`` constant. Providing explicit values changes only OP
            cutoff values and output directory names.

    Returns:
        dict[float, Path]: Mapping from threshold to the combined CSV path.

    Raises:
        FileNotFoundError: If config, scalers, top summaries, or ranked
            artifacts are missing.
        ValueError: If config, model, or scaler contracts are invalid.
        OSError: If outputs cannot be written.
    """
    config_payload = _load_nas_config(nas_run_dir)
    threshold_values = [
        float(threshold) for threshold in (thresholds if thresholds is not None else THRESHOLD)
    ]
    if not threshold_values:
        raise ValueError("thresholds must contain at least one threshold.")

    study_dir = nas_run_dir / "optuna_study"
    artifact_subdirs = config_payload["artifact_subdirs"]
    ranked_trials = _discover_ranked_trials(study_dir=study_dir, args_subdir=artifact_subdirs["args"])
    if TOP_K <= 0:
        raise ValueError("TOP_K must be positive.")
    ranked_trials = ranked_trials[:TOP_K]
    ranked_artifacts = [
        _resolve_ranked_trial_artifacts(study_dir, ranked_trial, artifact_subdirs)
        for ranked_trial in ranked_trials
    ]
    for artifact_paths in ranked_artifacts:
        _validate_ranked_trial_artifacts(artifact_paths)

    scaler_path = study_dir / artifact_subdirs["scaler"] / "scaler_xy.pkl"
    scaler_x, scaler_y = _load_scalers(scaler_path)
    scaler_features = int(getattr(scaler_x, "n_features_in_", -1))
    num_observed_ports = int(config_payload["observed_ports_budget"])
    observed_indices = np.asarray(config_payload["observed_indices"], dtype=np.int64)
    if observed_indices.shape[0] != num_observed_ports:
        raise ValueError(
            "observed_indices length does not match observed_ports_budget: "
            f"{observed_indices.shape[0]} != {num_observed_ports}."
        )
    if scaler_features != num_observed_ports:
        raise ValueError(
            f"Input scaler feature count does not match config: "
            f"scaler_X expects {scaler_features}, config expects {num_observed_ports}."
        )

    regime_labels, split_parts_map = _load_split_parts(config_payload)
    enabled_split_names = _get_enabled_split_names()
    thresholds_to_evaluate = [
        threshold
        for threshold in threshold_values
        if not SKIP_IF_OP_EVAL_EXISTS
        or not _is_op_eval_complete(
            output_dir=nas_run_dir / f"op_eval_t{threshold:.8g}",
            ranked_trials=ranked_trials,
            split_names=enabled_split_names,
            regime_labels=regime_labels,
        )
    ]
    for threshold in threshold_values:
        if threshold not in thresholds_to_evaluate:
            print(
                f"{BLUE}Skipping complete NAS OP evaluation at "
                f"{nas_run_dir / f'op_eval_t{threshold:.8g}'}.{RESET}"
            )

    combined_paths = {
        threshold: nas_run_dir
        / f"op_eval_t{threshold:.8g}"
        / "op_results_by_artifact_trial_split_and_regime.csv"
        for threshold in threshold_values
    }
    if not thresholds_to_evaluate:
        return combined_paths

    combined_frames_by_threshold: dict[float, list[object]] = {
        threshold: [] for threshold in thresholds_to_evaluate
    }

    for threshold in thresholds_to_evaluate:
        _write_threshold_config(
            output_dir=nas_run_dir / f"op_eval_t{threshold:.8g}",
            nas_run_dir=nas_run_dir,
            config_payload=config_payload,
            thresholds=thresholds_to_evaluate,
            ranked_trials=ranked_trials,
        )

    for artifact_paths in ranked_artifacts:
        keras_model = _load_keras_model(artifact_paths.keras_model_path)
        _validate_model_input_width(keras_model, num_observed_ports, "fp32_keras")
        keras_model = SparsePortImagePredictAdapter(
            model=keras_model,
            observed_indices=observed_indices,
            total_ports=TOTAL_PORTS,
        )
        keras_frames = _run_op_eval_for_artifact(
            model=keras_model,
            artifact_type="fp32_keras",
            artifact_output_dir=nas_run_dir / "fp32_keras",
            ranked_trial=artifact_paths.ranked_trial,
            regime_labels=regime_labels,
            split_parts_map=split_parts_map,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            thresholds=thresholds_to_evaluate,
            num_observed_ports=num_observed_ports,
        )
        tflite_model = TFLitePredictAdapter(
            artifact_paths.tflite_model_path,
            observed_indices=observed_indices,
        )
        _validate_model_input_width(tflite_model, num_observed_ports, "int8_tflite")
        tflite_frames = _run_op_eval_for_artifact(
            model=tflite_model,
            artifact_type="int8_tflite",
            artifact_output_dir=nas_run_dir / "int8_tflite",
            ranked_trial=artifact_paths.ranked_trial,
            regime_labels=regime_labels,
            split_parts_map=split_parts_map,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            thresholds=thresholds_to_evaluate,
            num_observed_ports=num_observed_ports,
        )
        for threshold in thresholds_to_evaluate:
            combined_frames_by_threshold[threshold].extend(keras_frames[threshold])
            combined_frames_by_threshold[threshold].extend(tflite_frames[threshold])

    for threshold in thresholds_to_evaluate:
        import pandas as pd

        threshold_dir = nas_run_dir / f"op_eval_t{threshold:.8g}"
        combined_path = threshold_dir / "op_results_by_artifact_trial_split_and_regime.csv"
        pd.concat(combined_frames_by_threshold[threshold], axis=0, ignore_index=True).to_csv(
            combined_path,
            index=False,
        )
    return combined_paths


def run_op_eval_nas_per_regime(
    nas_run_dirs: list[Path] | None = None,
    root_search_dir: Path | None = ROOT_SEARCH_DIR,
) -> dict[Path, dict[float, Path]]:
    """Run NAS OP evaluation for each discovered or listed NAS run.

    Args:
        nas_run_dirs (list[Path] | None): Explicit top-level NAS run
            directories. Used only when ``root_search_dir`` is ``None``.
        root_search_dir (Path | None): Directory whose direct children are NAS
            run directories. If provided, every child containing
            ``optuna_study`` is evaluated, and missing ``config.json`` files
            fail explicitly.

    Returns:
        dict[Path, dict[float, Path]]: Mapping from NAS run directory to
        threshold-specific combined CSV paths.

    Raises:
        ValueError: If no NAS run directories are provided or discovered.
        FileNotFoundError: If any discovered run is missing strict metadata or
            required top artifacts.
    """
    if root_search_dir is not None:
        dirs_to_evaluate = []
        if root_search_dir.exists():
            dirs_to_evaluate = sorted(
                child
                for child in root_search_dir.iterdir()
                if child.is_dir() and (child / "optuna_study").is_dir()
            )
    else:
        dirs_to_evaluate = nas_run_dirs if nas_run_dirs is not None else NAS_RUN_DIR

    if not dirs_to_evaluate:
        raise ValueError("Must provide at least one NAS run directory to evaluate.")

    outputs_by_run: dict[Path, dict[float, Path]] = {}
    total_dirs = len(dirs_to_evaluate)
    for dir_idx, nas_run_dir in enumerate(dirs_to_evaluate, 1):
        print(f"\n{BLUE}[Run {dir_idx}/{total_dirs}] Running NAS OP evaluation for {nas_run_dir}.{RESET}")
        outputs_by_run[nas_run_dir] = run_op_eval_nas_for_one_dir(
            nas_run_dir=nas_run_dir,
            thresholds=THRESHOLD,
        )
    return outputs_by_run


if __name__ == "__main__":
    run_op_eval_nas_per_regime()
