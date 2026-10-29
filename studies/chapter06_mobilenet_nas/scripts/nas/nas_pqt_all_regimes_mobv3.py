"""
All-regimes MobileNetV3-Small PTQ INT8 NAS study.

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev

#! CHECK COMPATIBLE TFLITE OPERATIONS:
https://ai.google.dev/edge/litert/conversion/tensorflow/op_select_allowlist
or check docs/ml/supported_select_tf_ops_tflite.md.

all_regimes:
B=2048 -> train 4102 batches, val 1368 batches
B=4096 -> train 2051 batches, val 684 batches
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
import json
from pathlib import Path
from time import perf_counter

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/nas/nas_pqt_all_regimes_mobv3.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Setup and Configuration                           #
# ———————————————————————————————————————————————————————————————————————————— #
import os

# Specify GPU to use (e.g., GPU:0, CPU:-1):
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use CUDA async allocator to reduce fragmentation:
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Allow TensorFlow to allocate GPU memory as needed:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Disable all auto-JIT clustering at the process level:
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

# Enable deterministic operations for reproducibility:
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# If it fails to determine best cudnn convolution algorithm:
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# Suppress TensorFlow C++ backend logs before TensorFlow is imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable oneDNN startup message and keep numerical behavior simpler to audit.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# %%—————————————————————————————————————————————————————————————————————————— #
#                                    Imports                                   #
# ———————————————————————————————————————————————————————————————————————————— #
from utils.imports_keras import *  # Centralized file containing all imports
from studies.chapter06_mobilenet_nas.scripts.nas.safe_selector_nas import (
    build_nas_selector_models,
    evaluate_keras_selector_op,
    evaluate_tflite_selector_diagnostics,
    evaluate_tflite_selector_op,
    prepare_selector_study_data,
)
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    compute_selection_snapshot,
)
from utils.data import KMS_REGIME_LABELS, get_observed_ports, split_kms_60_20_20_cache
from utils.qttf import (
    convert_ptq_model_to_strict_int8_tflite,
    estimate_tflite_peak_ram_bytes,
    test_strict_int8_io,
)
from utils.tflite_model_stats import set_user_attr_tflite_model_stats

# %%—————————————————————————————————————————————————————————————————————————— #
#                                Run Parameters                                #
# ———————————————————————————————————————————————————————————————————————————— #
NUM_TRIALS = 100
BATCH_SIZE = 4096

# Optional fast-test data multiplier. Use None for full train/validation data,
# or a value in (0, 1] to keep that fraction of each split.
TEST_MULTIPIER = None

# steps_per_execution can break validation when a loop has too few batches.
# Use a positive integer to keep a fixed value, or "auto" to use the top
# validate_steps_per_execution recommendation for the current train/val sizes.
# Ensure integer steps_per_execution <= min_batches across train, val, and eval loops.
# num_batches = ceil(num_samples / batch_size)
# Example: 1473 samples, batch_size=64 -> ceil(1473/64)=24 batches,
# so steps_per_execution must be <= 24.
STEPS_PER_EXECUTION = "auto"

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = True

# Task configuration:
TOTAL_FEATURES = 100
OBSERVED_PORTS_LIST = (3, 4, 5, 6, 7)
TRAIN_REGIME_LABELS = KMS_REGIME_LABELS

# Model configuration:
IMAGE_SIDE = 32
IMAGE_CHANNELS = 3
MOBILENETV3_TRUNCATION_BLOCK_NAMES = (
    "v3_block_0_project_gn",
    "v3_block_1_project_gn",
    "v3_block_2_add",
    "v3_block_3_project_gn",
    "v3_block_4_add",
    "v3_block_5_add",
    "v3_block_6_project_gn",
    "v3_block_7_add",
    "v3_block_8_project_gn",
    "v3_block_9_add",
    "v3_block_10_add",
)
ALPHA_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Training configuration:
NOT_QT_TRAINING_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
PRUNING_INTERVAL = None #! Does not support multi-objective

INT8_CALIBRATION_SAMPLE_COUNT = 256
INT8_CALIBRATION_NUM_BINS = 16
INT8_CALIBRATION_SEED = 0

POLICY = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(POLICY)
BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# Objective configuration:
# Pi Pico has 264 kB SRAM, and 2 MB on-board QSPI flash.
MAX_RAM_BYTES = 224 * 1024 # = 224 KiB
DOUBLE_CHECK_RAM = True # Check RAM budget again after training, since it can change after weight updates
PENALTY_INT8_VAL_MODEL_OP = 1e15 # Remove this value from the Pareto front afterwards
PENALTY_RAM_BYTES = 1e15 # Remove this value from the Pareto front afterwards
DIRECTION = ["minimize", "minimize"]

# Seed configuration:
SAMPLER_SEED = 0
DATA_SEED = 0
TRAIN_SEED = 0

# Final ranking configuration:
TOP_K = 30
ORDER = "ascending"
RANK_KEY = "best_qt_val_model_op"
EXTRA_ATTRS = [
    "best_not_qt_val_model_op",
    "best_qt_val_model_op",
    "quantization_op_degradation",
    "not_qt_distinct_selected_ports",
    "not_qt_permutation_change_rate",
    "qt_distinct_selected_ports",
    "qt_permutation_change_rate",
    "qt_model_size_bytes",
    "pretrain_ram_peak_bytes",
    "trained_ram_peak_bytes",
    "ram_budget_violated",
    "violating_ram_peak_bytes",
]
EXTRA_ATTRS_TFLITE = [
    "tflite_model_size",
    "tflite_peak_ram_bytes",
    "tflite_tensor_count",
    "tflite_operator_count",
    "tflite_inference_latency_mean",
    "tflite_inference_latency_peak",
]
EXTRA_ATTRS = EXTRA_ATTRS + EXTRA_ATTRS_TFLITE


# ———————————————————————————————————————————————————————————————————————————— #
#                               Utility functions                              #
# ———————————————————————————————————————————————————————————————————————————— #
def log_print(function_name: str, message: str) -> None:
    """Print a standardized log line for this script."""
    print(f"[{_COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/nas/nas_pqt_all_regimes_mobv3.py'.stem}:{function_name}] {message}")


def write_nas_ptq_config(
    run_dir: str,
    train_regime_labels: list[str] | tuple[str, ...],
    observed_ports_budget: int,
    study_data: dict[str, Any],
) -> str:
    """Write strict NAS PTQ metadata for later OP evaluation.

    Args:
        run_dir (str): Top-level NAS run directory passed to ``run_study``.
            The config is written below this directory at
            ``optuna_study/config.json`` so evaluators do not infer metadata
            from folder names.
        train_regime_labels (list[str] | tuple[str, ...]): KMS regimes used by
            this all-regimes study. The labels are serialized under
            ``train_regime_labels`` so OP evaluation can evaluate exactly the
            regimes used during NAS training.
        observed_ports_budget (int): Number of observed ports used as the model
            input width. It must match ``study_data["observed_ports_budget"]``
            and the saved model/scaler contracts.
        study_data (dict[str, Any]): Data dictionary returned by
            ``load_regimes_data``. The function reads only serializable
            metadata needed to reconstruct the OP split and validate artifact
            shapes.

    Returns:
        str: Path to the written ``config.json`` file.

    Raises:
        KeyError: If required metadata is missing from ``study_data``.
        OSError: If the config directory or file cannot be written.

    Examples:
        >>> callable(write_nas_ptq_config)
        True
    """
    selected_regime_labels = list(train_regime_labels)
    if not selected_regime_labels:
        raise ValueError("train_regime_labels must contain at least one regime.")

    config_path = Path(run_dir) / "optuna_study" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "regime_label": "all_regimes",
        "observed_ports_budget": int(observed_ports_budget),
        "observed_indices": [int(index) for index in study_data["observed_indices"]],
        "total_features": int(TOTAL_FEATURES),
        "data_seed": int(DATA_SEED),
        "split_mode": "60_20_20",
        "train_regime_labels": selected_regime_labels,
        "top_k": int(TOP_K),
        "rank_key": RANK_KEY,
        "order": ORDER,
        "schema_version": 3,
        "task_type": "safe_port_selector",
        "output_semantics": "decision_scores",
        "backbone_family": "MobileNetV3Small",
        "variant_name": "mobv3_alpha02",
        "objectives": [
            "best_qt_val_model_op",
            "trained_ram_peak_bytes",
        ],
        "unobserved_port_count": int(len(study_data["unobserved_indices"])),
        "unobserved_indices": [
            int(index) for index in study_data["unobserved_indices"]
        ],
        "outage_threshold": 1.1885398626327515,
        "normalization": "internal_shared_log1p",
        "training_target": (
            "uniform_distribution_over_non_outage_unobserved_ports"
        ),
        "epochs": int(NOT_QT_TRAINING_EPOCHS),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "reduce_lr_patience": int(REDUCE_LR_PATIENCE),
        "artifact_subdirs": {
            "args": "args",
            "model": "model",
            "tflite": "tflite",
            "scaler": "scaler",
        },
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(config_path)


def append_trial_micro_timing_row(csv_path: str, row_data: dict[str, Any]) -> None:
    """
    Append one trial timing record to a CSV file.
    """
    pd.DataFrame([row_data]).to_csv(
        csv_path,
        mode="a",
        index=False,
        header=not os.path.exists(csv_path),
    )


def mark_step(
    step_timings: dict[str, float | int | str],
    step_name: str,
    step_start: float,
) -> None:
    """
    Record elapsed time for a named objective step.

    Args:
        step_timings (dict[str, float | int | str]): Mutable timing accumulator
            updated in-place and later serialized as a CSV row.
        step_name (str): CSV column name for the timed step.
        step_start (float): ``perf_counter`` timestamp captured before the
            step began.
    """
    step_timings[step_name] = perf_counter() - step_start


def apply_test_multiplier(split_data: np.ndarray, split_name: str) -> np.ndarray:
    """Apply the optional fast-test multiplier to one train or validation split.

    Args:
        split_data (np.ndarray): Split matrix to keep or downsample. The first
            axis is interpreted as the sample axis, and the remaining axes are
            preserved unchanged.
        split_name (str): Human-readable split name used only in validation
            error messages. Expected values in this script are ``"train"`` and
            ``"validation"``.

    Returns:
        np.ndarray: Original ``split_data`` when ``TEST_MULTIPIER`` is ``None``;
            otherwise, the first ``floor(n_samples * TEST_MULTIPIER)`` rows.

    Raises:
        ValueError: If ``TEST_MULTIPIER`` is not ``None`` and is outside
            ``(0, 1]``, or if the selected multiplier leaves the split empty.

    Examples:
        >>> callable(apply_test_multiplier)
        True
    """
    if TEST_MULTIPIER is None:
        return split_data

    if TEST_MULTIPIER <= 0 or TEST_MULTIPIER > 1:
        raise ValueError("TEST_MULTIPIER must be None or a value in (0, 1].")

    sample_count = int(split_data.shape[0] * TEST_MULTIPIER)
    if sample_count < 1:
        raise ValueError(
            f"TEST_MULTIPIER={TEST_MULTIPIER!r} leaves the {split_name} split empty."
        )

    # Keep the subset deterministic and transparent for quick local training
    # checks. This does not alter test data or reshuffle split membership.
    return split_data[:sample_count]


def resolve_steps_per_execution(study_data: dict[str, Any]) -> int:
    """Resolve the configured Keras ``steps_per_execution`` value.

    Args:
        study_data (dict[str, Any]): Data dictionary returned by
            ``load_regimes_data``. It must contain selector train and
            validation arrays whose first dimension is the post-multiplier
            sample count used by training and validation.

    Returns:
        int: Positive integer to pass into ``model.compile``. If
            ``STEPS_PER_EXECUTION`` is an integer, that exact value is returned
            after validation. If it is ``"auto"``, the top Araras
            recommendation for the current train/validation sizes is returned.

    Raises:
        KeyError: If required arrays are missing from ``study_data``.
        TypeError: If a non-integer, non-``"auto"`` configured value is passed
            through to ``validate_steps_per_execution``.
        ValueError: If the configured integer is invalid, exceeds the available
            batch count, or Araras returns no recommendation for ``"auto"``.

    Examples:
        >>> callable(resolve_steps_per_execution)
        True
    """
    data_size = [
        study_data["X_train_selector"].shape[0],
        study_data["X_val_selector"].shape[0],
    ]

    if STEPS_PER_EXECUTION == "auto":
        validation_result = validate_steps_per_execution(
            name=["X_train", "X_val"],
            data_size=data_size,
            batch_size=BATCH_SIZE,
            steps_per_execution=1,
            top_k_recommendations=1,
            execute_once=False,
        )
        recommendations = validation_result["recommendations"]
        if not recommendations:
            raise ValueError("Araras returned no steps_per_execution recommendation.")
        return int(recommendations[0]["steps_per_execution"])

    validate_steps_per_execution(
        name=["X_train", "X_val"],
        data_size=data_size,
        batch_size=BATCH_SIZE,
        steps_per_execution=STEPS_PER_EXECUTION,
    )
    return int(STEPS_PER_EXECUTION)


def load_regimes_data(
    train_regime_labels: list[str] | tuple[str, ...],
    observed_ports_budget: int,
) -> dict[str, Any]:
    """Load and scale selected regimes for one observed-port budget split.

    Args:
        train_regime_labels (list[str] | tuple[str, ...]): Regimes selected
            from ``KMS_REGIME_LABELS``. All labels are passed together to
            ``split_kms_60_20_20_cache`` so training and validation use one
            concatenated split across the selected regimes.
        observed_ports_budget (int): Number of equally spaced observed ports to
            expose to the model input. Higher budgets increase input width and
            sensing cost; lower budgets reduce input information.

    Returns:
        dict[str, Any]: Dictionary containing scaled train/validation/test
            matrices, unscaled targets, fitted scalers, observed indices, and
            fixed calibration samples for the study.

    Raises:
        FileNotFoundError: If the runtime KMS cache is missing.
        TypeError: If ``train_regime_labels`` is not a list or tuple.
        ValueError: If any regime label or observed-port budget is invalid.
    """
    if not isinstance(train_regime_labels, (list, tuple)):
        raise TypeError("train_regime_labels must be a list or tuple.")
    if not train_regime_labels:
        raise ValueError("train_regime_labels must contain at least one regime.")

    unknown_regimes = [
        regime_label
        for regime_label in train_regime_labels
        if regime_label not in KMS_REGIME_LABELS
    ]
    if unknown_regimes:
        raise ValueError(f"Unknown train_regime_labels={unknown_regimes!r}.")

    if observed_ports_budget <= 0:
        raise ValueError("observed_ports_budget must be positive.")

    # Load the selected regimes in one call so the cache helper applies the
    # 60/20/20 split per regime, then returns concatenated train/val/test arrays
    # aligned with the all-regimes training contract.
    y_train, y_val, y_test, _, _, _ = split_kms_60_20_20_cache(
        common_path=None,
        seed=DATA_SEED,
        regime_labels=tuple(train_regime_labels),
        samples_per_regime=None,
        verbose=True,
    )

    y_train = apply_test_multiplier(y_train, "train")
    y_val = apply_test_multiplier(y_val, "validation")

    selector_data = prepare_selector_study_data(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        observed_ports_budget=observed_ports_budget,
        calibration_sample_count=INT8_CALIBRATION_SAMPLE_COUNT,
        calibration_num_bins=INT8_CALIBRATION_NUM_BINS,
        calibration_seed=INT8_CALIBRATION_SEED,
    )
    return {
        **selector_data,
        "regime_label": "all_regimes",
        "train_regime_labels": list(train_regime_labels),
        "observed_ports_budget": int(observed_ports_budget),
    }


def best_validation_loss(history: tf.keras.callbacks.History) -> float:
    """Extract the lowest validation selector loss from a Keras history.

    Args:
        history (tf.keras.callbacks.History): History returned by
            ``model.fit``. The function accepts either ``val_mse`` or
            ``val_mean_squared_error`` because Keras metric naming can differ
            by version.

    Returns:
        float: Lowest recorded validation loss.

    Raises:
        KeyError: If no validation loss series is present.
    """
    return float(np.min(history.history["val_loss"]))


def save_history_csv(
    history_dir: str,
    trial_number: int,
    history: tf.keras.callbacks.History,
) -> str:
    """Save FP32 training history for one PTQ trial.

    Args:
        history_dir (str): Study history directory supplied by ``run_study``.
        trial_number (int): Optuna trial number.
        history (tf.keras.callbacks.History): FP32 training history.

    Returns:
        str: Path to the saved CSV history file.

    Raises:
        OSError: If the history CSV cannot be written.
    """
    rows = []
    for epoch_index in range(len(history.history["loss"])):
        rows.append(
            {
                "phase": "not_qt_training",
                "epoch": epoch_index + 1,
                "train_loss": history.history["loss"][epoch_index],
                "val_loss": history.history["val_loss"][epoch_index],
                "val_target_argmax_accuracy": history.history[
                    "val_target_argmax_accuracy"
                ][epoch_index],
            }
        )

    history_path = os.path.join(history_dir, f"trial_{trial_number}.csv")
    pd.DataFrame(rows).to_csv(history_path, index=False)
    return history_path


def save_scalers(
    scaler_path: str,
    x_scaler: object,
    y_scaler: object,
) -> str:
    """Persist the train-fitted MinMax scalers for one study.

    Args:
        scaler_path (str): Destination path for the serialized scaler pair.
        x_scaler (object): Identity input transformer fitted on training data.
        y_scaler (object): Identity output transformer for decision scores.

    Returns:
        str: Path to the saved scaler pickle.

    Raises:
        OSError: If the scaler file cannot be written.
    """
    if os.path.exists(scaler_path):
        return scaler_path

    with open(scaler_path, "wb") as scaler_file:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, scaler_file)
    return scaler_path


def set_static_trial_attrs(trial: optuna.Trial, study_data: dict[str, Any]) -> None:
    """Store fixed per-study Optuna metadata on a trial.

    Args:
        trial (optuna.Trial): Trial receiving user attributes.
        study_data (dict[str, Any]): Data dictionary returned by
            ``load_regimes_data``. Only scalar and JSON-like metadata fields are
            copied to Optuna.

    Returns:
        None: The trial is updated in-place.

    Raises:
        Exception: Propagates Optuna storage errors.
    """
    trial.set_user_attr("ram_budget_violated", False)
    trial.set_user_attr("violating_ram_peak_bytes", float("nan"))
    trial.set_user_attr("backbone_family", "MobileNetV3Small")
    trial.set_user_attr("variant_name", "mobv3_alpha02")


def penalty_objectives() -> tuple[float, float]:
    """Return the fixed invalid-trial objective tuple.

    Args:
        None.

    Returns:
        tuple[float, float]: OP and RAM penalties so
            invalid trials cannot use measured RAM as a Pareto objective.

    Raises:
        None.
    """
    return float(PENALTY_INT8_VAL_MODEL_OP), float(PENALTY_RAM_BYTES)


def set_penalty_trial_attrs(
    trial: optuna.Trial,
    *,
    ram_budget_violated: bool,
    violating_ram_peak_bytes: int | float,
) -> None:
    """Store penalty metadata for an invalid two-objective trial.

    Args:
        trial (optuna.Trial): Trial receiving penalty user attributes.
        ram_budget_violated (bool): Whether the invalid trial violated the
            measured INT8 RAM budget.
        violating_ram_peak_bytes (int | float): Measured RAM in bytes for
            RAM-budget violations, or ``nan`` for non-RAM failures.

    Returns:
        None: The trial is updated in-place.

    Raises:
        Exception: Propagates Optuna storage errors.
    """

    trial.set_user_attr(
        "best_qt_val_model_op",
        float(PENALTY_INT8_VAL_MODEL_OP),
    )
    trial.set_user_attr("ram_budget_violated", bool(ram_budget_violated))
    trial.set_user_attr(
        "violating_ram_peak_bytes",
        float(violating_ram_peak_bytes),
    )

# ———————————————————————————————————————————————————————————————————————————— #
#                                Model Building                                #
# ———————————————————————————————————————————————————————————————————————————— #
def build_model(
    trial: optuna.Trial,
    study_data: dict[str, Any],
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, dict[str, Any]]:
    """Build one sampled GroupNorm safe-selector candidate.

    Args:
        trial (optuna.Trial): Current Optuna trial used to sample architecture
            and learning-rate hyperparameters.
        study_data (dict[str, Any]): Prepared selector data and fixed metadata.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, dict[str, Any]]:
            Explorer, compact selector, protocol selector, and metadata.

    Raises:
        ValueError: If the sampled truncation block is not present in the
            MobileNetV3-Small backbone.
    """
    alpha = trial.suggest_categorical("alpha", ALPHA_VALUES)
    truncation_block_name = trial.suggest_categorical(
        "truncation_block_name",
        MOBILENETV3_TRUNCATION_BLOCK_NAMES,
    )
    head_depth = trial.suggest_int("head_depth", 1, 3)
    head_width = trial.suggest_int("head_width", 16, 256, step=16)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    explorer_model, compact_model, protocol_model = (
        build_nas_selector_models(
            observed_indices=np.asarray(
                study_data["observed_indices"],
                dtype=np.int64,
            ),
            log_scale=float(study_data["log_scale"]),
            alpha=float(alpha),
            truncation_block_name=truncation_block_name,
            head_depth=int(head_depth),
            head_width=int(head_width),
            train_seed=TRAIN_SEED,
            variant_name="mobv3_alpha02",
        )
    )
    metadata = {
        "alpha": float(alpha),
        "image_side": int(IMAGE_SIDE),
        "truncation_block_name": truncation_block_name,
        "truncation_layer_name": truncation_block_name,
        "head_depth": int(head_depth),
        "head_width": int(head_width),
        "learning_rate": float(learning_rate),
    }
    return explorer_model, compact_model, protocol_model, metadata


# %%—————————————————————————————————————————————————————————————————————————— #
#                              Objective Function                              #
# ———————————————————————————————————————————————————————————————————————————— #
def objective(
    trial: optuna.Trial,
    **kwargs: Any,
) -> tuple[float, float]:
    """Evaluate one PTQ INT8 MobileNetV3-Small NAS candidate.

    Args:
        trial (optuna.Trial): Current Optuna trial.
        **kwargs (Any): Runtime arguments supplied by ``run_study``. The
            required ``study_data`` entry contains preloaded all-regimes data;
            directory entries such as ``model``, ``history``, ``scaler``,
            ``tflite``, and ``logs`` define artifact locations.

    Returns:
        tuple[float, float]: Two minimization objectives:
            ``(best_qt_val_model_op, trained_ram_peak_bytes)`` for valid trials,
            or the fixed penalty tuple for RAM violations and invalid
            conversion or evaluation failures.

    Raises:
        OSError: If timing/error logs cannot be written.
    """

    log_print("objective", f"Running trial {trial.number}.")
    clear_session()
    set_random_seed(TRAIN_SEED)

    trial_t0 = perf_counter()
    step_timings: dict[str, float | int | str] = {}
    trial_status = "unknown"

    study_data = kwargs["study_data"]
    model_dir = kwargs["model"]
    history_dir = kwargs["history"]
    scaler_dir = kwargs["scaler"]
    logs_dir = kwargs["logs"]
    tflite_dir = kwargs["tflite"]
    steps_per_execution = kwargs["steps_per_execution"]

    try:
        # ———————————————————————————————————————————————————————————————————————————— #
        #                                 Preprocessing                                #
        # ———————————————————————————————————————————————————————————————————————————— #
        set_static_trial_attrs(trial, study_data)

        save_scalers(
            scaler_path=os.path.join(scaler_dir, f"scaler_xy.pkl"),
            x_scaler=study_data["x_scaler"],
            y_scaler=study_data["y_scaler"],
        ) # Will save only once per study

        # ———————————————————————————————————————————————————————————————————————————— #
        #                        Model Construction and Training                       #
        # ———————————————————————————————————————————————————————————————————————————— #
        log_print("objective", "Building model.")
        (
            explorer_model,
            compact_decision_model,
            not_qt_model,
            model_metadata,
        ) = build_model(
            trial,
            study_data=study_data,
        )

        log_print("objective", "Converting pre-train model to INT8 TFLite for RAM estimation.")
        pretrain_qt_model = convert_ptq_model_to_strict_int8_tflite(
            not_qt_model,
            study_data["calibration_data"],
        )

        log_print("objective", "Testing strict int8 I/O compatibility of pre-train quantized model.")
        test_strict_int8_io(pretrain_qt_model) # Check if qt model can be converted to TFLite

        log_print("objective", "Estimating pre-train quantized model peak RAM usage.")
        pretrain_ram_peak_bytes = estimate_tflite_peak_ram_bytes(pretrain_qt_model)
        trial.set_user_attr("pretrain_ram_peak_bytes", int(pretrain_ram_peak_bytes))
        log_print("objective", f"Pre-train quantized model peak RAM: {pretrain_ram_peak_bytes / 1024:.2f} KiB")

        if pretrain_ram_peak_bytes > MAX_RAM_BYTES:
            trial_status = "ram_budget_violated"
            set_penalty_trial_attrs(
                trial,
                ram_budget_violated=True,
                violating_ram_peak_bytes=int(pretrain_ram_peak_bytes),
            )
            return penalty_objectives()

        log_print("objective", "Compiling model.")
        explorer_model.compile(
            optimizer=optimizers.Adam(learning_rate=model_metadata["learning_rate"]),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(
                    name="target_argmax_accuracy"
                )
            ],
            jit_compile=USE_JIT_COMPILE,
            steps_per_execution=steps_per_execution,
        )

        step_t0 = perf_counter()
        log_print("objective", "Training model.")
        not_qt_history = explorer_model.fit(
            study_data["X_train_selector"],
            study_data["y_train_selector"],
            validation_data=(
                study_data["X_val_selector"],
                study_data["y_val_selector"],
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            epochs=NOT_QT_TRAINING_EPOCHS,
            callbacks=get_callbacks_study(
                trial=trial,
                monitor="val_loss",
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                reduce_lr_patience=REDUCE_LR_PATIENCE,
                pruning_interval=PRUNING_INTERVAL,
                #! Can cause high memory usage
                # tensorboard_logs=tensorboard_dir,
            ),
            verbose=2,
        )
        best_validation_loss(not_qt_history)
        best_not_qt_val_model_op = evaluate_keras_selector_op(
            model=compact_decision_model,
            observed_values=study_data["X_val_raw"],
            y_values=study_data["y_val"],
            batch_size=BATCH_SIZE,
        )
        trial.set_user_attr(
            "best_not_qt_val_model_op",
            best_not_qt_val_model_op,
        )
        not_qt_diagnostics = compute_selection_snapshot(
            decision_model=compact_decision_model,
            y_values=study_data["y_val"][:5_000],
            num_observed_ports=study_data["observed_ports_budget"],
            batch_size=BATCH_SIZE,
            permutation_seed=TRAIN_SEED,
        )
        trial.set_user_attr(
            "not_qt_distinct_selected_ports",
            not_qt_diagnostics["distinct_selected_ports"],
        )
        trial.set_user_attr(
            "not_qt_permutation_change_rate",
            not_qt_diagnostics["permutation_change_rate"],
        )
        mark_step(step_timings, "not_qt_training", step_t0)

        # Save not qt model and history
        not_qt_model_path = os.path.join(model_dir, f"trial_{trial.number}.keras")
        not_qt_model.save(not_qt_model_path)
        save_history_csv(history_dir, trial.number, not_qt_history)

        # ———————————————————————————————————————————————————————————————————————————— #
        #                   Quantization of the Model and Evaluation                   #
        # ———————————————————————————————————————————————————————————————————————————— #
        log_print("objective", "Converting model to INT8 TFLite.")
        qt_model = convert_ptq_model_to_strict_int8_tflite(
            not_qt_model,
            study_data["calibration_data"],
        )

        # Estimate RAM again, just to see if it changes after training
        log_print("objective", "Estimating post-train quantized model peak RAM usage.")
        trained_ram_peak_bytes = estimate_tflite_peak_ram_bytes(qt_model)
        trial.set_user_attr("trained_ram_peak_bytes", int(trained_ram_peak_bytes))

        if DOUBLE_CHECK_RAM:
            if trained_ram_peak_bytes > MAX_RAM_BYTES:
                trial_status = "ram_budget_violated_after_training"
                set_penalty_trial_attrs(
                    trial,
                    ram_budget_violated=True,
                    violating_ram_peak_bytes=int(trained_ram_peak_bytes),
                )
                return penalty_objectives()

        log_print("objective", "Testing strict int8 I/O compatibility of quantized model.")
        test_strict_int8_io(qt_model)

        qt_model_path = os.path.join(tflite_dir, f"trial_{trial.number}.tflite")
        with open(qt_model_path, "wb") as tflite_file:
            tflite_file.write(qt_model)
        trial.set_user_attr("qt_model_size_bytes", int(len(qt_model)))

        step_t0 = perf_counter()
        log_print("objective", "Evaluating quantized model on validation set.")
        best_qt_val_model_op = evaluate_tflite_selector_op(
            tflite_model=qt_model,
            observed_values=study_data["X_val_raw"],
            y_values=study_data["y_val"],
            observed_indices=np.asarray(study_data["observed_indices"], dtype=np.int64),
            batch_size=BATCH_SIZE,
        )
        trial.set_user_attr(
            "best_qt_val_model_op",
            float(best_qt_val_model_op),
        )
        trial.set_user_attr(
            "quantization_op_degradation",
            float(best_qt_val_model_op - best_not_qt_val_model_op),
        )
        qt_diagnostics = evaluate_tflite_selector_diagnostics(
            tflite_model=qt_model,
            observed_values=study_data["X_val_raw"],
            observed_indices=np.asarray(
                study_data["observed_indices"],
                dtype=np.int64,
            ),
            batch_size=BATCH_SIZE,
            permutation_seed=TRAIN_SEED,
        )
        trial.set_user_attr(
            "qt_distinct_selected_ports",
            qt_diagnostics["distinct_selected_ports"],
        )
        trial.set_user_attr(
            "qt_permutation_change_rate",
            qt_diagnostics["permutation_change_rate"],
        )
        mark_step(step_timings, "qt_validation", step_t0)

        # ———————————————————————————————————————————————————————————————————————————— #
        #                          Save Model Characteristics                          #
        # ———————————————————————————————————————————————————————————————————————————— #
        log_print("objective", "Measuring and saving model characteristics.")
        set_user_attr_model_stats(
            trial=trial,
            model=not_qt_model,
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=1,  # ? Benchmarking per-sample stats
            test_runs=10,
            device="gpu/0",
            stats_to_measure=(
                "parameters",
                "model_size",
                "flops",
                "macs",
                "summary",
                "inference_latency",
                # "cpu_util_percent",
                # "cpu_power_rapl_w",
                # "ram_used_bytes",
                # "ram_util_percent",
                # "gpu_util_percent",
                # "gpu_mem_used_bytes",
                # "gpu_power_w",
            ),
            extra_attrs=None,
            verbose=1,
        )

        set_user_attr_tflite_model_stats(
            trial=trial,
            tflite_model=qt_model,
            batch_size=1,  # ? Benchmarking per-sample stats
            test_runs=10,
            stats_to_measure=(
                "model_size",
                "peak_ram_bytes",
                "tensor_count",
                "operator_count",
                "input_details",
                "output_details",
                "inference_latency",
            ),
            extra_attrs=None,
            verbose=1,
        )
        # ———————————————————————————————————————————————————————————————————————————— #

        trial_status = "complete"
        return float(best_qt_val_model_op), float(trained_ram_peak_bytes)
    except Exception as exc:
        trial_status = "failed"
        set_penalty_trial_attrs(
            trial,
            ram_budget_violated=False,
            violating_ram_peak_bytes=float("nan"),
        )
        trial.set_user_attr("trial_error_message", str(exc))
        log_trial_error(
            trial=trial,
            exc=exc,
            logs_dir=logs_dir,
            prune_on={},
            propagate={},
            force_crash_oom=5,
            raise_after_log=True,
        )
        # return penalty_objectives()
    finally:
        step_timings["total_seconds"] = perf_counter() - trial_t0
        step_timings["trial_number"] = trial.number
        step_timings["status"] = trial_status
        append_trial_micro_timing_row(
            os.path.join(logs_dir, "micro_task_timings.csv"),
            step_timings,
        )
        clear_session()


# %%—————————————————————————————————————————————————————————————————————————— #
#                                     Study                                    #
# ———————————————————————————————————————————————————————————————————————————— #
def run_all_regime_studies() -> None:
    """Run the complete MobileNetV3-Small all-regimes NAS campaign.

    Args:
        None.

    Returns:
        None: Study artifacts are written below
            ``./runs/nas_ptq_mobv3_all_regimes_*_ports``.

    Raises:
        Exception: Propagates data-loading, Optuna, training, PTQ conversion,
            strict-INT8 validation, and artifact-writing failures.

    Examples:
        Run this production campaign only from the script entrypoint:

        >>> callable(run_all_regime_studies)
        True
    """
    for observed_ports in OBSERVED_PORTS_LIST:
        run_dir = f"./runs/nas_ptq_mobv3_all_regimes_{observed_ports}_ports"
        log_print(
            "run_all_regime_studies",
            f"Preparing all-regimes study for observed_ports={observed_ports}.",
        )
        study_data = load_regimes_data(
            train_regime_labels=TRAIN_REGIME_LABELS,
            observed_ports_budget=observed_ports,
        )
        steps_per_execution = resolve_steps_per_execution(study_data)
        write_nas_ptq_config(
            run_dir=run_dir,
            train_regime_labels=TRAIN_REGIME_LABELS,
            observed_ports_budget=observed_ports,
            study_data=study_data,
        )

        run_study(
            objective=objective,
            run_dir=run_dir,
            num_trials=NUM_TRIALS,
            sampler_seed=SAMPLER_SEED,
            direction=DIRECTION,
            top_k=TOP_K,
            rank_key=RANK_KEY,
            order=ORDER,
            convergence_epoch_column="val_loss",
            convergence_epoch_direction="minimize",
            init_study_dirs=[
                "args",
                "fig",
                "backup",
                "history",
                "scaler",
                "model",
                "tflite",
                "logs",
                "tensorboard",
            ],
            cleanup_paths=[
                ("model", "trial_{trial_id}.keras"),
                ("history", "trial_{trial_id}.csv"),
                ("tflite", "trial_{trial_id}.tflite"),
                ("tensorboard", "trial_{trial_id}"),
            ],
            rename_paths=[
                ("model", ".keras"),
                ("history", ".csv"),
                ("scaler", ".pkl"),
                ("tflite", ".tflite"),
            ],
            extra_attrs=EXTRA_ATTRS,
            variance_threshold=None,
            prune_threshold=None,
            patience=None,
            study_data=study_data,
            steps_per_execution=steps_per_execution,
        )


if __name__ == "__main__":
    run_all_regime_studies()
