"""Reusable safe-selector components for MobileNetV1 and V3 PTQ NAS studies."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer

from mobilenet.keras.sparse_input import (
    build_sparse_port_images,
    iter_sparse_port_image_batches,
)
from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    IMAGE_SIDE,
    OUTAGE_THRESHOLD,
    ExpandedPortImage,
    ObservedPortLogitMask,
    SafeDecisionScores,
    build_protocol_decision_model,
    build_recovery_training_data,
    build_non_identity_permutations,
    convolution_block,
    depthwise_block,
    make_divisible,
    mobilenet_v3_block,
)
from utils.data import get_observed_ports
from utils.qttf import (
    build_quantile_balanced_calibration_data,
    make_tflite_interpreter,
)


TOTAL_PORTS = 100
TFLITE_EVALUATION_WORKERS = 8
MOBILENETV1_TRUNCATION_BLOCK_NAMES = tuple(
    f"conv_pw_{block_index}_relu"
    for block_index in range(1, 14)
)
_MOBILENETV1_BLOCK_SCHEDULE = (
    (64, 1),
    (128, 2),
    (128, 1),
    (256, 2),
    (256, 1),
    (512, 2),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (1024, 2),
    (1024, 1),
)
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
_MOBILENETV3_BLOCK_SPECS = (
    (16, 16, 3, 2, True, "relu"),
    (72, 24, 3, 2, False, "relu"),
    (88, 24, 3, 1, False, "relu"),
    (96, 40, 5, 2, True, "hard_swish"),
    (240, 40, 5, 1, True, "hard_swish"),
    (240, 40, 5, 1, True, "hard_swish"),
    (120, 48, 5, 1, True, "hard_swish"),
    (144, 48, 5, 1, True, "hard_swish"),
    (288, 96, 5, 2, True, "hard_swish"),
    (576, 96, 5, 1, True, "hard_swish"),
    (576, 96, 5, 1, True, "hard_swish"),
)


def prepare_selector_study_data(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    observed_ports_budget: int,
    calibration_sample_count: int,
    calibration_num_bins: int,
    calibration_seed: int,
) -> dict[str, Any]:
    """Build selector training targets, identity scalers, and PTQ inputs."""
    x_train, observed_indices = get_observed_ports(
        y_train,
        observed_ports_budget,
        TOTAL_PORTS,
    )
    x_val, _ = get_observed_ports(
        y_val,
        observed_ports_budget,
        TOTAL_PORTS,
    )
    x_test, _ = get_observed_ports(
        y_test,
        observed_ports_budget,
        TOTAL_PORTS,
    )
    unobserved_indices = np.setdiff1d(
        np.arange(TOTAL_PORTS, dtype=np.int64),
        observed_indices,
    )
    train_inputs, train_targets, checked_indices = (
        build_recovery_training_data(
            y_values=y_train,
            num_observed_ports=observed_ports_budget,
        )
    )
    val_inputs, val_targets, _ = build_recovery_training_data(
        y_values=y_val,
        num_observed_ports=observed_ports_budget,
    )
    if not np.array_equal(observed_indices, checked_indices):
        raise ValueError("Observed indices changed during selector preparation.")
    if len(train_inputs) == 0 or len(val_inputs) == 0:
        raise ValueError("No recoverable samples exist for selector NAS.")

    x_scaler = FunctionTransformer(validate=True)
    y_scaler = FunctionTransformer(validate=True)
    x_scaler.fit(x_train[: min(1024, len(x_train))])
    y_scaler.fit(y_train[: min(1024, len(y_train))])

    calibration_values, calibration_counts = (
        build_quantile_balanced_calibration_data(
            x_train,
            y_train,
            sample_count=calibration_sample_count,
            num_bins=calibration_num_bins,
            seed=calibration_seed,
        )
    )
    calibration_data = build_sparse_port_images(
        observed_values=calibration_values,
        observed_indices=observed_indices.astype(int),
        total_ports=TOTAL_PORTS,
        image_height=IMAGE_SIDE,
        image_width=IMAGE_SIDE,
    )
    scale_stride = max(1, len(y_train) // 100_000)
    log_scale = float(
        np.quantile(np.log1p(y_train[::scale_stride]), 0.999)
    )
    return {
        "observed_indices": observed_indices.astype(int).tolist(),
        "unobserved_indices": unobserved_indices.astype(int).tolist(),
        "log_scale": log_scale,
        "X_train_selector": train_inputs,
        "y_train_selector": train_targets,
        "X_val_selector": val_inputs,
        "y_val_selector": val_targets,
        "X_train_raw": x_train.astype(np.float32),
        "X_val_raw": x_val.astype(np.float32),
        "X_test_raw": x_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "calibration_data": calibration_data,
        "calibration_counts": calibration_counts,
    }


def build_nas_selector_models(
    observed_indices: np.ndarray,
    log_scale: float,
    alpha: float,
    truncation_block_name: str,
    head_depth: int,
    head_width: int,
    train_seed: int,
    variant_name: str = "mobv1_alpha02",
) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build sampled explorer, compact selector, and sparse-protocol models.

    Args:
        observed_indices (np.ndarray): One-dimensional indices of the observed
            ports. Values must identify valid ports in the 100-port output.
        log_scale (float): Positive shared ``log1p`` normalization scale used
            by the embedded compact-to-image projection.
        alpha (float): Positive MobileNet width multiplier. It scales backbone
            channel counts and therefore parameter count and compute.
        truncation_block_name (str): Backbone output at which feature
            extraction stops. MobileNetV1 accepts names from
            ``MOBILENETV1_TRUNCATION_BLOCK_NAMES``. MobileNetV3-Small accepts
            names from ``MOBILENETV3_TRUNCATION_BLOCK_NAMES``.
        head_depth (int): Positive number of dense selector-head layers.
            Increasing it adds parameters and compute.
        head_width (int): Positive number of units in each dense head layer.
            Increasing it adds parameters and compute.
        train_seed (int): Seed for deterministic Glorot initialization of the
            dense selector head.
        variant_name (str): Backbone selector. ``"mobv1_alpha02"`` builds the
            MobileNetV1 search space and preserves the legacy NAS behavior.
            ``"mobv3_alpha02"`` builds the MobileNetV3-Small search space with
            squeeze-and-excitation and mixed ReLU/hard-swish blocks. The value
            changes model structure and serialized model names but does not
            alter files or global device placement.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]: Explorer model
            trained on unobserved-port distributions, compact decision-score
            model, and 32 × 32 sparse-image protocol model used for strict-INT8
            PTQ.

    Raises:
        ValueError: If the variant or truncation block is unsupported or if a
            width/head value is non-positive.

    Examples:
        >>> callable(build_nas_selector_models)
        True
    """
    if variant_name == "mobv1_alpha02":
        truncation_names = MOBILENETV1_TRUNCATION_BLOCK_NAMES
        model_slug = "mobv1"
    elif variant_name == "mobv3_alpha02":
        truncation_names = MOBILENETV3_TRUNCATION_BLOCK_NAMES
        model_slug = "mobv3"
    else:
        raise ValueError(
            f"Unsupported NAS selector variant {variant_name!r}. "
            "Expected 'mobv1_alpha02' or 'mobv3_alpha02'."
        )
    if truncation_block_name not in truncation_names:
        raise ValueError(
            f"Unsupported {variant_name} truncation block "
            f"{truncation_block_name!r}."
        )
    if alpha <= 0.0 or head_depth <= 0 or head_width <= 0:
        raise ValueError("alpha, head_depth, and head_width must be positive.")
    inputs = tf.keras.layers.Input(
        shape=(len(observed_indices),),
        name="observed_snr",
    )
    images = ExpandedPortImage(
        observed_indices=observed_indices.tolist(),
        log_scale=log_scale,
        name="expanded_port_image",
    )(inputs)
    if variant_name == "mobv1_alpha02":
        outputs = convolution_block(
            images,
            filters=make_divisible(32 * alpha),
            strides=2,
            name="nas_stem",
        )
        for block_index, (base_filters, strides) in enumerate(
            _MOBILENETV1_BLOCK_SCHEDULE,
            start=1,
        ):
            outputs = depthwise_block(
                outputs,
                pointwise_filters=make_divisible(base_filters * alpha),
                strides=strides,
                block_index=block_index,
            )
            if truncation_block_name == f"conv_pw_{block_index}_relu":
                break
    else:
        outputs = convolution_block(
            images,
            filters=make_divisible(16 * alpha),
            strides=2,
            name="nas_stem",
            activation="hard_swish",
        )
        for block_index, (
            base_expansion,
            base_filters,
            kernel_size,
            strides,
            use_squeeze_excite,
            activation,
        ) in enumerate(_MOBILENETV3_BLOCK_SPECS):
            outputs = mobilenet_v3_block(
                inputs=outputs,
                expansion_filters=make_divisible(base_expansion * alpha),
                pointwise_filters=make_divisible(base_filters * alpha),
                kernel_size=kernel_size,
                strides=strides,
                use_squeeze_excite=use_squeeze_excite,
                activation=activation,
                block_index=block_index,
            )
            if truncation_block_name == MOBILENETV3_TRUNCATION_BLOCK_NAMES[block_index]:
                break

    outputs = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(outputs)
    initializer = tf.keras.initializers.GlorotUniform(seed=train_seed)
    for head_index in range(head_depth):
        outputs = tf.keras.layers.Dense(
            head_width,
            activation="relu",
            kernel_initializer=initializer,
            name=f"selector_dense_{head_index}",
        )(outputs)

    exploration_logits = tf.keras.layers.Dense(
        TOTAL_PORTS,
        activation=None,
        kernel_initializer=initializer,
        bias_initializer="zeros",
        name="exploration_logits",
    )(outputs)
    masked_exploration_logits = ObservedPortLogitMask(
        observed_indices=observed_indices.tolist(),
        name="observed_port_logit_mask",
    )(exploration_logits)
    explorer_model = tf.keras.Model(
        inputs=inputs,
        outputs=masked_exploration_logits,
        name=f"nas_ptq_{model_slug}_gn_explorer",
    )
    decision_scores = SafeDecisionScores(
        observed_indices=observed_indices.tolist(),
        threshold=OUTAGE_THRESHOLD,
        name="safe_decision_scores",
    )((inputs, exploration_logits))
    compact_decision_model = tf.keras.Model(
        inputs=inputs,
        outputs=decision_scores,
        name=f"nas_ptq_{model_slug}_gn_safe_selector",
    )
    protocol_model = build_protocol_decision_model(
        compact_decision_model=compact_decision_model,
        observed_indices=observed_indices,
        variant_name=variant_name,
    )
    return explorer_model, compact_decision_model, protocol_model


def evaluate_keras_selector_op(
    model: tf.keras.Model,
    observed_values: np.ndarray,
    y_values: np.ndarray,
    batch_size: int,
) -> float:
    """Return model-only OP for a compact Keras decision model."""
    scores = model.predict(
        observed_values,
        batch_size=batch_size,
        verbose=0,
    )
    selected_ports = np.argmax(scores, axis=1)
    rows = np.arange(len(y_values))
    return float(
        np.mean(y_values[rows, selected_ports] <= OUTAGE_THRESHOLD)
    )


def evaluate_tflite_selector_op(
    tflite_model: bytes,
    observed_values: np.ndarray,
    y_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
) -> float:
    """Return model-only OP for a strict-INT8 sparse-image selector."""
    selected_ports = _predict_tflite_selected_ports(
        tflite_model=tflite_model,
        observed_values=observed_values,
        observed_indices=observed_indices,
        batch_size=batch_size,
    )
    rows = np.arange(len(y_values))
    return float(
        np.mean(
            y_values[rows, selected_ports] <= OUTAGE_THRESHOLD
        )
    )


def evaluate_tflite_selector_diagnostics(
    tflite_model: bytes,
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
    permutation_seed: int,
    sample_count: int = 5_000,
) -> dict[str, float]:
    """Measure strict-INT8 decision diversity and input sensitivity."""
    diagnostic_values = observed_values[
        : min(len(observed_values), sample_count)
    ]
    selected_ports = _predict_tflite_selected_ports(
        tflite_model=tflite_model,
        observed_values=diagnostic_values,
        observed_indices=observed_indices,
        batch_size=batch_size,
    )
    change_rates = []
    for permutation in build_non_identity_permutations(
        num_observed_ports=diagnostic_values.shape[1],
        seed=permutation_seed,
    ):
        permuted_ports = _predict_tflite_selected_ports(
            tflite_model=tflite_model,
            observed_values=diagnostic_values[:, permutation],
            observed_indices=observed_indices,
            batch_size=batch_size,
        )
        change_rates.append(
            float(np.mean(selected_ports != permuted_ports))
        )
    return {
        "distinct_selected_ports": float(
            np.unique(selected_ports).size
        ),
        "permutation_change_rate": float(np.mean(change_rates)),
    }


def _predict_tflite_selected_ports(
    tflite_model: bytes,
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return selected ports using parallel batched TFLite inference.

    Args:
        tflite_model (bytes): Serialized strict-INT8 TFLite selector with a
            dynamic input batch dimension.
        observed_values (np.ndarray): Compact observed-port values with shape
            ``(n_samples, n_observed_ports)``. Every row is evaluated.
        observed_indices (np.ndarray): One-dimensional physical port indices
            aligned with the columns of ``observed_values``.
        batch_size (int): Positive total number of samples processed
            concurrently across all interpreters. It is divided across eight
            workers to bound aggregate activation memory. Each worker resizes
            its final partial batch exactly.

    Returns:
        np.ndarray: Selected physical port index for every input row, in the
        original sample order.

    Raises:
        ValueError: If inputs are empty, ``batch_size`` is not positive,
            quantization scales are zero, or the model batch dimension is not
            dynamic.
        Exception: Propagates TensorFlow Lite allocation and invocation errors.

    Examples:
        >>> callable(_predict_tflite_selected_ports)
        True
    """
    if len(observed_values) == 0:
        raise ValueError("observed_values must contain at least one sample.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    worker_count = min(TFLITE_EVALUATION_WORKERS, len(observed_values))
    worker_batch_size = (
        batch_size + worker_count - 1
    ) // worker_count
    observed_value_chunks = np.array_split(observed_values, worker_count)

    # Each interpreter remains thread-confined while its native invoke call
    # releases the Python GIL. Concatenating futures in submission order keeps
    # predictions aligned with the original validation rows.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _predict_tflite_selected_ports_single_interpreter,
                tflite_model,
                observed_value_chunk,
                observed_indices,
                worker_batch_size,
            )
            for observed_value_chunk in observed_value_chunks
        ]
        selected_port_chunks = [future.result() for future in futures]
    return np.concatenate(selected_port_chunks)


def _predict_tflite_selected_ports_single_interpreter(
    tflite_model: bytes,
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return selected ports from one dynamically batched interpreter.

    Args:
        tflite_model (bytes): Serialized strict-INT8 TFLite selector with a
            dynamic input batch dimension.
        observed_values (np.ndarray): Non-empty compact observed-port values
            assigned to this interpreter.
        observed_indices (np.ndarray): One-dimensional physical port indices
            aligned with the columns of ``observed_values``.
        batch_size (int): Positive number of samples per interpreter
            invocation. The final partial batch is resized exactly.

    Returns:
        np.ndarray: Selected physical port index for every assigned input row,
        in the original chunk order.

    Raises:
        ValueError: If quantization scales are zero or the model batch
            dimension is not dynamic.
        Exception: Propagates TensorFlow Lite allocation and invocation errors.

    Examples:
        >>> callable(_predict_tflite_selected_ports_single_interpreter)
        True
    """
    interpreter = make_tflite_interpreter(tflite_model)
    input_detail = interpreter.get_input_details()[0]
    input_shape_signature = np.asarray(
        input_detail["shape_signature"],
        dtype=np.int64,
    )
    if input_shape_signature[0] != -1:
        raise ValueError(
            "Batched TFLite validation requires a dynamic input batch dimension."
        )

    allocated_batch_size = min(batch_size, len(observed_values))
    input_shape = np.asarray(input_detail["shape"], dtype=np.int64)
    input_shape[0] = allocated_batch_size
    interpreter.resize_tensor_input(
        input_detail["index"],
        input_shape,
        strict=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale == 0 or output_scale == 0:
        raise ValueError("Strict INT8 selector requires quantized I/O.")

    selected_ports = []
    for sparse_batch in iter_sparse_port_image_batches(
        observed_values=observed_values,
        observed_indices=observed_indices,
        batch_size=batch_size,
        total_ports=TOTAL_PORTS,
    ):
        current_batch_size = sparse_batch.shape[0]
        if current_batch_size != allocated_batch_size:
            input_shape[0] = current_batch_size
            interpreter.resize_tensor_input(
                input_detail["index"],
                input_shape,
                strict=True,
            )
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            allocated_batch_size = current_batch_size

        quantized_input = np.round(
            sparse_batch / input_scale + input_zero_point
        )
        quantized_input = np.clip(
            quantized_input,
            -128,
            127,
        ).astype(np.int8)
        interpreter.set_tensor(
            input_detail["index"],
            quantized_input.reshape(input_detail["shape"]),
        )
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(
            output_detail["index"]
        ).astype(np.float32)
        scores = (
            quantized_output - output_zero_point
        ) * output_scale
        selected_ports.extend(
            np.argmax(scores.reshape(current_batch_size, -1), axis=1).tolist()
        )
    return np.asarray(selected_ports, dtype=np.int64)
