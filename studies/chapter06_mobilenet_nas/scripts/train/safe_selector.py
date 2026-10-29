"""Shared GroupNorm MobileNet safe-selector training components."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import gc
import json
import os
import pickle
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/safe_selector.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from mobilenet.keras.sparse_input import build_sparse_port_images
from utils.data import KMS_REGIME_LABELS, get_observed_ports, split_kms_60_20_20_cache


TOTAL_PORTS = 100
IMAGE_SIDE = 32
OUTAGE_THRESHOLD = 1.1885398626327515
DATA_SEED = 0
TRAIN_SEED = 0
VARIANT_METADATA = {
    "mobv1_alpha02": {
        "architecture": "MobileNetV1-style depthwise-separable GroupNorm",
        "backbone_family": "MobileNet",
        "alpha": 0.2,
    },
    "mobv2_alpha02": {
        "architecture": "MobileNetV2-style inverted-residual GroupNorm",
        "backbone_family": "MobileNetV2",
        "alpha": 0.2,
    },
    "mobv3_alpha02": {
        "architecture": "MobileNetV3Small-style inverted-residual GroupNorm",
        "backbone_family": "MobileNetV3Small",
        "alpha": 0.2,
    },
}
DEFAULT_RUN_ROOT = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/mobv1_gn_safe_selector_all_regimes'
)


@dataclass(frozen=True)
class RunOptions:
    """Store validated command-line experiment options."""

    run_root: Path
    observed_ports: list[int]
    samples_per_regime: int
    epochs: int
    batch_size: int
    selection_samples_per_regime: int
    splits: list[str]


@tf.keras.utils.register_keras_serializable(package="fluidra")
class ExpandedPortImage(tf.keras.layers.Layer):
    """Expand compact observed values across the full MobileNet image."""

    def __init__(
        self,
        observed_indices: list[int] | tuple[int, ...],
        log_scale: float,
        image_side: int = IMAGE_SIDE,
        fixed_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the fixed port-to-image projection.

        Args:
            observed_indices (list[int] | tuple[int, ...]): Zero-based physical
                ports supplied by the compact input.
            log_scale (float): Positive scale applied after the internal
                ``log1p`` transform.
            image_side (int): Width and height of the square output image.
                Defaults to ``IMAGE_SIDE``.
            fixed_batch_size (int | None): Exact batch size for deployment
                export. ``None`` preserves dynamic training batches; a positive
                value makes mask and coordinate broadcasts static for TFLM.
            **kwargs (Any): Standard Keras layer keyword arguments.

        Returns:
            None.

        Raises:
            ValueError: If ``fixed_batch_size`` is not positive when supplied.
        """
        super().__init__(**kwargs)
        self.observed_indices = tuple(int(index) for index in observed_indices)
        self.log_scale = float(log_scale)
        self.image_side = int(image_side)
        if fixed_batch_size is not None and fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive when supplied.")
        self.fixed_batch_size = fixed_batch_size

        basis = np.zeros(
            (len(self.observed_indices), self.image_side, self.image_side),
            dtype=np.float32,
        )
        for input_index, port_index in enumerate(self.observed_indices):
            port_row = port_index // 10
            port_col = port_index % 10
            center_row = int(round(port_row * (self.image_side - 1) / 9.0))
            center_col = int(round(port_col * (self.image_side - 1) / 9.0))
            row_start = max(0, center_row - 2)
            row_stop = min(self.image_side, center_row + 3)
            col_start = max(0, center_col - 2)
            col_stop = min(self.image_side, center_col + 3)
            basis[input_index, row_start:row_stop, col_start:col_stop] = 1.0

        mask = np.max(basis, axis=0)
        continuous_rows = np.linspace(0.0, 9.0, self.image_side, dtype=np.float32)
        continuous_cols = np.linspace(0.0, 9.0, self.image_side, dtype=np.float32)
        row_grid, col_grid = np.meshgrid(
            continuous_rows,
            continuous_cols,
            indexing="ij",
        )
        coordinates = (10.0 * row_grid + col_grid) / 99.0

        self._basis = tf.constant(basis, dtype=tf.float32)
        self._mask = tf.constant(mask, dtype=tf.float32)
        self._coordinates = tf.constant(coordinates, dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return value, observation-mask, and coordinate image channels."""
        values = tf.math.log1p(tf.maximum(tf.cast(inputs, tf.float32), 0.0))
        values = values / tf.cast(self.log_scale, tf.float32)
        value_channel = tf.einsum("bn,nhw->bhw", values, self._basis)
        batch_size = (
            tf.shape(inputs)[0]
            if self.fixed_batch_size is None
            else self.fixed_batch_size
        )
        mask_channel = tf.broadcast_to(
            self._mask[tf.newaxis, ...],
            (batch_size, self.image_side, self.image_side),
        )
        coordinate_channel = tf.broadcast_to(
            self._coordinates[tf.newaxis, ...],
            (batch_size, self.image_side, self.image_side),
        )
        return tf.stack(
            (value_channel, mask_channel, coordinate_channel),
            axis=-1,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "observed_indices": list(self.observed_indices),
                "log_scale": self.log_scale,
                "image_side": self.image_side,
                "fixed_batch_size": self.fixed_batch_size,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="fluidra")
class ObservedPortLogitMask(tf.keras.layers.Layer):
    """Exclude observed ports from the learned categorical distribution."""

    def __init__(
        self,
        observed_indices: list[int] | tuple[int, ...],
        total_ports: int = TOTAL_PORTS,
        **kwargs: Any,
    ) -> None:
        """Store the fixed observed-port mask.

        Args:
            observed_indices (list[int] | tuple[int, ...]): Physical indices
                observed by the model. These logits are replaced by a large
                negative value before categorical cross-entropy.
            total_ports (int): Number of physical output ports. Defaults to
                ``TOTAL_PORTS`` and must match the final logit width.
            **kwargs (Any): Standard Keras layer keyword arguments.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(**kwargs)
        self.observed_indices = tuple(int(index) for index in observed_indices)
        self.total_ports = int(total_ports)
        observed_mask = np.zeros(self.total_ports, dtype=bool)
        observed_mask[list(self.observed_indices)] = True
        self._observed_mask = tf.constant(observed_mask, dtype=tf.bool)

    def call(self, logits: tf.Tensor) -> tf.Tensor:
        """Return logits with observed-port entries excluded.

        Args:
            logits (tf.Tensor): Batch of 100 learned exploration logits.

        Returns:
            tf.Tensor: Logits with observed entries replaced by ``-1e4`` so
            their softmax probability is numerically zero while remaining
            finite under the supported mixed-float16 policy.

        Raises:
            None.
        """
        return tf.where(
            self._observed_mask[tf.newaxis, :],
            tf.cast(-1e4, logits.dtype),
            logits,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable layer configuration.

        Args:
            None.

        Returns:
            dict[str, Any]: Base Keras configuration plus the observed indices
            and total output width.

        Raises:
            None.
        """
        config = super().get_config()
        config.update(
            {
                "observed_indices": list(self.observed_indices),
                "total_ports": self.total_ports,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="fluidra")
class SafeDecisionScores(tf.keras.layers.Layer):
    """Embed the non-regression safety policy in the saved model output."""

    def __init__(
        self,
        observed_indices: list[int] | tuple[int, ...],
        threshold: float,
        total_ports: int = TOTAL_PORTS,
        **kwargs: Any,
    ) -> None:
        """Store fixed masks used to compose final decision scores."""
        super().__init__(**kwargs)
        self.observed_indices = tuple(int(index) for index in observed_indices)
        self.threshold = float(threshold)
        self.total_ports = int(total_ports)

        observed_matrix = np.zeros(
            (len(self.observed_indices), self.total_ports),
            dtype=np.float32,
        )
        observed_matrix[np.arange(len(self.observed_indices)), self.observed_indices] = 1.0
        observed_mask = np.max(observed_matrix, axis=0)

        self._observed_matrix = tf.constant(observed_matrix, dtype=tf.float32)
        self._observed_mask = tf.constant(observed_mask, dtype=tf.float32)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Return scores whose argmax implements safe exploration."""
        observed_values, exploration_logits = inputs
        exploration_scores = tf.math.sigmoid(exploration_logits)

        observed_values = tf.cast(observed_values, exploration_scores.dtype)
        observed_scores = tf.where(
            observed_values > tf.cast(self.threshold, observed_values.dtype),
            2.0 + tf.math.sigmoid(observed_values),
            tf.cast(-2.0, observed_values.dtype),
        )
        observed_scores_full = tf.matmul(observed_scores, self._observed_matrix)
        return (
            exploration_scores * (1.0 - self._observed_mask[tf.newaxis, :])
            + observed_scores_full
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "observed_indices": list(self.observed_indices),
                "threshold": self.threshold,
                "total_ports": self.total_ports,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="fluidra")
class SparseImageObservedValues(tf.keras.layers.Layer):
    """Recover compact raw observations from the standard sparse image."""

    def __init__(
        self,
        observed_indices: list[int] | tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        """Store the physical ports extracted from the top-left 10x10 grid."""
        super().__init__(**kwargs)
        self.observed_indices = tuple(int(index) for index in observed_indices)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return channel-zero values at the configured physical ports."""
        physical_grid = inputs[:, :10, :10, 0]
        flattened_grid = tf.reshape(
            physical_grid,
            (tf.shape(inputs)[0], TOTAL_PORTS),
        )
        return tf.gather(
            flattened_grid,
            self.observed_indices,
            axis=1,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable layer configuration."""
        config = super().get_config()
        config.update({"observed_indices": list(self.observed_indices)})
        return config


def parse_args() -> RunOptions:
    """Parse compact ``--name=value`` command-line experiment options."""
    values = {
        "run-root": str(DEFAULT_RUN_ROOT),
        "observed-ports": "3,7",
        "samples-per-regime": "20000",
        "epochs": "100",
        "batch-size": "512",
        "selection-samples-per-regime": "5000",
        "splits": "train,val",
    }
    for token in sys.argv[1:]:
        if not token.startswith("--") or "=" not in token:
            raise ValueError(
                "Arguments must use --name=value syntax. "
                f"Invalid argument: {token!r}."
            )
        name, value = token[2:].split("=", maxsplit=1)
        if name not in values:
            raise ValueError(
                f"Unknown argument {name!r}. Valid names are {sorted(values)}."
            )
        values[name] = value

    observed_ports = [
        int(value)
        for value in values["observed-ports"].split(",")
        if value
    ]
    splits = [
        value
        for value in values["splits"].split(",")
        if value
    ]
    invalid_splits = sorted(set(splits).difference(("train", "val", "test")))
    if invalid_splits:
        raise ValueError(f"Unsupported split names: {invalid_splits}.")
    if any(value <= 0 or value >= TOTAL_PORTS for value in observed_ports):
        raise ValueError("observed-ports values must be in [1, 99].")
    return RunOptions(
        run_root=Path(values["run-root"]),
        observed_ports=observed_ports,
        samples_per_regime=int(values["samples-per-regime"]),
        epochs=int(values["epochs"]),
        batch_size=int(values["batch-size"]),
        selection_samples_per_regime=int(
            values["selection-samples-per-regime"]
        ),
        splits=splits,
    )


def group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest small GroupNorm group count dividing channels."""
    groups = min(int(channels), int(maximum_groups))
    while channels % groups != 0:
        groups -= 1
    return groups


def convolution_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    name: str,
    activation: str = "relu6",
) -> tf.Tensor:
    """Build one normalized MobileNet convolution and activation."""
    outputs = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(inputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(filters),
        axis=-1,
        name=f"{name}_gn",
    )(outputs)
    if activation == "relu6":
        return tf.keras.layers.ReLU(
            max_value=6.0,
            name=f"{name}_relu",
        )(outputs)
    return tf.keras.layers.Activation(
        activation,
        name=f"{name}_activation",
    )(outputs)


def depthwise_block(
    inputs: tf.Tensor,
    pointwise_filters: int,
    strides: int,
    block_index: int,
) -> tf.Tensor:
    """Build one MobileNetV1 depthwise-separable GroupNorm block."""
    input_channels = int(inputs.shape[-1])
    outputs = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"block_{block_index}_depthwise",
    )(inputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(input_channels),
        axis=-1,
        name=f"block_{block_index}_depthwise_gn",
    )(outputs)
    outputs = tf.keras.layers.ReLU(
        max_value=6.0,
        name=f"block_{block_index}_depthwise_relu",
    )(outputs)
    outputs = tf.keras.layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        use_bias=False,
        name=f"block_{block_index}_pointwise",
    )(outputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(pointwise_filters),
        axis=-1,
        name=f"block_{block_index}_pointwise_gn",
    )(outputs)
    return tf.keras.layers.ReLU(
        max_value=6.0,
        name=f"block_{block_index}_pointwise_relu",
    )(outputs)


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round one MobileNet channel count without shrinking it excessively."""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def inverted_residual_block(
    inputs: tf.Tensor,
    expansion: float,
    pointwise_filters: int,
    strides: int,
    block_index: int,
) -> tf.Tensor:
    """Build one MobileNetV2 inverted-residual block with GroupNorm."""
    input_channels = int(inputs.shape[-1])
    expanded_channels = make_divisible(input_channels * expansion)
    outputs = inputs
    if expansion != 1.0:
        outputs = tf.keras.layers.Conv2D(
            expanded_channels,
            kernel_size=1,
            use_bias=False,
            name=f"v2_block_{block_index}_expand",
        )(outputs)
        outputs = tf.keras.layers.GroupNormalization(
            groups=group_count(expanded_channels),
            axis=-1,
            name=f"v2_block_{block_index}_expand_gn",
        )(outputs)
        outputs = tf.keras.layers.ReLU(
            max_value=6.0,
            name=f"v2_block_{block_index}_expand_relu",
        )(outputs)

    outputs = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"v2_block_{block_index}_depthwise",
    )(outputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(expanded_channels),
        axis=-1,
        name=f"v2_block_{block_index}_depthwise_gn",
    )(outputs)
    outputs = tf.keras.layers.ReLU(
        max_value=6.0,
        name=f"v2_block_{block_index}_depthwise_relu",
    )(outputs)
    outputs = tf.keras.layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        use_bias=False,
        name=f"v2_block_{block_index}_project",
    )(outputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(pointwise_filters),
        axis=-1,
        name=f"v2_block_{block_index}_project_gn",
    )(outputs)
    if strides == 1 and input_channels == pointwise_filters:
        outputs = tf.keras.layers.Add(
            name=f"v2_block_{block_index}_add"
        )([inputs, outputs])
    return outputs


def squeeze_excite_block(
    inputs: tf.Tensor,
    squeeze_channels: int,
    block_index: int,
) -> tf.Tensor:
    """Apply the MobileNetV3 squeeze-and-excitation operation."""
    input_channels = int(inputs.shape[-1])
    excitation = tf.keras.layers.GlobalAveragePooling2D(
        keepdims=True,
        name=f"v3_block_{block_index}_se_pool",
    )(inputs)
    excitation = tf.keras.layers.Conv2D(
        squeeze_channels,
        kernel_size=1,
        activation="relu",
        name=f"v3_block_{block_index}_se_reduce",
    )(excitation)
    excitation = tf.keras.layers.Conv2D(
        input_channels,
        kernel_size=1,
        activation="hard_sigmoid",
        name=f"v3_block_{block_index}_se_expand",
    )(excitation)
    return tf.keras.layers.Multiply(
        name=f"v3_block_{block_index}_se_scale"
    )([inputs, excitation])


def mobilenet_v3_block(
    inputs: tf.Tensor,
    expansion_filters: int,
    pointwise_filters: int,
    kernel_size: int,
    strides: int,
    use_squeeze_excite: bool,
    activation: str,
    block_index: int,
) -> tf.Tensor:
    """Build one MobileNetV3Small inverted-residual block with GroupNorm."""
    input_channels = int(inputs.shape[-1])
    outputs = tf.keras.layers.Conv2D(
        expansion_filters,
        kernel_size=1,
        use_bias=False,
        name=f"v3_block_{block_index}_expand",
    )(inputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(expansion_filters),
        axis=-1,
        name=f"v3_block_{block_index}_expand_gn",
    )(outputs)
    outputs = tf.keras.layers.Activation(
        activation,
        name=f"v3_block_{block_index}_expand_activation",
    )(outputs)
    outputs = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"v3_block_{block_index}_depthwise",
    )(outputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(expansion_filters),
        axis=-1,
        name=f"v3_block_{block_index}_depthwise_gn",
    )(outputs)
    outputs = tf.keras.layers.Activation(
        activation,
        name=f"v3_block_{block_index}_depthwise_activation",
    )(outputs)
    if use_squeeze_excite:
        outputs = squeeze_excite_block(
            inputs=outputs,
            squeeze_channels=max(8, make_divisible(expansion_filters / 4)),
            block_index=block_index,
        )
    outputs = tf.keras.layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        use_bias=False,
        name=f"v3_block_{block_index}_project",
    )(outputs)
    outputs = tf.keras.layers.GroupNormalization(
        groups=group_count(pointwise_filters),
        axis=-1,
        name=f"v3_block_{block_index}_project_gn",
    )(outputs)
    if strides == 1 and input_channels == pointwise_filters:
        outputs = tf.keras.layers.Add(
            name=f"v3_block_{block_index}_add"
        )([inputs, outputs])
    return outputs


def build_variant_backbone(
    images: tf.Tensor,
    variant_name: str,
) -> tf.Tensor:
    """Build the requested alpha-0.2 MobileNet-style GroupNorm backbone."""
    if variant_name not in VARIANT_METADATA:
        raise ValueError(
            f"Unsupported safe-selector variant {variant_name!r}. "
            f"Expected one of {sorted(VARIANT_METADATA)}."
        )

    stem_activation = (
        "hard_swish" if variant_name == "mobv3_alpha02" else "relu6"
    )
    outputs = convolution_block(
        images,
        filters=8,
        strides=2,
        name="stem",
        activation=stem_activation,
    )
    if variant_name == "mobv1_alpha02":
        schedule = (
            (16, 1),
            (24, 2),
            (24, 1),
            (48, 2),
            (48, 1),
            (96, 2),
            (96, 1),
            (96, 1),
            (96, 1),
            (160, 2),
            (160, 1),
        )
        for block_index, (filters, strides) in enumerate(schedule):
            outputs = depthwise_block(
                outputs,
                pointwise_filters=filters,
                strides=strides,
                block_index=block_index,
            )
        return outputs

    if variant_name == "mobv2_alpha02":
        block_specs = (
            (1.0, 16, 1, 1),
            (6.0, 24, 2, 2),
            (6.0, 32, 3, 2),
            (6.0, 64, 4, 2),
            (6.0, 96, 3, 1),
            (6.0, 160, 3, 2),
            (6.0, 320, 1, 1),
        )
        block_index = 0
        for expansion, base_filters, repeats, first_stride in block_specs:
            pointwise_filters = make_divisible(base_filters * 0.2)
            for repeat_index in range(repeats):
                outputs = inverted_residual_block(
                    inputs=outputs,
                    expansion=expansion,
                    pointwise_filters=pointwise_filters,
                    strides=first_stride if repeat_index == 0 else 1,
                    block_index=block_index,
                )
                block_index += 1
        return convolution_block(
            outputs,
            filters=make_divisible(1280 * 0.2),
            strides=1,
            name="v2_final",
        )

    block_specs = (
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
    for block_index, (
        base_expansion,
        base_filters,
        kernel_size,
        strides,
        use_squeeze_excite,
        activation,
    ) in enumerate(block_specs):
        outputs = mobilenet_v3_block(
            inputs=outputs,
            expansion_filters=make_divisible(base_expansion * 0.2),
            pointwise_filters=make_divisible(base_filters * 0.2),
            kernel_size=kernel_size,
            strides=strides,
            use_squeeze_excite=use_squeeze_excite,
            activation=activation,
            block_index=block_index,
        )
    return convolution_block(
        outputs,
        filters=make_divisible(576 * 0.2),
        strides=1,
        name="v3_final",
        activation="hard_swish",
    )


def build_models(
    observed_indices: np.ndarray,
    log_scale: float,
    train_seed: int,
    variant_name: str = "mobv1_alpha02",
    fixed_batch_size: int | None = None,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build shared training-logit and decision-score variant models.

    Args:
        observed_indices (np.ndarray): Sorted zero-based observed ports.
        log_scale (float): Positive internal ``log1p`` normalization scale.
        train_seed (int): Seed for the exploration-logit initializer.
        variant_name (str): Supported alpha-0.2 MobileNet variant. Defaults to
            ``mobv1_alpha02``.
        fixed_batch_size (int | None): Exact batch size for deployment export.
            ``None`` preserves dynamic training batches; a positive value emits
            static batch-one-compatible broadcasts for TFLM.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]: Explorer-logit and decision-score
        models sharing one MobileNet backbone.

    Raises:
        ValueError: If the variant or fixed batch size is unsupported.
    """
    if fixed_batch_size is not None and fixed_batch_size <= 0:
        raise ValueError("fixed_batch_size must be positive when supplied.")
    input_kwargs: dict[str, Any] = {"name": "observed_snr"}
    if fixed_batch_size is None:
        input_kwargs["shape"] = (len(observed_indices),)
    else:
        input_kwargs["batch_shape"] = (fixed_batch_size, len(observed_indices))
    inputs = tf.keras.layers.Input(**input_kwargs)
    images = ExpandedPortImage(
        observed_indices=observed_indices.tolist(),
        log_scale=log_scale,
        fixed_batch_size=fixed_batch_size,
        name="expanded_port_image",
    )(inputs)

    outputs = build_variant_backbone(images, variant_name)
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(outputs)
    outputs = tf.keras.layers.Dense(64, activation="relu", name="selector_dense")(outputs)

    exploration_logits = tf.keras.layers.Dense(
        TOTAL_PORTS,
        activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=train_seed),
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
        name=f"{variant_name}_gn_explorer",
    )
    decision_scores = SafeDecisionScores(
        observed_indices=observed_indices.tolist(),
        threshold=OUTAGE_THRESHOLD,
        name="safe_decision_scores",
    )((inputs, exploration_logits))
    decision_model = tf.keras.Model(
        inputs=inputs,
        outputs=decision_scores,
        name=f"{variant_name}_gn_safe_selector",
    )
    return explorer_model, decision_model


def build_protocol_decision_model(
    compact_decision_model: tf.keras.Model,
    observed_indices: np.ndarray,
    variant_name: str,
    fixed_batch_size: int | None = None,
) -> tf.keras.Model:
    """Wrap the compact selector in the original sparse-image API.

    Args:
        compact_decision_model (tf.keras.Model): Compact selector accepting the
            observed-port vector.
        observed_indices (np.ndarray): Sorted zero-based observed ports used to
            extract values from the sparse image.
        variant_name (str): Variant name used in the saved model name.
        fixed_batch_size (int | None): Exact deployment batch size. ``None``
            keeps the original dynamic protocol input; a positive value must
            match the compact model and exports a static batch shape.

    Returns:
        tf.keras.Model: Sparse ``32 x 32 x 3`` protocol model.

    Raises:
        ValueError: If ``fixed_batch_size`` is not positive when supplied.
    """
    if fixed_batch_size is not None and fixed_batch_size <= 0:
        raise ValueError("fixed_batch_size must be positive when supplied.")
    input_kwargs: dict[str, Any] = {"name": "sparse_port_image"}
    if fixed_batch_size is None:
        input_kwargs["shape"] = (IMAGE_SIDE, IMAGE_SIDE, 3)
    else:
        input_kwargs["batch_shape"] = (
            fixed_batch_size,
            IMAGE_SIDE,
            IMAGE_SIDE,
            3,
        )
    image_inputs = tf.keras.layers.Input(**input_kwargs)
    observed_values = SparseImageObservedValues(
        observed_indices=observed_indices.tolist(),
        name="sparse_image_observed_values",
    )(image_inputs)
    decision_scores = compact_decision_model(observed_values)
    return tf.keras.Model(
        inputs=image_inputs,
        outputs=decision_scores,
        name=f"{variant_name}_gn_safe_selector_protocol",
    )


def build_recovery_training_data(
    y_values: np.ndarray,
    num_observed_ports: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build equal-mass targets for recoverable unobserved-port decisions."""
    observed_values, observed_indices = get_observed_ports(
        y_values,
        num_observed_ports,
        TOTAL_PORTS,
    )
    observed_failed = np.max(observed_values, axis=1) <= OUTAGE_THRESHOLD
    unobserved_good = y_values > OUTAGE_THRESHOLD
    unobserved_good[:, observed_indices] = False
    recoverable = observed_failed & np.any(unobserved_good, axis=1)

    recovered_inputs = observed_values[recoverable].astype(np.float32)
    recovered_good = unobserved_good[recoverable].astype(np.float32)
    recovered_good /= np.sum(recovered_good, axis=1, keepdims=True)
    return recovered_inputs, recovered_good, observed_indices.astype(np.int64)


def balanced_diagnostic_subset(
    parts: tuple[np.ndarray, ...],
    samples_per_regime: int,
) -> np.ndarray:
    """Return a deterministic regime-balanced validation diagnostic subset."""
    selected_parts = [
        part[: min(len(part), int(samples_per_regime))]
        for part in parts
    ]
    return np.concatenate(selected_parts, axis=0).astype(np.float32)


def build_non_identity_permutations(
    num_observed_ports: int,
    seed: int,
    count: int = 3,
) -> tuple[np.ndarray, ...]:
    """Return deterministic, distinct permutations that are never identity."""
    if num_observed_ports < 2:
        raise ValueError("At least two observed ports are required.")
    if count <= 0:
        raise ValueError("count must be positive.")

    identity = np.arange(num_observed_ports, dtype=np.int64)
    rng = np.random.default_rng(seed)
    permutations: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    maximum_attempts = max(32, count * 16)
    for _ in range(maximum_attempts):
        permutation = rng.permutation(num_observed_ports).astype(np.int64)
        key = tuple(int(value) for value in permutation)
        if np.array_equal(permutation, identity) or key in seen:
            continue
        permutations.append(permutation)
        seen.add(key)
        if len(permutations) == count:
            return tuple(permutations)

    for shift in range(1, num_observed_ports):
        permutation = np.roll(identity, shift)
        key = tuple(int(value) for value in permutation)
        if key in seen:
            continue
        permutations.append(permutation)
        seen.add(key)
        if len(permutations) == count:
            break

    if not permutations:
        raise RuntimeError("Unable to construct a non-identity permutation.")
    return tuple(permutations)


def compute_selection_snapshot(
    decision_model: tf.keras.Model,
    y_values: np.ndarray,
    num_observed_ports: int,
    batch_size: int,
    permutation_seed: int,
) -> dict[str, float]:
    """Compute validation OP, diversity, and input-sensitivity diagnostics."""
    observed_values, observed_indices = get_observed_ports(
        y_values,
        num_observed_ports,
        TOTAL_PORTS,
    )
    scores = decision_model.predict(
        observed_values,
        batch_size=batch_size,
        verbose=0,
    )
    selected_ports = np.argmax(scores, axis=1)
    rows = np.arange(len(y_values))
    model_outage = y_values[rows, selected_ports] <= OUTAGE_THRESHOLD
    observed_outage = (
        np.max(y_values[:, observed_indices], axis=1) <= OUTAGE_THRESHOLD
    )

    permutation_change_rates = []
    for permutation in build_non_identity_permutations(
        num_observed_ports=num_observed_ports,
        seed=permutation_seed,
    ):
        permuted_scores = decision_model.predict(
            observed_values[:, permutation],
            batch_size=batch_size,
            verbose=0,
        )
        permuted_ports = np.argmax(permuted_scores, axis=1)
        permutation_change_rates.append(
            float(np.mean(selected_ports != permuted_ports))
        )
    return {
        "model_op": float(np.mean(model_outage)),
        "observed_op": float(np.mean(observed_outage)),
        "distinct_selected_ports": float(np.unique(selected_ports).size),
        "permutation_change_rate": float(np.mean(permutation_change_rates)),
    }


class SelectionCheckpoint(tf.keras.callbacks.Callback):
    """Checkpoint explorer weights using model-only validation OP."""

    def __init__(
        self,
        decision_model: tf.keras.Model,
        validation_values: np.ndarray,
        num_observed_ports: int,
        batch_size: int,
        checkpoint_path: Path,
        patience: int | None = None,
        permutation_seed: int = TRAIN_SEED,
    ) -> None:
        """Store validation data and the OP-first checkpoint policy.

        ``patience`` remains accepted for compatibility with archived
        experiment scripts, but checkpoint selection never stops training.
        """
        super().__init__()
        del patience
        self.decision_model = decision_model
        self.validation_values = validation_values
        self.num_observed_ports = int(num_observed_ports)
        self.batch_size = int(batch_size)
        self.checkpoint_path = checkpoint_path
        self.permutation_seed = int(permutation_seed)
        self.records: list[dict[str, float]] = []
        self.best_key: tuple[float, ...] | None = None
        self.best_epoch: int | None = None

    def _record(self, epoch: int) -> dict[str, float]:
        """Evaluate one checkpoint and append its selection diagnostics."""
        snapshot = compute_selection_snapshot(
            decision_model=self.decision_model,
            y_values=self.validation_values,
            num_observed_ports=self.num_observed_ports,
            batch_size=self.batch_size,
            permutation_seed=self.permutation_seed,
        )
        record = {"epoch": float(epoch), **snapshot}
        self.records.append(record)
        return record

    @staticmethod
    def _selection_key(record: dict[str, float]) -> tuple[float, ...]:
        """Prefer valid diverse models, then lower OP and higher sensitivity."""
        diagnostics_pass = (
            record["distinct_selected_ports"] >= 10.0
            and record["permutation_change_rate"] >= 0.05
        )
        return (
            0.0 if diagnostics_pass else 1.0,
            record["model_op"],
            -record["permutation_change_rate"],
        )

    def _consider_checkpoint(self, record: dict[str, float]) -> None:
        """Save weights when the OP-first validation key improves."""
        key = self._selection_key(record)
        if self.best_key is None or key < self.best_key:
            self.best_key = key
            self.best_epoch = int(record["epoch"])
            self.model.save_weights(self.checkpoint_path)

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        """Record and checkpoint the initialized training-only baseline."""
        del logs
        record = self._record(epoch=0)
        self._consider_checkpoint(record)

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, float] | None = None,
    ) -> None:
        """Evaluate the epoch and retain the best OP-first checkpoint."""
        record = self._record(epoch=epoch + 1)
        if logs is not None:
            logs["val_model_op"] = record["model_op"]
            logs["val_distinct_selected_ports"] = record["distinct_selected_ports"]
            logs["val_permutation_change_rate"] = record["permutation_change_rate"]
        self._consider_checkpoint(record)
        print(
            " - val_model_op: "
            f"{record['model_op']:.6f}"
            " - val_observed_op: "
            f"{record['observed_op']:.6f}"
            " - distinct: "
            f"{int(record['distinct_selected_ports'])}"
            " - permutation_change: "
            f"{record['permutation_change_rate']:.4f}"
        )
    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Restore the weights chosen by validation model-only OP."""
        del logs
        self.model.load_weights(self.checkpoint_path)


def paired_bootstrap_ci(
    model_outage: np.ndarray,
    observed_outage: np.ndarray,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float]:
    """Return an exact multinomial paired-bootstrap CI for OP difference."""
    differences = (
        model_outage.astype(np.int8) - observed_outage.astype(np.int8)
    )
    counts = np.asarray(
        [
            np.count_nonzero(differences == -1),
            np.count_nonzero(differences == 0),
            np.count_nonzero(differences == 1),
        ],
        dtype=np.int64,
    )
    probabilities = counts / np.sum(counts)
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(int(np.sum(counts)), probabilities, size=draws)
    deltas = (samples[:, 2] - samples[:, 0]) / np.sum(counts)
    lower, upper = np.quantile(deltas, (0.025, 0.975))
    return float(lower), float(upper)


def evaluate_regime(
    decision_model: tf.keras.Model,
    values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float | int], pd.DataFrame, np.ndarray, np.ndarray]:
    """Evaluate one regime and return metrics, samples, and outage pairs."""
    observed_values = values[:, observed_indices].astype(np.float32)
    scores = decision_model.predict(
        observed_values,
        batch_size=batch_size,
        verbose=0,
    )
    selected_ports = np.argmax(scores, axis=1).astype(np.int64)
    rows = np.arange(len(values))
    model_selected = values[rows, selected_ports]
    observed_selected = np.max(values[:, observed_indices], axis=1)
    ideal_selected = np.max(values, axis=1)
    model_outage = model_selected <= OUTAGE_THRESHOLD
    observed_outage = observed_selected <= OUTAGE_THRESHOLD
    ideal_outage = ideal_selected <= OUTAGE_THRESHOLD
    metrics: dict[str, float | int] = {
        "num_samples": int(len(values)),
        "ideal_op": float(np.mean(ideal_outage)),
        "observed_op": float(np.mean(observed_outage)),
        "model_only_op": float(np.mean(model_outage)),
        "model_minus_observed_op": float(
            np.mean(model_outage) - np.mean(observed_outage)
        ),
        "mean_selection_regret": float(np.mean(ideal_selected - model_selected)),
        "distinct_selected_ports": int(np.unique(selected_ports).size),
    }
    sample_count = min(100, len(values))
    predictions = pd.DataFrame(
        {
            "sample_index": np.arange(sample_count),
            "selected_port": selected_ports[:sample_count],
            "model_only_selected_value": model_selected[:sample_count],
            "observed_selected_value": observed_selected[:sample_count],
            "ideal_selected_value": ideal_selected[:sample_count],
        }
    )
    return metrics, predictions, model_outage, observed_outage


def evaluate_splits(
    decision_model: tf.keras.Model,
    split_parts: dict[str, tuple[np.ndarray, ...]],
    observed_indices: np.ndarray,
    batch_size: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate requested splits per regime and save OP result artifacts."""
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for split_name, parts in split_parts.items():
        split_model_outages: list[np.ndarray] = []
        split_observed_outages: list[np.ndarray] = []
        split_predictions: list[pd.DataFrame] = []
        for regime_label, values in zip(KMS_REGIME_LABELS, parts, strict=True):
            metrics, predictions, model_outage, observed_outage = evaluate_regime(
                decision_model=decision_model,
                values=values,
                observed_indices=observed_indices,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "split": split_name,
                    "regime": regime_label,
                    **metrics,
                }
            )
            predictions.insert(0, "regime", regime_label)
            split_predictions.append(predictions)
            split_model_outages.append(model_outage)
            split_observed_outages.append(observed_outage)

        model_outages = np.concatenate(split_model_outages)
        observed_outages = np.concatenate(split_observed_outages)
        lower, upper = paired_bootstrap_ci(
            model_outage=model_outages,
            observed_outage=observed_outages,
            seed=DATA_SEED,
        )
        split_frame = pd.DataFrame(
            [row for row in rows if row["split"] == split_name]
        )
        total_samples = int(split_frame["num_samples"].sum())
        aggregate = {
            "split": split_name,
            "regime": "ALL",
            "num_samples": total_samples,
            "ideal_op": float(
                np.average(split_frame["ideal_op"], weights=split_frame["num_samples"])
            ),
            "observed_op": float(
                np.average(split_frame["observed_op"], weights=split_frame["num_samples"])
            ),
            "model_only_op": float(np.mean(model_outages)),
            "model_minus_observed_op": float(
                np.mean(model_outages) - np.mean(observed_outages)
            ),
            "mean_selection_regret": float(
                np.average(
                    split_frame["mean_selection_regret"],
                    weights=split_frame["num_samples"],
                )
            ),
            "distinct_selected_ports": int(
                split_frame["distinct_selected_ports"].max()
            ),
        }
        rows.append(aggregate)
        summary[split_name] = {
            **aggregate,
            "paired_bootstrap_95_ci": [lower, upper],
            "ci_upper_below_zero": bool(upper < 0.0),
        }
        pd.concat(split_predictions, ignore_index=True).to_csv(
            output_dir / f"{split_name}_sample_predictions.csv",
            index=False,
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "op_results_by_split_and_regime.csv", index=False)
    return results, summary


def create_run_directories(run_dir: Path) -> dict[str, Path]:
    """Create a new non-overwriting artifact tree for one observation budget."""
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {run_dir}"
        )
    paths = {
        "args": run_dir / "model_training" / "args",
        "history": run_dir / "model_training" / "history",
        "model": run_dir / "model_training" / "model",
        "scaler": run_dir / "model_training" / "scaler",
        "evaluation": run_dir / "evaluation",
        "checkpoint": run_dir / "model_training" / "checkpoint",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return paths


def save_identity_transformers(
    scaler_dir: Path,
    observed_values: np.ndarray,
    target_values: np.ndarray,
) -> None:
    """Save explicit identity preprocessing artifacts for reproducibility."""
    scaler_x = FunctionTransformer(validate=True)
    scaler_y = FunctionTransformer(validate=True)
    scaler_x.fit(observed_values[: min(1024, len(observed_values))])
    scaler_y.fit(target_values[: min(1024, len(target_values))])
    with open(scaler_dir / "scaler_X.pkl", "wb") as scaler_file:
        pickle.dump(scaler_x, scaler_file)
    with open(scaler_dir / "scaler_y.pkl", "wb") as scaler_file:
        pickle.dump(scaler_y, scaler_file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one stable, human-readable JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_reload_diagnostics(
    reference_model: tf.keras.Model,
    reloaded_model: tf.keras.Model,
    inputs: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    """Compare saved-model outputs using the same graph inference path."""
    reference_scores = reference_model.predict(
        inputs,
        batch_size=batch_size,
        verbose=0,
    )
    reloaded_scores = reloaded_model.predict(
        inputs,
        batch_size=batch_size,
        verbose=0,
    )
    max_abs_difference = float(
        np.max(np.abs(reference_scores - reloaded_scores))
    )
    decision_change_rate = float(
        np.mean(
            np.argmax(reference_scores, axis=1)
            != np.argmax(reloaded_scores, axis=1)
        )
    )
    return max_abs_difference, decision_change_rate


def run_safe_selector_training_case(
    variant_name: str,
    run_dir: str | Path,
    epochs: int,
    batch_size: int,
    data_seed: int,
    train_seed: int,
    num_observed_ports: int,
    total_ports: int,
    train_regime_labels: tuple[str, ...],
    y_train: np.ndarray,
    y_val: np.ndarray,
    val_parts: tuple[np.ndarray, ...],
    steps_per_execution: int | str = 1,
    show_summary: bool = True,
    jit_compile: bool | str = False,
    precision_policy: str = "float32",
    enable_op_determinism: bool = True,
    learning_rate: float = 3e-4,
    selection_samples_per_regime: int = 5_000,
) -> dict[str, Path]:
    """Train one all-regime safe-selector case in the standard artifact tree."""
    # Keep orchestration dependencies out of model-only and PTQ smoke imports.
    # Full training still requires the same Araras runtime when this function runs.
    from araras.ml.optuna.utils import init_study_dirs
    from mobilenet.keras.common import resolve_training_steps_per_execution

    if variant_name not in VARIANT_METADATA:
        raise ValueError(
            f"Unsupported safe-selector variant {variant_name!r}. "
            f"Expected one of {sorted(VARIANT_METADATA)}."
        )
    if total_ports != TOTAL_PORTS:
        raise ValueError(
            f"The validated selector requires total_ports={TOTAL_PORTS}."
        )
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if selection_samples_per_regime <= 0:
        raise ValueError("selection_samples_per_regime must be positive.")

    run_dir_path = Path(run_dir)
    method_name = f"{variant_name}_gn_safe_selector"
    metadata = VARIANT_METADATA[variant_name]
    print(
        "[START] "
        f"method={method_name}, observed_ports={num_observed_ports}, "
        f"train_rows={len(y_train)}, val_rows={len(y_val)}"
    )
    tf.keras.mixed_precision.set_global_policy(precision_policy)
    tf.keras.utils.set_random_seed(train_seed)
    if enable_op_determinism:
        tf.config.experimental.enable_op_determinism()

    backup_dir: Path | None = None
    logs_dir: Path | None = None
    explorer_model: tf.keras.Model | None = None
    decision_model: tf.keras.Model | None = None
    protocol_model: tf.keras.Model | None = None

    try:
        study_paths = init_study_dirs(
            str(run_dir_path),
            study_name="model_training",
        )
        (
            study_dir,
            args_dir,
            fig_dir,
            backup_dir,
            history_dir,
            scaler_dir,
            model_dir,
            logs_dir,
            tensorboard_dir,
        ) = tuple(Path(path) for path in study_paths)
        del study_dir
        del fig_dir
        del tensorboard_dir

        _, observed_indices = get_observed_ports(
            y_train,
            num_observed_ports,
            total_ports,
        )
        unobserved_indices = np.setdiff1d(
            np.arange(total_ports, dtype=np.int64),
            observed_indices,
        )
        train_inputs, train_targets, checked_indices = (
            build_recovery_training_data(
                y_values=y_train,
                num_observed_ports=num_observed_ports,
            )
        )
        val_inputs, val_targets, _ = build_recovery_training_data(
            y_values=y_val,
            num_observed_ports=num_observed_ports,
        )
        if not np.array_equal(observed_indices, checked_indices):
            raise ValueError("Observed indices changed during data preparation.")
        if len(train_inputs) == 0 or len(val_inputs) == 0:
            raise ValueError("No recoverable samples exist for selector training.")

        scale_stride = max(1, len(y_train) // 100_000)
        log_scale = float(
            np.quantile(
                np.log1p(y_train[::scale_stride]),
                0.999,
            )
        )
        explorer_model, decision_model = build_models(
            observed_indices=observed_indices,
            log_scale=log_scale,
            train_seed=train_seed,
            variant_name=variant_name,
        )
        resolved_steps_per_execution = resolve_training_steps_per_execution(
            name=["x", "validation_data"],
            data_size=[len(train_inputs), len(val_inputs)],
            batch_size=batch_size,
            steps_per_execution=steps_per_execution,
        )
        explorer_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(
                    name="target_argmax_accuracy"
                ),
            ],
            steps_per_execution=resolved_steps_per_execution,
            jit_compile=jit_compile,
        )
        if show_summary:
            explorer_model.summary()

        validation_diagnostic = balanced_diagnostic_subset(
            parts=val_parts,
            samples_per_regime=selection_samples_per_regime,
        )
        checkpoint_dir = backup_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        selection_checkpoint = SelectionCheckpoint(
            decision_model=decision_model,
            validation_values=validation_diagnostic,
            num_observed_ports=num_observed_ports,
            batch_size=batch_size,
            checkpoint_path=checkpoint_dir / "best.weights.h5",
            permutation_seed=train_seed,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=10,
            factor=0.3,
            min_lr=1e-6,
            verbose=1,
        )
        backup_callback = tf.keras.callbacks.BackupAndRestore(
            backup_dir=backup_dir / "training",
            save_freq="epoch",
            delete_checkpoint=True,
        )

        initial_config: dict[str, Any] = {
            "variant_name": variant_name,
            "method_name": method_name,
            "architecture": metadata["architecture"],
            "backbone_family": metadata["backbone_family"],
            "alpha": metadata["alpha"],
            "input_representation": (
                "standard_sparse_32x32_input_then_compact_raw_"
                "observations_with_internal_expanded_32x32_"
                "value_mask_coordinate_image"
            ),
            "input_shape": [IMAGE_SIDE, IMAGE_SIDE, 3],
            "compact_selector_input_shape": [num_observed_ports],
            "total_ports": total_ports,
            "num_observed_ports": num_observed_ports,
            "observed_indices": observed_indices.tolist(),
            "unobserved_indices": unobserved_indices.tolist(),
            "unobserved_port_count": int(len(unobserved_indices)),
            "threshold": OUTAGE_THRESHOLD,
            "target": (
                "uniform_distribution_over_non_outage_unobserved_ports"
            ),
            "decision_policy": (
                "choose a known non-outage observed port; otherwise use "
                "MobileNet exploration scores"
            ),
            "normalization": "internal_shared_log1p_and_group_normalization",
            "log_scale": log_scale,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "early_stopping_enabled": False,
            "reduce_lr_patience": 10,
            "checkpoint_metric": "validation_model_only_op",
            "schema_version": 3,
            "task_type": "safe_port_selector",
            "output_semantics": "decision_scores",
            "learning_rate": learning_rate,
            "steps_per_execution": resolved_steps_per_execution,
            "jit_compile": jit_compile,
            "precision_policy": precision_policy,
            "enable_op_determinism": enable_op_determinism,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "samples_per_regime": None,
            "split_mode": "60_20_20",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "unused_ratio": 0.0,
            "train_regime_labels": list(train_regime_labels),
            "train_recoverable_samples": int(len(train_inputs)),
            "val_recoverable_samples": int(len(val_inputs)),
            "selection_samples_per_regime": selection_samples_per_regime,
            "op_evaluation": "separate_pass_required",
        }
        config_path = args_dir / "mobilenet_config.json"
        write_json(config_path, initial_config)

        history = explorer_model.fit(
            train_inputs,
            train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[
                selection_checkpoint,
                reduce_lr,
                backup_callback,
            ],
            verbose=2,
        )

        protocol_model = build_protocol_decision_model(
            compact_decision_model=decision_model,
            observed_indices=observed_indices,
            variant_name=variant_name,
        )
        model_path = model_dir / "model.keras"
        compact_model_path = model_dir / "compact_decision_model.keras"
        explorer_model_path = model_dir / "explorer_model.keras"
        protocol_model.save(model_path)
        decision_model.save(compact_model_path)
        explorer_model.save(explorer_model_path)
        save_identity_transformers(
            scaler_dir=scaler_dir,
            observed_values=train_inputs,
            target_values=y_train,
        )

        history_frame = pd.DataFrame(history.history)
        history_frame.insert(
            0,
            "epoch",
            np.arange(1, len(history_frame) + 1),
        )
        history_path = history_dir / "history.csv"
        selection_history_path = history_dir / "selection_history.csv"
        history_frame.to_csv(history_path, index=False)
        pd.DataFrame(selection_checkpoint.records).to_csv(
            selection_history_path,
            index=False,
        )

        diagnostic_compact_inputs = validation_diagnostic[
            : min(2048, len(validation_diagnostic)),
            observed_indices,
        ]
        diagnostic_protocol_inputs = build_sparse_port_images(
            observed_values=diagnostic_compact_inputs,
            observed_indices=observed_indices,
            total_ports=total_ports,
            image_height=IMAGE_SIDE,
            image_width=IMAGE_SIDE,
        )
        inference_scores = protocol_model(
            diagnostic_protocol_inputs,
            training=False,
        ).numpy()
        training_scores = protocol_model(
            diagnostic_protocol_inputs,
            training=True,
        ).numpy()
        inference_training_max_abs = float(
            np.max(np.abs(inference_scores - training_scores))
        )
        reloaded_model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )
        reload_max_abs, reload_decision_change_rate = (
            compute_reload_diagnostics(
                reference_model=protocol_model,
                reloaded_model=reloaded_model,
                inputs=diagnostic_protocol_inputs,
                batch_size=batch_size,
            )
        )
        if reload_max_abs > 1e-2 or reload_decision_change_rate > 0.005:
            raise ValueError(
                "Saved-model reload changed graph-mode selector predictions "
                "beyond tolerance: "
                f"max_abs={reload_max_abs}, "
                f"decision_change_rate={reload_decision_change_rate}."
            )
        final_snapshot = compute_selection_snapshot(
            decision_model=decision_model,
            y_values=validation_diagnostic,
            num_observed_ports=num_observed_ports,
            batch_size=batch_size,
            permutation_seed=train_seed,
        )
        final_config = {
            **initial_config,
            "epochs_completed": len(history_frame),
            "selected_epoch": selection_checkpoint.best_epoch,
            "inference_training_max_abs_difference": (
                inference_training_max_abs
            ),
            "reload_max_abs_difference": reload_max_abs,
            "reload_decision_change_rate": reload_decision_change_rate,
            "final_validation_diagnostic": final_snapshot,
        }
        write_json(config_path, final_config)

        stats_path = args_dir / "model_stats.txt"
        stats_path.write_text(
            "\n".join(
                (
                    f"variant_name={variant_name}",
                    f"method_name={method_name}",
                    (
                        "decision_model_parameters="
                        f"{protocol_model.count_params()}"
                    ),
                    (
                        "explorer_model_parameters="
                        f"{explorer_model.count_params()}"
                    ),
                    f"epochs_completed={len(history_frame)}",
                    f"selected_epoch={selection_checkpoint.best_epoch}",
                    (
                        "inference_training_max_abs_difference="
                        f"{inference_training_max_abs}"
                    ),
                    f"reload_max_abs_difference={reload_max_abs}",
                    (
                        "reload_decision_change_rate="
                        f"{reload_decision_change_rate}"
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[DONE] Saved training artifacts to {run_dir_path}")
        return {
            "config_path": config_path,
            "model_path": model_path,
            "compact_model_path": compact_model_path,
            "explorer_model_path": explorer_model_path,
            "history_path": history_path,
            "selection_history_path": selection_history_path,
            "stats_path": stats_path,
        }
    except Exception as error:
        print(f"[ERROR] Training failed: {error}")
        traceback.print_exc()
        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "training_error.log").write_text(
                f"{error}\n\n{traceback.format_exc()}\n",
                encoding="utf-8",
            )
        raise
    finally:
        del explorer_model
        del decision_model
        del protocol_model
        tf.keras.backend.clear_session()
        gc.collect()
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        if (
            logs_dir is not None
            and logs_dir.exists()
            and not any(logs_dir.iterdir())
        ):
            logs_dir.rmdir()


def train_one_budget(
    y_train: np.ndarray,
    y_val: np.ndarray,
    split_parts: dict[str, tuple[np.ndarray, ...]],
    num_observed_ports: int,
    args: RunOptions,
) -> dict[str, Any]:
    """Train, select, save, reload, and evaluate one observation budget."""
    run_dir = args.run_root / f"{num_observed_ports}_observed_ports"
    paths = create_run_directories(run_dir)
    _, observed_indices = get_observed_ports(
        y_train,
        num_observed_ports,
        TOTAL_PORTS,
    )
    unobserved_indices = np.setdiff1d(
        np.arange(TOTAL_PORTS, dtype=np.int64),
        observed_indices,
    )
    train_inputs, train_targets, checked_indices = build_recovery_training_data(
        y_values=y_train,
        num_observed_ports=num_observed_ports,
    )
    val_inputs, val_targets, _ = build_recovery_training_data(
        y_values=y_val,
        num_observed_ports=num_observed_ports,
    )
    if not np.array_equal(observed_indices, checked_indices):
        raise ValueError("Observed indices changed while building training data.")
    if len(train_inputs) == 0 or len(val_inputs) == 0:
        raise ValueError("No recoverable samples exist for selector training.")

    scale_stride = max(1, len(y_train) // 100_000)
    log_scale = float(
        np.quantile(
            np.log1p(y_train[::scale_stride]),
            0.999,
        )
    )
    explorer_model, decision_model = build_models(
        observed_indices=observed_indices,
        log_scale=log_scale,
        train_seed=TRAIN_SEED,
    )
    explorer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="target_argmax_accuracy"),
        ],
    )

    validation_diagnostic = balanced_diagnostic_subset(
        parts=split_parts["val"],
        samples_per_regime=args.selection_samples_per_regime,
    )
    checkpoint = SelectionCheckpoint(
        decision_model=decision_model,
        validation_values=validation_diagnostic,
        num_observed_ports=num_observed_ports,
        batch_size=max(args.batch_size, 1024),
        checkpoint_path=paths["checkpoint"] / "best.weights.h5",
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        patience=10,
        factor=0.3,
        min_lr=1e-6,
        verbose=1,
    )
    history = explorer_model.fit(
        train_inputs,
        train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[checkpoint, reduce_lr],
        verbose=2,
    )

    decision_model_path = paths["model"] / "model.keras"
    explorer_model_path = paths["model"] / "explorer_model.keras"
    decision_model.save(decision_model_path)
    explorer_model.save(explorer_model_path)
    save_identity_transformers(
        scaler_dir=paths["scaler"],
        observed_values=train_inputs,
        target_values=y_train,
    )

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(paths["history"] / "history.csv", index=False)
    pd.DataFrame(checkpoint.records).to_csv(
        paths["history"] / "selection_history.csv",
        index=False,
    )

    diagnostic_inputs = validation_diagnostic[
        : min(2048, len(validation_diagnostic)),
        observed_indices,
    ]
    inference_scores = decision_model(
        diagnostic_inputs,
        training=False,
    ).numpy()
    training_scores = decision_model(
        diagnostic_inputs,
        training=True,
    ).numpy()
    inference_training_max_abs = float(
        np.max(np.abs(inference_scores - training_scores))
    )

    reloaded_model = tf.keras.models.load_model(
        decision_model_path,
        compile=False,
    )
    reload_max_abs, reload_decision_change_rate = (
        compute_reload_diagnostics(
            reference_model=decision_model,
            reloaded_model=reloaded_model,
            inputs=diagnostic_inputs,
            batch_size=args.batch_size,
        )
    )
    if reload_max_abs > 2e-3 or reload_decision_change_rate > 0.005:
        raise ValueError(
            "Saved-model reload changed graph-mode selector predictions "
            "beyond tolerance: "
            f"max_abs={reload_max_abs}, "
            f"decision_change_rate={reload_decision_change_rate}."
        )

    selected_split_parts = {
        name: split_parts[name]
        for name in args.splits
    }
    _, evaluation_summary = evaluate_splits(
        decision_model=reloaded_model,
        split_parts=selected_split_parts,
        observed_indices=observed_indices,
        batch_size=max(args.batch_size, 1024),
        output_dir=paths["evaluation"],
    )
    final_snapshot = compute_selection_snapshot(
        decision_model=reloaded_model,
        y_values=validation_diagnostic,
        num_observed_ports=num_observed_ports,
        batch_size=max(args.batch_size, 1024),
        permutation_seed=TRAIN_SEED,
    )

    config = {
        "variant_name": "mobv1_gn_safe_selector",
        "architecture": "MobileNetV1-style depthwise-separable GroupNorm",
        "input_representation": "expanded_full_32x32_value_mask_coordinate",
        "input_shape": [num_observed_ports],
        "total_ports": TOTAL_PORTS,
        "num_observed_ports": num_observed_ports,
        "observed_indices": observed_indices.tolist(),
        "unobserved_indices": unobserved_indices.tolist(),
        "unobserved_port_count": int(len(unobserved_indices)),
        "threshold": OUTAGE_THRESHOLD,
        "target": "uniform_distribution_over_non_outage_unobserved_ports",
        "decision_policy": (
            "choose a known non-outage observed port; otherwise use "
            "MobileNet exploration scores"
        ),
        "normalization": "internal_shared_log1p",
        "log_scale": log_scale,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_frame),
        "selected_epoch": checkpoint.best_epoch,
        "early_stopping_enabled": False,
        "reduce_lr_patience": 10,
        "checkpoint_metric": "validation_model_only_op",
        "schema_version": 3,
        "task_type": "safe_port_selector",
        "output_semantics": "decision_scores",
        "learning_rate": 3e-4,
        "data_seed": DATA_SEED,
        "train_seed": TRAIN_SEED,
        "samples_per_regime": (
            None if args.samples_per_regime == 0 else args.samples_per_regime
        ),
        "train_recoverable_samples": int(len(train_inputs)),
        "val_recoverable_samples": int(len(val_inputs)),
        "selection_samples_per_regime": args.selection_samples_per_regime,
        "evaluation_splits": list(args.splits),
        "inference_training_max_abs_difference": inference_training_max_abs,
        "reload_max_abs_difference": reload_max_abs,
        "reload_decision_change_rate": reload_decision_change_rate,
        "final_validation_diagnostic": final_snapshot,
        "evaluation_summary": evaluation_summary,
    }
    (paths["args"] / "mobilenet_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths["evaluation"] / "summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths["args"] / "model_stats.json").write_text(
        json.dumps(
            {
                "decision_model_parameters": int(decision_model.count_params()),
                "explorer_model_parameters": int(explorer_model.count_params()),
                "inference_training_max_abs_difference": inference_training_max_abs,
                "reload_max_abs_difference": reload_max_abs,
                "reload_decision_change_rate": reload_decision_change_rate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def main() -> None:
    """Run the requested observation budgets in one fresh experiment root."""
    args = parse_args()
    if args.samples_per_regime < 0:
        raise ValueError("samples_per_regime must be zero or positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if args.selection_samples_per_regime <= 0:
        raise ValueError("selection_samples_per_regime must be positive.")

    tf.keras.utils.set_random_seed(TRAIN_SEED)
    tf.config.experimental.enable_op_determinism()
    samples_per_regime = (
        None if args.samples_per_regime == 0 else args.samples_per_regime
    )
    (
        y_train,
        y_val,
        y_test,
        train_parts,
        val_parts,
        test_parts,
    ) = split_kms_60_20_20_cache(
        seed=DATA_SEED,
        regime_labels=KMS_REGIME_LABELS,
        samples_per_regime=samples_per_regime,
        verbose=True,
    )
    all_parts = {
        "train": train_parts,
        "val": val_parts,
        "test": test_parts,
    }
    summaries: dict[str, Any] = {}
    for num_observed_ports in args.observed_ports:
        print(f"[RUN] num_observed_ports={num_observed_ports}")
        summaries[str(num_observed_ports)] = train_one_budget(
            y_train=y_train,
            y_val=y_val,
            split_parts=all_parts,
            num_observed_ports=num_observed_ports,
            args=args,
        )
        tf.keras.backend.clear_session()
        gc.collect()

    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "experiment_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del y_train
    del y_val
    del y_test
    gc.collect()


if __name__ == "__main__":
    main()
