"""Variant registry for the supported MobileNet backbones."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from typing import Any

import tensorflow as tf


_VARIANT_METADATA: dict[str, dict[str, Any]] = {
    "mobv1_alpha02": {
        "backbone_family": "MobileNet",
        "alpha": 0.2,
    },
    "mobv2_alpha02": {
        "backbone_family": "MobileNetV2",
        "alpha": 0.2,
    },
    "mobv3_alpha02": {
        "backbone_family": "MobileNetV3Small",
        "alpha": 0.2,
    },
}


def get_supported_variant_names() -> tuple[str, ...]:
    """Return the supported MobileNet variant names in canonical order.

    Args:
        None.

    Returns:
        tuple[str, ...]: Supported variant names. The returned order is stable
            and is intended for validation messages and iteration.

    Raises:
        None.
    """
    return tuple(_VARIANT_METADATA.keys())


def get_variant_metadata(variant_name: str) -> dict[str, Any]:
    """Return immutable-style metadata for one supported MobileNet variant.

    Args:
        variant_name (str): Supported MobileNet variant selector.
            If ``"mobv1_alpha02"``, the variant uses MobileNetV1 with
            ``alpha=0.2`` and a full convolutional backbone.
            If ``"mobv2_alpha02"``, the variant uses MobileNetV2 with
            ``alpha=0.2`` and a full convolutional backbone.
            If ``"mobv3_alpha02"``, the variant uses MobileNetV3Small with
            ``alpha=0.2`` and no built-in preprocessing layer.
            Any other value is invalid and causes an explicit failure.

    Returns:
        dict[str, Any]: Shallow copy of the metadata for the requested variant.
            The mapping always contains ``backbone_family`` and ``alpha``.

    Raises:
        ValueError: If ``variant_name`` is not one of the supported variants.

    Examples:
        >>> get_variant_metadata("mobv3_alpha02")["alpha"]
        0.2
    """
    if variant_name not in _VARIANT_METADATA:
        raise ValueError(
            "Unsupported MobileNet variant "
            f"{variant_name!r}. Valid options are {list(get_supported_variant_names())}."
        )

    return dict(_VARIANT_METADATA[variant_name])


def build_backbone_model(
    variant_name: str,
    input_shape: tuple[int, int, int],
) -> tf.keras.Model:
    """Build one supported MobileNet backbone with ``weights=None``.

    Args:
        variant_name (str): Supported MobileNet variant selector.
            If ``"mobv1_alpha02"``, builds ``tf.keras.applications.MobileNet``
            with ``alpha=0.2``.
            If ``"mobv2_alpha02"``, builds ``tf.keras.applications.MobileNetV2``
            with ``alpha=0.2``.
            If ``"mobv3_alpha02"``, builds
            ``tf.keras.applications.MobileNetV3Small`` with ``alpha=0.2`` and
            ``include_preprocessing=False``.
            Any other value is invalid and causes an explicit failure.
        input_shape (tuple[int, int, int]): Static spatial input shape passed
            directly to the Keras application constructor. This project expects
            ``(32, 32, 3)`` so sparse observed-port value, mask, and coordinate
            channels can be consumed directly by MobileNet.

    Returns:
        tf.keras.Model: Untrained backbone model with ``include_top=False`` and
            ``weights=None``.

    Raises:
        ValueError: If ``variant_name`` is not supported.

    Examples:
        >>> model = build_backbone_model("mobv1_alpha02", (32, 32, 3))
        >>> model.input_shape
        (None, 32, 32, 3)
    """
    if variant_name == "mobv1_alpha02":
        return tf.keras.applications.MobileNet(
            input_shape=input_shape,
            alpha=0.2,
            include_top=False,
            weights=None,
        )

    if variant_name == "mobv2_alpha02":
        return tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=0.2,
            include_top=False,
            weights=None,
        )

    if variant_name == "mobv3_alpha02":
        return tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            alpha=0.2,
            include_top=False,
            weights=None,
            include_preprocessing=False,
        )

    raise ValueError(
        f"Unsupported MobileNet variant {variant_name!r}. "
        f"Valid options are {list(get_supported_variant_names())}."
    )
