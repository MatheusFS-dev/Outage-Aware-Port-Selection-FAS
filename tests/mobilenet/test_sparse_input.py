"""Tests for MobileNet sparse image batching helpers."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import sys
from pathlib import Path

import numpy as np
import pytest

CURRENT_FILE = _COMPANION_ROOT / 'tests/mobilenet/test_sparse_input.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from mobilenet.keras.sparse_input import (
    SparsePortImageDataset,
    SparsePortImagePredictAdapter,
    build_sparse_port_images,
)


class RecordingPredictor:
    """Fake sparse-image predictor that records chunk shapes for assertions."""

    def __init__(self) -> None:
        """Initialize an empty call-shape log.

        Args:
            None.

        Returns:
            None: Creates ``calls`` on the fake predictor.

        Raises:
            None.
        """
        self.calls: list[tuple[int, ...]] = []

    def predict(self, x: np.ndarray, batch_size: int = 64, verbose: int = 0) -> np.ndarray:
        """Return the first sparse pixel value while recording batch shape.

        Args:
            x (np.ndarray): Sparse image batch received from
                ``SparsePortImagePredictAdapter``.
            batch_size (int): Batch size forwarded by the adapter. It does not
                change this fake predictor's output, but tests assert that the
                incoming sparse batch already respects it.
            verbose (int): Verbosity flag forwarded by the adapter. ``0`` and
                non-zero values produce the same fake output.

        Returns:
            np.ndarray: One-column prediction matrix containing the channel-0
                value at port index ``0`` for each sample.

        Raises:
            None.
        """
        del batch_size, verbose
        self.calls.append(tuple(x.shape))
        return x[:, 0, 0, 0:1].astype(np.float32)


def test_sparse_port_image_dataset_matches_direct_sparse_builder() -> None:
    """Assert compact dataset batches match direct sparse image construction.

    Args:
        None.

    Returns:
        None: The test passes only when the lazy dataset output is exactly
            equal to direct ``build_sparse_port_images`` output.

    Raises:
        AssertionError: If sparse images or aligned targets differ.
    """
    values = np.asarray([[0.25, 0.75], [0.5, 0.1], [0.0, 1.0]], dtype=np.float32)
    targets = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    observed_indices = np.asarray([0, 99], dtype=np.int64)

    dataset = SparsePortImageDataset(
        observed_values=values,
        targets=targets,
        observed_indices=observed_indices,
        batch_size=2,
        shuffle=False,
        seed=0,
    )

    batch_images, batch_targets = dataset[0]
    expected_images = build_sparse_port_images(values[:2], observed_indices)
    np.testing.assert_array_equal(batch_images, expected_images)
    np.testing.assert_array_equal(batch_targets, targets[:2])


def test_sparse_port_image_dataset_preserves_order_when_shuffle_is_false() -> None:
    """Assert validation-style datasets preserve compact row order.

    Args:
        None.

    Returns:
        None: The test passes only when target batches follow input order.

    Raises:
        AssertionError: If stable-order batches are reordered.
    """
    values = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
    targets = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)

    dataset = SparsePortImageDataset(
        observed_values=values,
        targets=targets,
        observed_indices=np.asarray([0], dtype=np.int64),
        batch_size=2,
        shuffle=False,
        seed=123,
    )

    _, first_targets = dataset[0]
    _, second_targets = dataset[1]
    np.testing.assert_array_equal(first_targets, targets[:2])
    np.testing.assert_array_equal(second_targets, targets[2:])


def test_sparse_port_image_dataset_rejects_invalid_shapes_and_batch_size() -> None:
    """Assert invalid compact inputs fail during dataset construction.

    Args:
        None.

    Returns:
        None: The test passes only when invalid row counts and batch sizes
            raise explicit ``ValueError`` exceptions.

    Raises:
        AssertionError: If an invalid dataset configuration is accepted.
    """
    values = np.asarray([[0.1], [0.2]], dtype=np.float32)
    targets = np.asarray([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="same number of rows"):
        SparsePortImageDataset(
            observed_values=values,
            targets=targets,
            observed_indices=np.asarray([0], dtype=np.int64),
            batch_size=1,
            shuffle=False,
            seed=0,
        )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        SparsePortImageDataset(
            observed_values=values,
            targets=np.asarray([[1.0], [2.0]], dtype=np.float32),
            observed_indices=np.asarray([0], dtype=np.int64),
            batch_size=0,
            shuffle=False,
            seed=0,
        )


def test_sparse_port_image_predict_adapter_chunks_without_reordering() -> None:
    """Assert chunked prediction caps sparse batch size and preserves order.

    Args:
        None.

    Returns:
        None: The test passes only when sparse prediction chunks respect
            ``batch_size`` and concatenated predictions match input order.

    Raises:
        AssertionError: If chunk shapes or prediction order differ.
    """
    values = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
    predictor = RecordingPredictor()
    adapter = SparsePortImagePredictAdapter(
        model=predictor,
        observed_indices=np.asarray([0], dtype=np.int64),
    )

    predictions = adapter.predict(values, batch_size=2, verbose=0)

    assert predictor.calls == [(2, 32, 32, 3), (2, 32, 32, 3), (1, 32, 32, 3)]
    np.testing.assert_array_equal(predictions, values)
