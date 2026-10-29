"""Sparse image input helpers for MobileNet port-regression models."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import warnings
from typing import Any, Iterator

import numpy as np
import tensorflow as tf


SPARSE_IMAGE_HEIGHT = 32
SPARSE_IMAGE_WIDTH = 32
SPARSE_IMAGE_CHANNELS = 3
SPARSE_TOTAL_PORTS = 100
SPARSE_PORT_GRID_SIDE = 10


def build_sparse_port_images(
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    total_ports: int = SPARSE_TOTAL_PORTS,
    image_height: int = SPARSE_IMAGE_HEIGHT,
    image_width: int = SPARSE_IMAGE_WIDTH,
) -> np.ndarray:
    """Build deterministic sparse MobileNet images from scaled observed ports.

    Args:
        observed_values (np.ndarray): Scaled observed-port values with shape
            ``(n_samples, n_observed_ports)``. Values are copied into channel
            ``0`` at their true row-major 10x10 port locations. The caller is
            responsible for applying the train-fitted input scaler before this
            function is called.
        observed_indices (np.ndarray): One-dimensional integer array with
            length ``n_observed_ports``. Each value is a physical port index in
            ``[0, total_ports - 1]`` and must align column-for-column with
            ``observed_values``.
        total_ports (int): Total number of physical ports. Only ``100`` is
            accepted because the sparse layout is explicitly defined as a
            row-major ``10x10`` grid placed in the top-left corner of a
            ``32x32`` image.
        image_height (int): Output image height. Only ``32`` is accepted to
            match the MobileNet application minimum input size and the active
            MobileNet experiment contract.
        image_width (int): Output image width. Only ``32`` is accepted to
            match the MobileNet application minimum input size and the active
            MobileNet experiment contract.

    Returns:
        np.ndarray: ``float32`` image batch with shape
            ``(n_samples, 32, 32, 3)``. Channel ``0`` contains observed values,
            channel ``1`` contains an observed mask, and channel ``2`` contains
            normalized port coordinates for all 100 valid port cells.

    Raises:
        ValueError: If inputs have invalid rank, incompatible shapes, unsupported
            layout dimensions, duplicate/out-of-range observed indices, or if
            the sparse 10x10 grid cannot fit inside the requested image.

    Examples:
        >>> values = np.asarray([[0.25, 0.75]], dtype=np.float32)
        >>> indices = np.asarray([0, 99], dtype=np.int64)
        >>> images = build_sparse_port_images(values, indices)
        >>> images.shape
        (1, 32, 32, 3)
    """
    if total_ports != SPARSE_TOTAL_PORTS:
        raise ValueError("total_ports must be 100 for the sparse 10x10 MobileNet layout.")
    if image_height != SPARSE_IMAGE_HEIGHT or image_width != SPARSE_IMAGE_WIDTH:
        raise ValueError("image_height and image_width must both be 32.")
    if observed_values.ndim != 2:
        raise ValueError("observed_values must have shape (n_samples, n_observed_ports).")
    if observed_indices.ndim != 1:
        raise ValueError("observed_indices must be one-dimensional.")
    if observed_values.shape[1] != observed_indices.shape[0]:
        raise ValueError(
            "observed_values width must match observed_indices length: "
            f"{observed_values.shape[1]} != {observed_indices.shape[0]}."
        )
    if SPARSE_PORT_GRID_SIDE > image_height or SPARSE_PORT_GRID_SIDE > image_width:
        raise ValueError("The 10x10 sparse port grid must fit inside the requested image.")

    indices = observed_indices.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("observed_indices must contain at least one port.")
    if np.any(indices < 0) or np.any(indices >= total_ports):
        raise ValueError("observed_indices values must be in [0, total_ports - 1].")
    if np.unique(indices).size != indices.size:
        raise ValueError("observed_indices must not contain duplicate ports.")

    images = np.zeros(
        (
            observed_values.shape[0],
            image_height,
            image_width,
            SPARSE_IMAGE_CHANNELS,
        ),
        dtype=np.float32,
    )

    # Channel 2 is sample-independent physical context. It is filled for every
    # real port cell so MobileNet can see where each physical port lies even
    # when that port was not observed in the current sample.
    port_indices = np.arange(total_ports, dtype=np.float32)
    port_rows = (port_indices.astype(np.int64) // SPARSE_PORT_GRID_SIDE)
    port_cols = (port_indices.astype(np.int64) % SPARSE_PORT_GRID_SIDE)
    images[:, port_rows, port_cols, 2] = port_indices / np.float32(total_ports - 1)

    observed_rows = indices // SPARSE_PORT_GRID_SIDE
    observed_cols = indices % SPARSE_PORT_GRID_SIDE
    images[:, observed_rows, observed_cols, 0] = observed_values.astype(np.float32, copy=False)
    images[:, observed_rows, observed_cols, 1] = np.float32(1.0)

    return images


def iter_sparse_port_image_batches(
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
    total_ports: int = SPARSE_TOTAL_PORTS,
    image_height: int = SPARSE_IMAGE_HEIGHT,
    image_width: int = SPARSE_IMAGE_WIDTH,
) -> Iterator[np.ndarray]:
    """Yield sparse MobileNet image batches from compact observed-port inputs.

    Args:
        observed_values (np.ndarray): Scaled compact observed-port matrix with
            shape ``(n_samples, n_observed_ports)``. The full sparse
            ``32x32x3`` representation is built only for one yielded batch at a
            time.
        observed_indices (np.ndarray): One-dimensional integer array mapping
            each compact input column to its physical port index. Values must
            be valid for ``build_sparse_port_images``.
        batch_size (int): Maximum number of compact samples expanded in one
            batch. Larger values reduce Python overhead but increase peak
            memory for the temporary sparse image batch. Smaller values reduce
            peak memory and may reduce throughput.
        total_ports (int): Total number of physical ports. Only ``100`` is
            accepted by the sparse image layout.
        image_height (int): Sparse image height. Only ``32`` is accepted by
            the active MobileNet layout.
        image_width (int): Sparse image width. Only ``32`` is accepted by the
            active MobileNet layout.

    Yields:
        np.ndarray: Sparse image batch with shape
            ``(batch_size_or_tail, 32, 32, 3)``.

    Returns:
        Iterator[np.ndarray]: Generator that lazily yields sparse image
            batches.

    Raises:
        ValueError: If ``observed_values`` is not a non-empty 2D matrix,
            ``batch_size`` is not positive, or sparse-image validation fails.

    Examples:
        >>> values = np.asarray([[0.25], [0.75]], dtype=np.float32)
        >>> batches = list(iter_sparse_port_image_batches(values, np.asarray([0]), 1))
        >>> [batch.shape[0] for batch in batches]
        [1, 1]
    """
    observed_values = np.asarray(observed_values, dtype=np.float32)
    if observed_values.ndim != 2:
        raise ValueError("observed_values must have shape (n_samples, n_observed_ports).")
    if observed_values.shape[0] == 0:
        raise ValueError("observed_values must contain at least one sample.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Validate the index contract once before yielding so caller failures are
    # deterministic and do not depend on which batch happens to be requested.
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    build_sparse_port_images(
        observed_values=observed_values[:1],
        observed_indices=observed_indices,
        total_ports=total_ports,
        image_height=image_height,
        image_width=image_width,
    )

    for start in range(0, observed_values.shape[0], batch_size):
        batch_values = observed_values[start : start + batch_size]
        yield build_sparse_port_images(
            observed_values=batch_values,
            observed_indices=observed_indices,
            total_ports=total_ports,
            image_height=image_height,
            image_width=image_width,
        )


class SparsePortImageDataset(tf.keras.utils.PyDataset):
    """Keras dataset that expands compact observed-port rows per batch."""

    def __init__(
        self,
        observed_values: np.ndarray,
        targets: np.ndarray,
        observed_indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        total_ports: int = SPARSE_TOTAL_PORTS,
        image_height: int = SPARSE_IMAGE_HEIGHT,
        image_width: int = SPARSE_IMAGE_WIDTH,
        **kwargs: Any,
    ) -> None:
        """Store compact arrays and sparse-image batching parameters.

        Args:
            observed_values (np.ndarray): Scaled compact observed-port matrix
                with shape ``(n_samples, n_observed_ports)``. It stays compact
                in memory and is expanded only inside ``__getitem__``.
            targets (np.ndarray): Scaled target matrix with shape
                ``(n_samples, total_ports)``. It must have the same first
                dimension as ``observed_values``.
            observed_indices (np.ndarray): One-dimensional integer array
                mapping compact input columns to physical port indices.
            batch_size (int): Number of samples returned per Keras batch.
                Larger values can improve throughput but increase temporary
                sparse-image memory. Smaller values reduce memory pressure at
                the cost of more Python calls.
            shuffle (bool): If ``True``, sample order is shuffled at
                construction and after each epoch using ``seed``. This matches
                Keras training-array behavior and changes batch membership
                between epochs. If ``False``, sample order is stable, which is
                required for deterministic validation batches.
            seed (int): NumPy RNG seed used only when ``shuffle`` is ``True``.
                Reusing the same seed gives the same initial order. When
                ``shuffle`` is ``False``, the seed has no effect.
            total_ports (int): Total physical port count. Only ``100`` is
                supported by the active sparse image layout.
            image_height (int): Sparse image height. Only ``32`` is supported.
            image_width (int): Sparse image width. Only ``32`` is supported.
            **kwargs (Any): Optional Keras ``PyDataset`` runtime settings such
                as worker configuration. These values are forwarded to the base
                class and do not change the sparse image representation.

        Returns:
            None: Initializes the Keras-compatible dataset.

        Raises:
            ValueError: If arrays are empty, dimensions are incompatible,
                ``batch_size`` is not positive, or sparse-image validation
                fails.

        Examples:
            >>> values = np.asarray([[0.25], [0.75]], dtype=np.float32)
            >>> targets = np.asarray([[0.0], [1.0]], dtype=np.float32)
            >>> dataset = SparsePortImageDataset(values, targets, np.asarray([0]), 1, False, 0)
            >>> len(dataset)
            2
        """
        super().__init__(**kwargs)
        observed_values = np.asarray(observed_values, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if observed_values.ndim != 2:
            raise ValueError("observed_values must have shape (n_samples, n_observed_ports).")
        if targets.ndim != 2:
            raise ValueError("targets must have shape (n_samples, n_targets).")
        if observed_values.shape[0] == 0:
            raise ValueError("observed_values must contain at least one sample.")
        if observed_values.shape[0] != targets.shape[0]:
            raise ValueError("observed_values and targets must contain the same number of rows.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.observed_values = observed_values
        self.targets = targets
        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.total_ports = int(total_ports)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(self.observed_values.shape[0], dtype=np.int64)

        # Validate the sparse layout once up front; each batch can then reuse
        # the same index contract without hiding invalid metadata until fit().
        build_sparse_port_images(
            observed_values=self.observed_values[:1],
            observed_indices=self.observed_indices,
            total_ports=self.total_ports,
            image_height=self.image_height,
            image_width=self.image_width,
        )

        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        """Return the number of Keras batches in one epoch.

        Args:
            None.

        Returns:
            int: Ceiling division of sample count by ``batch_size``.

        Raises:
            None.
        """
        return int(np.ceil(self.observed_values.shape[0] / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one sparse-image batch and its aligned target batch.

        Args:
            index (int): Zero-based Keras batch index. Values outside the
                dataset length are invalid and raise ``IndexError``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sparse image batch with shape
            ``(batch, 32, 32, 3)`` and aligned target batch.

        Raises:
            IndexError: If ``index`` is outside ``[0, len(self) - 1]``.
            ValueError: If sparse-image construction fails for the selected
                compact batch.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index {index} is outside the dataset range.")

        start = index * self.batch_size
        batch_indices = self.order[start : start + self.batch_size]
        batch_values = self.observed_values[batch_indices]
        batch_targets = self.targets[batch_indices]

        # The sparse projection is intentionally delayed to this boundary so
        # the full train/validation split never exists as dense image tensors.
        batch_images = build_sparse_port_images(
            observed_values=batch_values,
            observed_indices=self.observed_indices,
            total_ports=self.total_ports,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        return batch_images, batch_targets

    def on_epoch_end(self) -> None:
        """Shuffle sample order after each epoch when training requires it.

        Args:
            None.

        Returns:
            None: Mutates only the internal index order when ``shuffle`` is
            ``True``. When ``shuffle`` is ``False``, validation order remains
            unchanged.

        Raises:
            None.
        """
        if self.shuffle:
            self.rng.shuffle(self.order)


def evaluate_sparse_port_tflite_model_from_bytes(
    tflite_model: bytes,
    observed_values: np.ndarray,
    y_scaled: np.ndarray,
    observed_indices: np.ndarray,
    batch_size: int,
    total_ports: int = SPARSE_TOTAL_PORTS,
    *,
    suppress_output: bool = True,
    suppress_warning: bool = True,
) -> float:
    """Evaluate a TFLite sparse-image model from compact observed-port inputs.

    Args:
        tflite_model (bytes): Serialized TFLite flatbuffer whose input tensor
            has fixed shape ``[1, 32, 32, 3]``.
        observed_values (np.ndarray): Compact scaled observed-port matrix with
            shape ``(n_samples, n_observed_ports)``. It is expanded into sparse
            image batches incrementally.
        y_scaled (np.ndarray): Scaled target matrix with shape
            ``(n_samples, total_ports)``. MSE is accumulated directly against
            this matrix without storing all predictions.
        observed_indices (np.ndarray): One-dimensional physical port indices
            aligned with ``observed_values`` columns.
        batch_size (int): Number of compact rows converted to sparse images at
            a time. TFLite invocation remains sample-by-sample because the
            active INT8 models expose batch size ``1``.
        total_ports (int): Total physical port count. Only ``100`` is accepted
            by the active sparse image layout.
        suppress_output (bool): If ``True``, suppresses native TensorFlow Lite
            stdout/stderr during interpreter allocation. If ``False``, backend
            diagnostics remain visible.
        suppress_warning (bool): If ``True``, suppresses the TensorFlow Lite
            interpreter deprecation warning. If ``False``, warnings are emitted
            normally.

    Returns:
        float: Mean squared error between dequantized TFLite predictions and
            ``y_scaled``.

    Raises:
        ValueError: If inputs are empty, shapes are incompatible, quantization
            scales are zero, or the TFLite input shape is unsupported.
        Exception: Propagates TensorFlow Lite allocation and invocation errors.

    Examples:
        >>> callable(evaluate_sparse_port_tflite_model_from_bytes)
        True
    """
    observed_values = np.asarray(observed_values, dtype=np.float32)
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    if observed_values.ndim != 2:
        raise ValueError("observed_values must have shape (n_samples, n_observed_ports).")
    if y_scaled.ndim != 2:
        raise ValueError("y_scaled must have shape (n_samples, total_ports).")
    if observed_values.shape[0] == 0:
        raise ValueError("observed_values must contain at least one sample.")
    if observed_values.shape[0] != y_scaled.shape[0]:
        raise ValueError("observed_values and y_scaled must contain the same number of rows.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from utils.qttf import make_tflite_interpreter, suppress_native_output

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
    expected_shape = np.asarray(input_detail["shape"], dtype=np.int64)

    if expected_shape.tolist() != [1, SPARSE_IMAGE_HEIGHT, SPARSE_IMAGE_WIDTH, SPARSE_IMAGE_CHANNELS]:
        raise ValueError(f"Unsupported TFLite input shape: {expected_shape.tolist()}.")
    if input_scale == 0 or output_scale == 0:
        raise ValueError("Strict INT8 TFLite model must expose non-zero input and output scales.")

    squared_error_sum = np.float64(0.0)
    value_count = 0
    target_start = 0
    for sparse_batch in iter_sparse_port_image_batches(
        observed_values=observed_values,
        observed_indices=observed_indices,
        batch_size=batch_size,
        total_ports=total_ports,
    ):
        target_batch = y_scaled[target_start : target_start + sparse_batch.shape[0]]
        target_start += sparse_batch.shape[0]

        for sample, target in zip(sparse_batch, target_batch, strict=True):
            # The interpreter input is batch-one, so each sparse image is
            # quantized and invoked independently while the sparse allocation
            # remains limited to the outer chunk.
            quantized_input = np.round(sample[np.newaxis, ...] / input_scale + input_zero_point)
            quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_detail["index"], quantized_input.reshape(input_detail["shape"]))
            interpreter.invoke()

            quantized_output = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
            output = ((quantized_output - output_zero_point) * output_scale).reshape(-1)
            squared_error_sum += np.sum(np.square(output - target), dtype=np.float64)
            value_count += int(target.size)

    return float(squared_error_sum / value_count)


class SparsePortImagePredictAdapter:
    """Adapt flat observed-port predictors to sparse MobileNet image inputs."""

    def __init__(
        self,
        model: Any,
        observed_indices: np.ndarray,
        total_ports: int = SPARSE_TOTAL_PORTS,
        image_height: int = SPARSE_IMAGE_HEIGHT,
        image_width: int = SPARSE_IMAGE_WIDTH,
    ) -> None:
        """Store sparse-image conversion settings for a wrapped predictor.

        Args:
            model (Any): Predictor exposing a Keras-compatible
                ``predict(x, batch_size=..., verbose=...)`` method. The wrapped
                model must accept sparse image tensors shaped
                ``(n_samples, 32, 32, 3)``.
            observed_indices (np.ndarray): One-dimensional integer array of
                observed physical port indices aligned with flat input columns
                that will be received by ``predict``.
            total_ports (int): Total physical port count. Only ``100`` is
                supported by ``build_sparse_port_images``.
            image_height (int): Sparse image height. Only ``32`` is supported.
            image_width (int): Sparse image width. Only ``32`` is supported.

        Returns:
            None: Initializes the adapter.

        Raises:
            ValueError: If the observed indices are invalid when a prediction
                call converts flat inputs to sparse images.
        """
        self.model = model
        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        self.total_ports = int(total_ports)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 64,
        verbose: int = 0,
    ) -> np.ndarray:
        """Predict from flat scaled observed values using sparse image inputs.

        Args:
            x (np.ndarray): Scaled observed-port matrix with shape
                ``(n_samples, n_observed_ports)``. Columns must align with the
                adapter's ``observed_indices``.
            batch_size (int): Maximum number of compact samples expanded to
                sparse images and forwarded to the wrapped model per
                prediction call. Larger values can improve throughput but
                increase temporary sparse-image memory. Smaller values reduce
                memory pressure and increase Python call overhead.
            verbose (int): Verbosity flag forwarded to the wrapped model. The
                adapter does not log on its own. The same value is forwarded to
                every chunked ``predict`` call.

        Returns:
            np.ndarray: Wrapped model predictions in the same scaled target
            space returned by the wrapped predictor.

        Raises:
            ValueError: If ``x`` cannot be converted into sparse images or
                ``batch_size`` is not positive.
            Exception: Propagates prediction failures from the wrapped model.
        """
        predictions = []
        for sparse_batch in iter_sparse_port_image_batches(
            observed_values=x,
            observed_indices=self.observed_indices,
            batch_size=batch_size,
            total_ports=self.total_ports,
            image_height=self.image_height,
            image_width=self.image_width,
        ):
            # Forward one sparse chunk at a time so OP evaluation can process
            # full splits without holding the expanded image tensor in memory.
            predictions.append(self.model.predict(sparse_batch, batch_size=batch_size, verbose=verbose))

        return np.concatenate(predictions, axis=0).astype(np.float32)
