"""Shared training helpers for MobileNet Keras runs."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
from pathlib import Path
from typing import Any

from mobilenet.keras.sparse_input import SparsePortImageDataset
from studies.chapter06_mobilenet_nas.scripts.train.multi_seed_runner import release_training_case_resources
from mobilenet.keras.variants import build_backbone_model, get_variant_metadata
from utils.constants import BLUE, GREEN, RED, RESET
from utils.data import (
    KMS_REGIME_LABELS,
    get_observed_ports,
    split_kms_60_20_20_cache,
    split_kms_ratio_cache,
)
from utils.imports_keras import *  # noqa: F403


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
HEAD_DENSE_UNITS = (64, 32)
LEARNING_RATE = 1e-3


def configure_training_runtime(
    train_seed: int,
    precision_policy: str = "float32",
    enable_op_determinism: bool = True,
) -> mixed_precision.Policy:
    """Configure deterministic Keras runtime settings for one training run.

    Args:
        train_seed (int): Seed shared across Python, NumPy, Keras, and
            TensorFlow random generators. Reusing the same value improves
            reproducibility when the hardware, input order, and TensorFlow build
            remain unchanged. Changing it changes initialization and minibatch
            shuffling, which may produce different training curves and final
            metrics.
        precision_policy (str): TensorFlow Keras mixed-precision policy name.
            If ``"float32"``, all model compute and variables use standard
            single-precision behavior, matching the existing MobileNet runs. If
            ``"mixed_float16"`` or another TensorFlow-supported policy is used,
            compatible layers may compute with reduced precision, which can
            reduce GPU memory use and improve throughput on supported hardware
            but may change numerics and hardware compatibility.
        enable_op_determinism (bool): Controls TensorFlow deterministic
            operation mode. If ``True``, TensorFlow is asked to use
            deterministic kernels where available, preserving the existing
            reproducibility contract at possible throughput cost. If ``False``,
            TensorFlow may use faster nondeterministic kernels, so repeated
            runs with the same seeds may differ slightly.

    Returns:
        mixed_precision.Policy: Active mixed-precision policy used for the run.
            The returned policy matches ``precision_policy``.

    Raises:
        RuntimeError: If TensorFlow cannot enable deterministic execution.
        ValueError: If ``precision_policy`` is not accepted by TensorFlow.

    Examples:
        >>> policy = configure_training_runtime(0)
        >>> policy.name
        'float32'
    """
    set_random_seed(train_seed)

    # Deterministic kernels preserve the existing experiment contract, while
    # allowing callers to disable them gives a narrow throughput tuning switch.
    if enable_op_determinism:
        tf.config.experimental.enable_op_determinism()

    policy = mixed_precision.Policy(precision_policy)
    mixed_precision.set_global_policy(policy)
    return policy


def load_training_data(
    num_observed_ports: int,
    total_ports: int,
    data_seed: int,
    train_regime_labels: list[str] | tuple[str, ...] | None,
    samples_per_regime: int | None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, np.ndarray]:
    """Load the full 60/20/20 KMS split and scale train and validation tensors.

    Args:
        num_observed_ports (int): Number of equally spaced ports exposed to the
            model as input features. Larger values preserve more channel
            information but increase input dimensionality. Smaller values reduce
            sensed input information and may reduce accuracy.
        total_ports (int): Number of regression targets per sample. This module
            expects ``100`` so the MobileNet head predicts the full port vector.
        data_seed (int): Seed forwarded to ``split_kms_60_20_20_cache``.
            Reusing the same value reconstructs the same per-regime train and
            validation membership. Changing it changes the 60/20/20 assignment
            across all selected regimes.
        train_regime_labels (list[str] | tuple[str, ...] | None): Regime
            selector for the cache-backed split.
            If ``None``, all regimes in ``KMS_REGIME_LABELS`` are loaded.
            If a list or tuple of labels is provided, only those regimes are
            loaded, in canonical order, and both train and validation remain
            limited to that subset.
        samples_per_regime (int | None): Optional per-regime sample budget
            forwarded to ``split_kms_60_20_20_cache``.
            If ``None``, the full available selected-regime data is used,
            preserving current training behavior and memory usage.
            If an integer is provided, exactly that many rows are sampled from
            each selected regime before splitting, reducing runtime while
            keeping multi-regime training balanced by regime.
        verbose (bool): Controls split helper logging.
            If ``True``, the underlying data helper prints dataset-part shapes.
            If ``False``, loading is silent apart from this module's own logs.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, np.ndarray]:
            ``(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
            scaler_x, scaler_y, observed_indices)`` where input arrays remain
            compact observed-port matrices, target arrays use ``float32``, and
            both scalers are fitted on the training partition only.

    Raises:
        FileNotFoundError: If the runtime cache is missing.
        ValueError: If the selected regimes cannot produce a valid split.

    Examples:
        >>> outputs = load_training_data(4, 100, 0, None, None, verbose=False)
        >>> len(outputs)
        7
    """
    print(f"{BLUE}Loading and preprocessing full cache-backed 60/20/20 data...{RESET}")
    y_train, y_val, _, _, _, _ = split_kms_60_20_20_cache(
        common_path=None,
        seed=data_seed,
        regime_labels=train_regime_labels,
        samples_per_regime=samples_per_regime,
        verbose=verbose,
    )

    X_train, observed_indices = get_observed_ports(y_train, num_observed_ports, total_ports)
    X_val, _ = get_observed_ports(y_val, num_observed_ports, total_ports)

    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    X_train_scaled = scaler_x.transform(X_train).astype(np.float32)
    X_val_scaled = scaler_x.transform(X_val).astype(np.float32)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val).astype(np.float32)

    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_x, scaler_y, observed_indices


def load_training_data_with_ratio(
    num_observed_ports: int,
    total_ports: int,
    data_seed: int,
    train_regime_labels: list[str] | tuple[str, ...] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, np.ndarray]:
    """Load a cache-backed KMS ratio split and scale train and validation tensors.

    Args:
        num_observed_ports (int): Number of equally spaced ports exposed to the
            model as input features. Larger values preserve more channel
            information but increase input dimensionality.
        total_ports (int): Number of regression targets per sample. This module
            expects ``100`` so the MobileNet head predicts the full port vector.
        data_seed (int): Seed forwarded to ``split_kms_ratio_cache``. Reusing
            the same value reconstructs the same per-regime train, validation,
            and test membership for the configured ratios.
        train_regime_labels (list[str] | tuple[str, ...] | None): Regime
            selector for the cache-backed split. If ``None``, all regimes are
            loaded. If labels are provided, only those regimes are loaded in
            canonical order.
        train_ratio (float): Fraction of each selected regime assigned to
            training. The ratio must be in ``[0.0, 1.0]`` and the sum of
            train, validation, and test ratios must not exceed ``1.0``.
        val_ratio (float): Fraction of each selected regime assigned to
            validation. The value is fixed by the analysis scripts at ``0.2``.
        test_ratio (float): Fraction of each selected regime assigned to test.
            The value is fixed by the analysis scripts at ``0.2``. Test data is
            reconstructed later by OP evaluation, not used during training.
        verbose (bool): Controls split helper logging.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, np.ndarray]:
            ``(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
            scaler_x, scaler_y, observed_indices)`` where input arrays remain
            compact observed-port matrices, target arrays use ``float32``, and
            both scalers are fitted on the training partition only.

    Raises:
        FileNotFoundError: If the runtime cache is missing.
        ValueError: If the selected regimes cannot produce a valid ratio split.

    Examples:
        >>> callable(load_training_data_with_ratio)
        True
    """
    print(
        f"{BLUE}Loading and preprocessing cache-backed ratio data "
        f"(train={train_ratio}, val={val_ratio}, test={test_ratio})...{RESET}"
    )
    y_train, y_val, _, _, _, _ = split_kms_ratio_cache(
        common_path=None,
        seed=data_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        regime_labels=train_regime_labels,
        samples_per_regime=None,
        verbose=verbose,
    )

    X_train, observed_indices = get_observed_ports(y_train, num_observed_ports, total_ports)
    X_val, _ = get_observed_ports(y_val, num_observed_ports, total_ports)

    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    X_train_scaled = scaler_x.transform(X_train).astype(np.float32)
    X_val_scaled = scaler_x.transform(X_val).astype(np.float32)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val).astype(np.float32)

    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_x, scaler_y, observed_indices


def build_mobilenet_regressor(
    variant_name: str,
    num_observed_ports: int,
    total_ports: int,
    steps_per_execution: int,
    jit_compile: bool | str = False,
    show_summary: bool = True,
) -> Model:
    """Build and compile one MobileNet-based regressor for port prediction.

    Args:
        variant_name (str): Supported MobileNet variant selector.
            If ``"mobv1_alpha02"``, the feature extractor is MobileNetV1 with
            ``alpha=0.2``.
            If ``"mobv2_alpha02"``, the feature extractor is MobileNetV2 with
            ``alpha=0.2``.
            If ``"mobv3_alpha02"``, the feature extractor is MobileNetV3Small
            with ``alpha=0.2``.
            Any other value is invalid and raises an explicit error.
        num_observed_ports (int): Number of observed ports represented in the
            sparse image input. This value is validated for experiment metadata
            consistency, while the model input tensor itself is always
            ``(32, 32, 3)``.
        total_ports (int): Number of regression outputs. This project expects
            ``100`` because each sample predicts the full port vector.
        steps_per_execution (int): Keras execution batching control passed to
            ``model.compile``.
            If set to ``1``, training and validation execute one step at a
            time, which matches the current example and is safest for smaller
            validation loops.
            Larger values can improve throughput but may break validation metric
            reporting when a loop has too few batches.
        jit_compile (bool | str): XLA compilation mode passed to
            ``model.compile``. If ``False``, Keras does not request XLA
            compilation, matching the existing MobileNet behavior. If ``True``,
            Keras requests XLA compilation, which may improve throughput after
            compilation warmup but can fail for unsupported TensorFlow ops. If a
            supported string mode such as ``"auto"`` is used, Keras decides
            whether to compile based on framework support.
        show_summary (bool): Controls whether ``model.summary()`` is printed.
            If ``True``, the full assembled network is printed.
            If ``False``, model construction stays silent.

    Returns:
        Model: Compiled Keras regression model.

    Raises:
        ValueError: If ``variant_name`` is unsupported.

    Examples:
        >>> model = build_mobilenet_regressor(
        ...     "mobv2_alpha02",
        ...     4,
        ...     100,
        ...     steps_per_execution=1,
        ...     show_summary=False,
        ... )
        >>> model.output_shape
        (None, 100)
    """
    get_variant_metadata(variant_name)

    if num_observed_ports <= 0:
        raise ValueError("num_observed_ports must be positive.")

    inputs = layers.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        name="sparse_port_image",
    )

    backbone_model = build_backbone_model(
        variant_name=variant_name,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    )
    extracted_features = backbone_model(inputs)
    pooled_features = layers.GlobalAveragePooling2D(name="backbone_gap")(extracted_features)

    regression_features = pooled_features
    for layer_index, units in enumerate(HEAD_DENSE_UNITS):
        regression_features = layers.Dense(
            units,
            activation="relu",
            name=f"regressor_dense_{layer_index}",
        )(regression_features)

    outputs = layers.Dense(total_ports, activation="linear", name="output_ports")(regression_features)
    model = Model(inputs=inputs, outputs=outputs, name=f"{variant_name}_regressor")

    if show_summary:
        model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mse"],
        jit_compile=jit_compile,
        steps_per_execution=steps_per_execution,
    )

    return model


def resolve_training_steps_per_execution(
    name: list[str],
    data_size: list[int],
    batch_size: int,
    steps_per_execution: int | str,
) -> int:
    """Resolve integer or automatic Keras ``steps_per_execution`` settings.

    Args:
        name (list[str]): Dataset labels forwarded to
            ``validate_steps_per_execution``. The list must align with
            ``data_size`` so Araras reports the train and validation loops with
            meaningful names.
        data_size (list[int]): Number of samples in each loop named by
            ``name``. Values are row counts, not batch counts. The automatic
            mode ranks candidate values from these sizes and ``batch_size``.
        batch_size (int): Keras batch size used for all listed loops. Larger
            values reduce batch count and can change the top automatic
            recommendation; smaller values increase batch count and Python
            overhead.
        steps_per_execution (int | str): Keras execution batching control.
            If an integer, that value is validated and returned unchanged. If
            ``"auto"``, Araras computes the top recommendation for the current
            train and validation sizes, and that integer recommendation is
            returned. Any other string or a boolean value is invalid and raises
            explicitly instead of being coerced.

    Returns:
        int: Positive validated ``steps_per_execution`` value to pass to
            ``model.compile``.

    Raises:
        ValueError: If ``steps_per_execution`` is unsupported, Araras returns
            no automatic recommendation, or validation rejects the configured
            value.
        TypeError: If Araras rejects the provided dataset labels, sizes, or
            batch size.

    Examples:
        >>> resolve_training_steps_per_execution(["x"], [64], 32, 1)
        1
    """
    if steps_per_execution == "auto":
        # Automatic mode mirrors the NAS scripts: use Araras to choose the
        # highest-ranked value for the actual compact train/validation sizes.
        validation_result = validate_steps_per_execution(
            name=name,
            data_size=data_size,
            batch_size=batch_size,
            steps_per_execution=1,
            top_k_recommendations=1,
            execute_once=False,
        )
        recommendations = validation_result["recommendations"]
        if not recommendations:
            raise ValueError("Araras returned no steps_per_execution recommendation.")
        return int(recommendations[0]["steps_per_execution"])

    if isinstance(steps_per_execution, bool) or not isinstance(steps_per_execution, int):
        raise ValueError("steps_per_execution must be a positive integer or 'auto'.")

    validate_steps_per_execution(
        name=name,
        data_size=data_size,
        batch_size=batch_size,
        steps_per_execution=steps_per_execution,
    )
    return int(steps_per_execution)


def save_mobilenet_config(
    args_dir: Path,
    variant_name: str,
    num_observed_ports: int,
    total_ports: int,
    train_regime_labels: list[str] | tuple[str, ...] | None,
    samples_per_regime: int | None,
    data_seed: int,
    train_seed: int,
    observed_indices: list[int] | tuple[int, ...] | np.ndarray,
    split_mode: str = "60_20_20",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Path:
    """Save strict MobileNet run metadata for later OP auto-detection.

    Args:
        args_dir (Path): ``args`` directory created by ``init_study_dirs`` for
            the current training run.
        variant_name (str): Supported MobileNet variant selector stored exactly
            as the canonical variant name used by the training entrypoint.
        num_observed_ports (int): Number of observed ports exposed at runtime
            and stored so OP evaluation can validate model and scaler width.
        total_ports (int): Number of output ports predicted by the regressor.
        train_regime_labels (list[str] | tuple[str, ...] | None): Regime subset
            used for training.
            If ``None``, the JSON stores ``null`` to record that all regimes
            were used.
            If a list or tuple is provided, those labels are written exactly as
            the selected subset for auditability.
        samples_per_regime (int | None): Per-regime sample budget used for
            training.
            If ``None``, the JSON stores ``null`` to record full selected-regime
            training data.
            If an integer is provided, that exact budget is written so
            evaluation and later result summaries can reconstruct the run.
        data_seed (int): Seed used to select reduced samples and assign rows to
            train, validation, and test splits. Different values create
            different split memberships while preserving split counts.
        train_seed (int): Seed used for model initialization and training-time
            randomness. Reusing the same value should reproduce the same
            initialization path under the same runtime settings, while changing
            it is the mechanism used for the seed-variability study.
        observed_indices (list[int] | tuple[int, ...] | np.ndarray): Physical
            port indices used to populate the sparse image value and mask
            channels. These indices are deterministic for a given
            ``num_observed_ports`` and are stored so evaluation can rebuild the
            exact sparse inputs.
        split_mode (str): Split strategy stored for OP evaluation. Accepted
            values are ``"60_20_20"`` for standard runs and ``"ratio"`` for
            analysis runs that use configurable ratios. This changes how OP
            evaluation reconstructs split parts.
        train_ratio (float): Fraction assigned to training. Standard runs store
            ``0.6``. Ratio-analysis runs store their configured train
            percentage.
        val_ratio (float): Fraction assigned to validation. Analysis scripts
            keep this at ``0.2``.
        test_ratio (float): Fraction assigned to test. Analysis scripts keep
            this at ``0.2``. If the three ratios sum to less than ``1.0``, the
            unused remainder is recorded in ``unused_ratio``.

    Returns:
        Path: Path to the written ``mobilenet_config.json`` file.

    Raises:
        OSError: If the metadata file cannot be written.
        ValueError: If ``variant_name`` is unsupported.

    Examples:
        >>> import tempfile
        >>> args_dir = Path(tempfile.mkdtemp())
        >>> path = save_mobilenet_config(args_dir, "mobv1_alpha02", 4, 100, None, None, 0, [0, 33, 66, 99])
        >>> path.name
        'mobilenet_config.json'
    """
    if split_mode not in {"60_20_20", "ratio"}:
        raise ValueError("split_mode must be '60_20_20' or 'ratio'.")
    unused_ratio = 1.0 - (train_ratio + val_ratio + test_ratio)
    if unused_ratio < 0.0 and not np.isclose(unused_ratio, 0.0, atol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must be less than or equal to 1.0.")
    if np.isclose(unused_ratio, 0.0, atol=1e-8):
        unused_ratio = 0.0

    variant_metadata = get_variant_metadata(variant_name)
    config_path = args_dir / "mobilenet_config.json"
    config_payload = {
        "variant_name": variant_name,
        "backbone_family": variant_metadata["backbone_family"],
        "alpha": variant_metadata["alpha"],
        "input_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
        "num_observed_ports": num_observed_ports,
        "observed_indices": [int(index) for index in observed_indices],
        "total_ports": total_ports,
        "train_regime_labels": list(train_regime_labels) if train_regime_labels is not None else None,
        "samples_per_regime": samples_per_regime,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "split_mode": split_mode,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "unused_ratio": float(unused_ratio),
    }

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, indent=2, sort_keys=True)

    return config_path


def save_training_history(history: callbacks.History, history_dir: Path) -> Path:
    """Persist the standard training-history CSV for one MobileNet run.

    Args:
        history (callbacks.History): Keras history object returned by
            ``model.fit``. This helper expects ``loss`` and ``val_loss`` to be
            present because the training configuration always includes a
            validation split.
        history_dir (Path): Output directory for history artifacts.

    Returns:
        Path: Path to the written ``history.csv`` file.

    Raises:
        KeyError: If ``loss`` or ``val_loss`` metrics are missing.
        OSError: If the CSV cannot be written.
    """
    history_path = history_dir / "history.csv"
    history_data = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }
    history_df = pd.DataFrame(history_data)
    history_df.to_csv(history_path, index=False)
    return history_path


def run_training_case(
    variant_name: str,
    run_dir: str | Path,
    epochs: int,
    batch_size: int,
    data_seed: int,
    train_seed: int,
    num_observed_ports: int,
    total_ports: int,
    train_regime_labels: list[str] | tuple[str, ...] | None,
    samples_per_regime: int | None = None,
    steps_per_execution: int | str = 1,
    show_summary: bool = True,
    split_mode: str = "60_20_20",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    jit_compile: bool | str = False,
    precision_policy: str = "float32",
    enable_op_determinism: bool = True,
) -> dict[str, Any]:
    """Execute one complete MobileNet Keras training run and save artifacts.

    Args:
        variant_name (str): Supported MobileNet variant selector.
            If ``"mobv1_alpha02"``, trains a MobileNetV1-based regressor.
            If ``"mobv2_alpha02"``, trains a MobileNetV2-based regressor.
            If ``"mobv3_alpha02"``, trains a MobileNetV3Small-based regressor
            with ``alpha=0.2``.
            Any other value is invalid and causes an explicit failure.
        run_dir (str | Path): Top-level output directory for the run. The
            standard ``model_training`` subdirectories are created inside it.
        epochs (int): Number of training epochs passed to ``model.fit``.
            Larger values allow more optimization steps but increase total
            runtime. Smaller values reduce runtime and may underfit.
        batch_size (int): Batch size used for training, validation, and model
            stats inference checks. Larger values increase memory usage but may
            improve throughput. Smaller values reduce memory pressure at the
            cost of slower training.
        data_seed (int): Seed forwarded to the 60/20/20 split helper.
        train_seed (int): Random seed used for model initialization and runtime
            determinism setup.
        num_observed_ports (int): Number of observed ports used as model input.
        total_ports (int): Number of regression targets predicted per sample.
        train_regime_labels (list[str] | tuple[str, ...] | None): Regime subset
            used during training.
            If ``None``, all KMS regimes are used.
            If a list or tuple is provided, only those regimes are loaded.
        samples_per_regime (int | None): Optional per-regime sample budget used
            before the 60/20/20 split.
            If ``None``, training uses the full selected-regime dataset.
            If an integer is provided, each selected regime contributes exactly
            that many samples before splitting, which keeps multi-regime runs
            balanced and makes learning-curve runs comparable.
        steps_per_execution (int | str): Keras ``steps_per_execution`` value.
            If an integer, that value is validated against the train and
            validation loop sizes and then passed to ``model.compile``. If
            ``"auto"``, Araras selects the top recommendation for the current
            train and validation sizes before model compilation. Automatic
            mode can reduce Python overhead without hard-coding loop-specific
            constants, while invalid mode strings fail explicitly.
        show_summary (bool): Whether to print the compiled model summary.
        split_mode (str): Split strategy. If ``"60_20_20"``, use the standard
            60/20/20 helper and honor ``samples_per_regime``. If ``"ratio"``,
            use ``split_kms_ratio_cache`` with ``train_ratio``, ``val_ratio``,
            and ``test_ratio`` and ignore sample budgets.
        train_ratio (float): Training fraction used only when
            ``split_mode="ratio"``. Standard runs store the default ``0.6``.
        val_ratio (float): Validation fraction used only when
            ``split_mode="ratio"``. Analysis scripts keep this at ``0.2``.
        test_ratio (float): Test fraction used only when ``split_mode="ratio"``.
            Analysis scripts keep this at ``0.2``.
        jit_compile (bool | str): XLA compilation mode forwarded to
            ``model.compile``. If ``False``, training uses the current
            non-compiled behavior. If ``True`` or a supported string mode is
            provided, Keras may compile the train step, which can change startup
            time, supported-op requirements, and throughput.
        precision_policy (str): TensorFlow Keras mixed-precision policy name
            forwarded to ``configure_training_runtime``. ``"float32"`` keeps
            current numerical behavior. Other TensorFlow-supported policies can
            change compute dtype, memory use, hardware requirements, and loss
            numerics.
        enable_op_determinism (bool): Runtime determinism switch forwarded to
            ``configure_training_runtime``. If ``True``, deterministic kernels
            are requested as before. If ``False``, TensorFlow may use faster
            nondeterministic kernels and repeated seeded runs may differ.

    Returns:
        dict[str, Any]: Summary of the saved run artifacts, including the
            training directory, metadata path, and model path.

    Raises:
        Exception: Re-raises any dataset loading, model construction, training,
            or artifact-writing failure after logging it to disk.

    Examples:
        >>> run_dir = Path("runs") / "train_mobv1_alpha02_60_20_20_4_ports"
        >>> isinstance(run_dir, Path)
        True
    """
    run_dir_path = Path(run_dir)
    get_variant_metadata(variant_name)
    if split_mode not in {"60_20_20", "ratio"}:
        raise ValueError("split_mode must be '60_20_20' or 'ratio'.")

    print(f"{BLUE}Starting MobileNet Keras training for {variant_name}...{RESET}")
    if train_regime_labels is None:
        print(f"{BLUE}Training on all KMS regimes ({len(KMS_REGIME_LABELS)} total).{RESET}")
    else:
        print(f"{BLUE}Training on selected KMS regimes ({len(train_regime_labels)} total).{RESET}")
        print(f"{BLUE}Selected regimes: {', '.join(train_regime_labels)}{RESET}")
    if split_mode == "ratio":
        print(
            f"{BLUE}Using ratio split train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}.{RESET}"
        )
    elif samples_per_regime is None:
        print(f"{BLUE}Using full selected-regime sample count.{RESET}")
    else:
        print(f"{BLUE}Using samples_per_regime={samples_per_regime}.{RESET}")

    policy = configure_training_runtime(
        train_seed=train_seed,
        precision_policy=precision_policy,
        enable_op_determinism=enable_op_determinism,
    )
    bytes_per_param = tf.dtypes.as_dtype(policy.variable_dtype).size

    if split_mode == "ratio":
        (
            X_train_scaled,
            X_val_scaled,
            y_train_scaled,
            y_val_scaled,
            scaler_x,
            scaler_y,
            observed_indices,
        ) = load_training_data_with_ratio(
            num_observed_ports=num_observed_ports,
            total_ports=total_ports,
            data_seed=data_seed,
            train_regime_labels=train_regime_labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            verbose=True,
        )
        config_samples_per_regime = None
    else:
        (
            X_train_scaled,
            X_val_scaled,
            y_train_scaled,
            y_val_scaled,
            scaler_x,
            scaler_y,
            observed_indices,
        ) = load_training_data(
            num_observed_ports=num_observed_ports,
            total_ports=total_ports,
            data_seed=data_seed,
            train_regime_labels=train_regime_labels,
            samples_per_regime=samples_per_regime,
            verbose=True,
        )
        config_samples_per_regime = samples_per_regime

    backup_dir: Path | None = None
    logs_dir: Path | None = None
    model: Model | None = None
    history: callbacks.History | None = None
    train_dataset: SparsePortImageDataset | None = None
    validation_dataset: SparsePortImageDataset | None = None

    try:
        study_paths = init_study_dirs(str(run_dir_path), study_name="model_training")
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
        del study_dir, fig_dir, tensorboard_dir

        config_path = save_mobilenet_config(
            args_dir=args_dir,
            variant_name=variant_name,
            num_observed_ports=num_observed_ports,
            total_ports=total_ports,
            train_regime_labels=train_regime_labels,
            samples_per_regime=config_samples_per_regime,
            data_seed=data_seed,
            train_seed=train_seed,
            observed_indices=observed_indices,
            split_mode=split_mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        resolved_steps_per_execution = resolve_training_steps_per_execution(
            name=["x", "validation_data"],
            data_size=[X_train_scaled.shape[0], X_val_scaled.shape[0]],
            batch_size=batch_size,
            steps_per_execution=steps_per_execution,
        )

        model = build_mobilenet_regressor(
            variant_name=variant_name,
            num_observed_ports=num_observed_ports,
            total_ports=total_ports,
            steps_per_execution=resolved_steps_per_execution,
            jit_compile=jit_compile,
            show_summary=show_summary,
        )

        train_dataset = SparsePortImageDataset(
            observed_values=X_train_scaled,
            targets=y_train_scaled,
            observed_indices=observed_indices.astype(int),
            batch_size=batch_size,
            shuffle=True,
            seed=train_seed,
            total_ports=total_ports,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        )
        validation_dataset = SparsePortImageDataset(
            observed_values=X_val_scaled,
            targets=y_val_scaled,
            observed_indices=observed_indices.astype(int),
            batch_size=batch_size,
            shuffle=False,
            seed=train_seed,
            total_ports=total_ports,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        )

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=get_callbacks_model(
                backup_dir=str(backup_dir / "training"),
                checkpoint_dir=str(backup_dir / "checkpoints"),
                early_stopping_patience=5,
                reduce_lr_patience=3,
            ),
            verbose=2,
        )

        model_path = model_dir / "model.keras"
        model.save(str(model_path))
        print(f"{GREEN}Saved MobileNet model artifact to {model_path}.{RESET}")

        scaler_x_path = scaler_dir / "scaler_X.pkl"
        scaler_y_path = scaler_dir / "scaler_y.pkl"
        with open(scaler_x_path, "wb") as scaler_file:
            pickle.dump(scaler_x, scaler_file)
        with open(scaler_y_path, "wb") as scaler_file:
            pickle.dump(scaler_y, scaler_file)
        print(f"{GREEN}Saved MobileNet scalers.{RESET}")

        history_path = save_training_history(history=history, history_dir=history_dir)
        print(f"{GREEN}Saved MobileNet training history to {history_path}.{RESET}")

        stats_path = args_dir / "model_stats.txt"
        write_model_stats_to_file(
            model=model,
            file_path=str(stats_path),
            batch_size=batch_size,
            bytes_per_param=bytes_per_param,
            device="gpu/0",
            stats_to_measure=(
                "parameters",
                "model_size",
                "flops",
                "macs",
                "summary",
                "inference_latency",
            ),
            extra_attrs={
                "variant_name": variant_name,
                "final_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
            },
            test_runs=10,
            verbose=1,
        )
        print(f"{GREEN}Saved MobileNet model stats to {stats_path}.{RESET}")
        print(f"{GREEN}MobileNet training finished successfully for {variant_name}.{RESET}")

        return {
            "run_dir": run_dir_path,
            "config_path": config_path,
            "model_path": model_path,
            "scaler_x_path": scaler_x_path,
            "scaler_y_path": scaler_y_path,
            "history_path": history_path,
            "stats_path": stats_path,
        }
    except Exception as error:
        print(f"\n{RED}An error occurred during {variant_name} training: {error}{RESET}\n")
        traceback.print_exc()

        if logs_dir is not None:
            error_log_path = logs_dir / "training_error.log"
            with open(error_log_path, "a", encoding="utf-8") as error_log_file:
                error_log_file.write(
                    "An error occurred during MobileNet training:\n"
                    f"{error}\n{traceback.format_exc()}\n\n"
                )

        raise
    finally:
        # These references hold the largest Python-side arrays and TensorFlow
        # objects for each case. Dropping them before the session reset avoids
        # retaining them across the long per-script case loops.
        del model
        del history
        del train_dataset
        del validation_dataset
        del X_train_scaled
        del X_val_scaled
        del y_train_scaled
        del y_val_scaled
        del scaler_x
        del scaler_y
        del observed_indices

        release_training_case_resources(tf.keras.backend.clear_session)

        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        if logs_dir is not None and logs_dir.exists() and not any(logs_dir.iterdir()):
            logs_dir.rmdir()


def run_training_ratio_case(
    variant_name: str,
    run_dir: str | Path,
    epochs: int,
    batch_size: int,
    data_seed: int,
    train_seed: int,
    num_observed_ports: int,
    total_ports: int,
    train_regime_labels: list[str] | tuple[str, ...] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    steps_per_execution: int | str = 1,
    show_summary: bool = True,
    jit_compile: bool | str = False,
    precision_policy: str = "float32",
    enable_op_determinism: bool = True,
) -> dict[str, Any]:
    """Execute one MobileNet analysis run with configurable split ratios.

    Args:
        variant_name (str): Supported MobileNet variant selector.
        run_dir (str | Path): Top-level output directory for the run.
        epochs (int): Number of training epochs passed to ``model.fit``.
        batch_size (int): Batch size used for training, validation, and model
            stats inference checks.
        data_seed (int): Seed forwarded to the ratio split helper.
        train_seed (int): Random seed used for model initialization and runtime
            determinism setup.
        num_observed_ports (int): Number of observed ports used as model input.
        total_ports (int): Number of regression targets predicted per sample.
        train_regime_labels (list[str] | tuple[str, ...] | None): Regime subset
            used during training. If ``None``, all KMS regimes are used.
        train_ratio (float): Fraction of each selected regime assigned to
            training. The data-size analysis iterates this value.
        val_ratio (float): Fraction of each selected regime assigned to
            validation. The analysis scripts keep this fixed at ``0.2``.
        test_ratio (float): Fraction of each selected regime assigned to test.
            The analysis scripts keep this fixed at ``0.2``.
        steps_per_execution (int | str): Keras ``steps_per_execution`` value.
            If an integer, that value is validated by ``run_training_case``. If
            ``"auto"``, ``run_training_case`` resolves the top Araras
            recommendation after loading the split and before model
            compilation.
        show_summary (bool): Whether to print the compiled model summary.
        jit_compile (bool | str): XLA compilation mode forwarded to
            ``run_training_case`` and then ``model.compile``. ``False`` keeps
            current behavior, while ``True`` or a supported string mode may
            change compile overhead, supported-op requirements, and throughput.
        precision_policy (str): TensorFlow Keras mixed-precision policy name.
            ``"float32"`` preserves current behavior; other TensorFlow-supported
            policies may change compute dtype, memory use, and numerical
            behavior.
        enable_op_determinism (bool): Controls deterministic TensorFlow
            operation mode. ``True`` preserves the existing deterministic
            request, while ``False`` may improve speed at the cost of exact
            repeatability.

    Returns:
        dict[str, Any]: Summary of the saved run artifacts, including the
            training directory, metadata path, and model path.

    Raises:
        Exception: Re-raises any dataset loading, model construction, training,
            or artifact-writing failure from ``run_training_case``.

    Examples:
        >>> callable(run_training_ratio_case)
        True
    """
    return run_training_case(
        variant_name=variant_name,
        run_dir=run_dir,
        epochs=epochs,
        batch_size=batch_size,
        data_seed=data_seed,
        train_seed=train_seed,
        num_observed_ports=num_observed_ports,
        total_ports=total_ports,
        train_regime_labels=train_regime_labels,
        samples_per_regime=None,
        steps_per_execution=steps_per_execution,
        show_summary=show_summary,
        split_mode="ratio",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        jit_compile=jit_compile,
        precision_policy=precision_policy,
        enable_op_determinism=enable_op_determinism,
    )
