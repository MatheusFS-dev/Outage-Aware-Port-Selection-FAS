
# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))
import sys
from typing import NamedTuple

import scipy
import numpy as np
from pathlib import Path
from utils.constants import BLUE, GREEN, RESET

# Ensure the repository src/ directory is on sys.path when running this file
# directly, so local package imports resolve consistently across entrypoints.
CURRENT_FILE = _COMPANION_ROOT / 'src/utils/data.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

DEFAULT_DATASET_ROOT_PATH = str(_COMPANION_ROOT / 'data/kappa-mu_shadowed')
DEFAULT_KMS_RUNTIME_CACHE_PATH = str(_COMPANION_ROOT / 'data/kms_runtime_cache_float32')

# Configuration for kappa-mu-shadowed channel regimes: (kappa_param, mu_param, m_param)
# Describes all 14 parameter combinations used in the training protocol.
_KAPPA_MU_SHADOWED_REGIMES = [
    (1.0e-16, 1.0, 0.0),  # Rayleigh-like (kappa→0, mu=1, m→∞)
    (1.0e-16, 1.0, 2.0),
    (1.0e-16, 1.0, 50.0),
    (1.0e-16, 2.0, 50.0),
    (1.0e-16, 5.0, 50.0),
    (5.0e00, 1.0, 0.0),  # Rician-like (kappa=K, mu=1)
    (5.0e00, 1.0, 2.0),
    (5.0e00, 1.0, 50.0),
    (5.0e00, 2.0, 0.0),
    (5.0e00, 2.0, 2.0),
    (5.0e00, 2.0, 50.0),
    (5.0e00, 5.0, 0.0),
    (5.0e00, 5.0, 2.0),
    (5.0e00, 5.0, 50.0),
]

# Canonical labels for the 14 KMS regimes. The order must stay aligned with
# _KAPPA_MU_SHADOWED_REGIMES so label-based selection resolves predictably.
KMS_REGIME_LABELS = (
    "kappa0_mu1_m0",
    "kappa0_mu1_m2",
    "kappa0_mu1_m50",
    "kappa0_mu2_m50",
    "kappa0_mu5_m50",
    "kappa5_mu1_m0",
    "kappa5_mu1_m2",
    "kappa5_mu1_m50",
    "kappa5_mu2_m0",
    "kappa5_mu2_m2",
    "kappa5_mu2_m50",
    "kappa5_mu5_m0",
    "kappa5_mu5_m2",
    "kappa5_mu5_m50",
)


class DatasetRegime(NamedTuple):
    """Typed representation of a kappa-mu-shadowed channel regime configuration."""

    kappa: float
    mu: float
    m: float

    @property
    def filename_key(self) -> str:
        """Generate the standardized parameter key for file matching."""
        return f"kappa{self.kappa:.1e}_mu{self.mu}_m{self.m:.1f}"


def _resolve_selected_kms_regime_indices(
    regime_labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[int, ...]:
    """Resolve a label-based KMS regime selection into canonical indices.

    Args:
        regime_labels (list[str] | tuple[str, ...] | None): Optional
            user-provided regime labels. If ``None``, all 14 regimes are
            selected. If a list or tuple is provided, each entry must be one of
            ``KMS_REGIME_LABELS``. Empty inputs, duplicate labels, and unknown
            labels are rejected explicitly so training configuration mistakes
            fail fast instead of silently changing the dataset composition.

    Returns:
        tuple[int, ...]: Canonical regime indices corresponding to the
            requested labels. Even if labels are provided in a different order,
            the returned indices follow ``KMS_REGIME_LABELS`` order so all
            downstream loaders stay deterministic.

    Raises:
        TypeError: If ``regime_labels`` is not ``None`` and not a list or
            tuple of strings.
        ValueError: If ``regime_labels`` is empty, contains duplicates, or
            contains unknown labels.

    Examples:
        >>> _resolve_selected_kms_regime_indices(None)[0]
        0
        >>> _resolve_selected_kms_regime_indices(["kappa5_mu5_m50"])
        (13,)
    """
    if regime_labels is None:
        return tuple(range(len(KMS_REGIME_LABELS)))

    if not isinstance(regime_labels, (list, tuple)):
        raise TypeError("regime_labels must be None, a list[str], or a tuple[str, ...].")
    if len(regime_labels) == 0:
        raise ValueError("regime_labels must be None or a non-empty list/tuple of KMS labels.")
    if not all(isinstance(label, str) for label in regime_labels):
        raise TypeError("Every value in regime_labels must be a string.")

    # Validate the selection explicitly so the examples fail on typos, repeated
    # labels, or unsupported regimes instead of silently loading the wrong data.
    seen_labels = set()
    duplicate_labels = []
    for label in regime_labels:
        if label in seen_labels and label not in duplicate_labels:
            duplicate_labels.append(label)
        seen_labels.add(label)

    if duplicate_labels:
        raise ValueError(f"Duplicate KMS regime label(s): {duplicate_labels}.")

    unknown_labels = [label for label in regime_labels if label not in KMS_REGIME_LABELS]
    if unknown_labels:
        raise ValueError(
            f"Unknown KMS regime label(s): {unknown_labels}. "
            f"Valid labels are: {list(KMS_REGIME_LABELS)}."
        )

    selected_label_set = set(regime_labels)
    return tuple(
        index
        for index, label in enumerate(KMS_REGIME_LABELS)
        if label in selected_label_set
    )


def log_print(function_name: str, message: str) -> None:
    """Print a standardized log line.

    Args:
        function_name (str): Function or block identifier that emits the
            message. If this name changes, log filtering by source function
            changes accordingly. Keeping stable names improves greppability
            across runs.
        message (str): Human-readable message body. If this text is verbose,
            console output grows proportionally; concise messages reduce noise
            during large studies.

    Returns:
        None: Writes one formatted line to stdout.

    Raises:
        OSError: If stdout write fails.

    Examples:
        >>> log_print("split_kms_60_20_20_cache", "Loaded dataset split")
    """
    print(f"{BLUE}[{_COMPANION_ROOT / 'src/utils/data.py'.stem}:{function_name}] {message}{RESET}")


# ———————————————————————————————————————————————————————————————————————————— #
#                                Raw KMS Loader                                #
# ———————————————————————————————————————————————————————————————————————————— #
def _load_kms_dataset_file(common_path: str, kappa: float, mu: float, m: float) -> np.ndarray:
    """Load a single kappa-mu-shadowed SNR_events dataset file.

    Args:
        common_path (str): Base directory containing dataset files.
        kappa (float): Kappa parameter (shape parameter).
        mu (float): Mu parameter (scale parameter).
        m (float): M parameter (shadowing parameter).

    Returns:
        np.ndarray: Loaded SNR_events matrix.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the SNR_events key is missing from the file.
    """
    # Construct filename using standardized naming convention
    filename = f"SNR_events_W1.0_U1_N100_kappa{kappa:.1e}_mu{mu}_m{m}.mat"
    filepath = f"{common_path}/{filename}"
    return scipy.io.loadmat(filepath)["SNR_events"]


def load_raw_kms(common_path: str = None):
    """Load all kappa-mu shadowed datasets used in OP studies.

    The function loads 14 SNR_events matrices that cover multiple
    combinations of kappa, mu, and m parameters. Datasets are returned
    in the order defined by _KAPPA_MU_SHADOWED_REGIMES.

    Args:
        common_path (str | None): Base directory used to resolve all SNR_events file paths.
            If a non-empty path is provided, that path is used directly.
            If None, DEFAULT_DATASET_ROOT_PATH is used.

    Returns:
        tuple[np.ndarray, ...]:
            Fourteen 2D arrays in the order defined by _KAPPA_MU_SHADOWED_REGIMES.

    Raises:
        FileNotFoundError: If any expected .mat file does not exist.
        KeyError: If the SNR_events key is missing in a file.
        OSError: If SciPy cannot read a dataset file.

    Examples:
        >>> datasets = load_raw_kms()
        >>> len(datasets)
        14
    """
    if common_path is None:
        common_path = DEFAULT_DATASET_ROOT_PATH

    datasets = []
    for kappa, mu, m in _KAPPA_MU_SHADOWED_REGIMES:
        data = _load_kms_dataset_file(common_path, kappa, mu, m)
        datasets.append(data)

    return tuple(datasets)


# ———————————————————————————————————————————————————————————————————————————— #
#                          KMS Runtime Cache Loaders                           #
# ———————————————————————————————————————————————————————————————————————————— #
def _build_kms_runtime_cache_filename(kappa: float, mu: float, m: float) -> str:
    """Build the canonical runtime-cache filename for one KMS regime.

    Args:
        kappa (float): Kappa parameter encoded into the cache filename. Each
            distinct value identifies a different propagation regime and
            therefore a different cache file.
        mu (float): Mu parameter encoded into the cache filename. Different
            values identify different cluster-count regimes and therefore
            different files.
        m (float): Shadowing parameter encoded into the cache filename.
            Different values identify different shadowing regimes and therefore
            different files.

    Returns:
        str: Runtime-cache filename ending in ``.float32.npy``.

    Raises:
        None.

    Examples:
        >>> _build_kms_runtime_cache_filename(1.0e-16, 1.0, 2.0)
        'SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m2.0.float32.npy'
    """
    return f"SNR_events_W1.0_U1_N100_kappa{kappa:.1e}_mu{mu}_m{m}.float32.npy"


def _resolve_kms_runtime_cache_root(cache_path: str = None) -> Path:
    """Resolve the runtime-cache root directory for KMS data.

    Args:
        cache_path (str | None): Directory containing the derived ``.npy``
            cache files. If a non-empty path is provided, that path is used
            directly. If ``None``, ``DEFAULT_KMS_RUNTIME_CACHE_PATH`` is used.

    Returns:
        Path: Resolved runtime-cache directory path.

    Raises:
        None.

    Examples:
        >>> _resolve_kms_runtime_cache_root().name
        'kms_runtime_cache_float32'
    """
    if cache_path is None:
        return Path(DEFAULT_KMS_RUNTIME_CACHE_PATH)
    return Path(cache_path)


def _get_kms_runtime_cache_paths(cache_root: Path) -> tuple[Path, ...]:
    """Return all expected runtime-cache file paths in canonical regime order.

    Args:
        cache_root (Path): Directory containing one cached ``.npy`` file per
            KMS regime. Different directories allow the same loader logic to
            target alternate derived-cache locations without changing the file
            naming scheme.

    Returns:
        tuple[Path, ...]: Expected cache file paths for all 14 KMS regimes.

    Raises:
        None.

    Examples:
        >>> len(_get_kms_runtime_cache_paths(Path('/tmp/cache')))
        14
    """
    return tuple(
        cache_root / _build_kms_runtime_cache_filename(kappa, mu, m)
        for kappa, mu, m in _KAPPA_MU_SHADOWED_REGIMES
    )


def _validate_kms_runtime_cache(
    cache_root: Path,
    regime_indices: tuple[int, ...] | None = None,
) -> tuple[Path, ...]:
    """Ensure the derived KMS runtime cache exists before loading from it.

    Args:
        cache_root (Path): Directory expected to contain per-regime cache
            files. The function checks only the requested regime files so
            subset training can run without forcing unrelated regimes to be
            present in the runtime cache.
        regime_indices (tuple[int, ...] | None): Canonical regime indices to
            validate. If ``None``, the full 14-regime cache is required. If a
            tuple is provided, only those selected regime files are required.
            This changes validation scope only; returned path order always
            remains canonical.

    Returns:
        tuple[Path, ...]: Validated cache paths in canonical regime order.

    Raises:
        FileNotFoundError: If the cache directory is missing or any required
            selected cache file is absent. The error message explicitly tells
            the user which generator script to run.

    Examples:
        >>> paths = _validate_kms_runtime_cache(Path('/tmp/cache'), regime_indices=(0,))
        >>> isinstance(paths, tuple)
        True
    """
    all_cache_paths = _get_kms_runtime_cache_paths(cache_root)
    if regime_indices is None:
        expected_paths = all_cache_paths
        expected_regime_labels = KMS_REGIME_LABELS
    else:
        expected_paths = tuple(all_cache_paths[index] for index in regime_indices)
        expected_regime_labels = tuple(KMS_REGIME_LABELS[index] for index in regime_indices)

    missing_paths = [path for path in expected_paths if not path.exists()]

    if not missing_paths:
        return expected_paths

    raise FileNotFoundError(
        "KMS runtime cache is missing. "
        f"Expected cache files under {cache_root} for regimes {list(expected_regime_labels)}. "
        "Run python src/data/generate_kms_runtime_cache.py first."
    )


def _load_kms_runtime_cache_file(cache_file_path: Path) -> np.ndarray:
    """Open one derived KMS runtime-cache file as a read-only memory map.

    Args:
        cache_file_path (Path): Path to a cached regime file saved as an
            uncompressed ``float32`` ``.npy`` array. Different paths point to
            different regimes and therefore different dataset subsets.

    Returns:
        np.ndarray: Read-only memory-mapped array with shape
            ``(n_samples, n_ports)`` and dtype ``float32``.

    Raises:
        FileNotFoundError: If ``cache_file_path`` does not exist.
        ValueError: If the cached array is not a 2D ``float32`` matrix.
        OSError: If NumPy cannot open the cache file.

    Examples:
        >>> isinstance(_build_kms_runtime_cache_filename(1.0e-16, 1.0, 0.0), str)
        True
    """
    if not cache_file_path.exists():
        raise FileNotFoundError(f"Missing KMS runtime cache file: {cache_file_path}")

    cached_array = np.load(cache_file_path, mmap_mode="r", allow_pickle=False)
    if cached_array.ndim != 2:
        raise ValueError(
            f"KMS runtime cache file must contain a 2D matrix, got shape {cached_array.shape} "
            f"at {cache_file_path}."
        )
    if cached_array.dtype != np.float32:
        raise ValueError(
            f"KMS runtime cache file must use dtype float32, got {cached_array.dtype} "
            f"at {cache_file_path}."
        )
    return cached_array


def generate_kms_runtime_cache(
    common_path: str = None,
    cache_path: str = None,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> tuple[Path, ...]:
    """Generate the derived ``float32`` KMS runtime cache from MATLAB files.

    Args:
        common_path (str | None): Base directory containing the source MATLAB
            ``SNR_events`` files. If a non-empty path is provided, that path is
            used directly. If ``None``, ``DEFAULT_DATASET_ROOT_PATH`` is used.
        cache_path (str | None): Destination directory for the derived runtime
            cache. If a non-empty path is provided, that path is used
            directly. If ``None``, ``DEFAULT_KMS_RUNTIME_CACHE_PATH`` is used.
            The function creates the directory if it does not already exist.
        force_rebuild (bool): Controls overwrite behavior for existing cache
            files. If ``False``, existing cache files are reused and only
            missing files are generated. If ``True``, every regime cache file
            is regenerated from the MATLAB source files, which is slower but
            guarantees a fresh cache snapshot.
        verbose (bool): Controls progress logging for cache generation. If
            ``True``, the function prints the source path, cache path, and one
            progress line per regime describing whether that cache file is
            being reused or regenerated. If ``False``, the function runs
            silently and only returns the written paths.

    Returns:
        tuple[Path, ...]: Paths of cache files written during this call. If the
            cache was already complete and ``force_rebuild`` is ``False``, this
            tuple is empty.

    Raises:
        FileNotFoundError: If any source MATLAB file is missing.
        KeyError: If any source MATLAB file does not contain ``SNR_events``.
        ValueError: If a source dataset is not a 2D matrix.
        OSError: If the cache directory or files cannot be written.

    Examples:
        >>> isinstance(generate_kms_runtime_cache, object)
        True
    """
    if common_path is None:
        common_path = DEFAULT_DATASET_ROOT_PATH

    cache_root = _resolve_kms_runtime_cache_root(cache_path=cache_path)
    cache_root.mkdir(parents=True, exist_ok=True)

    written_paths = []
    n_regimes = len(_KAPPA_MU_SHADOWED_REGIMES)
    log_tag = f"[{_COMPANION_ROOT / 'src/utils/data.py'.stem}:generate_kms_runtime_cache]"

    if verbose:
        print(f"{BLUE}{log_tag} Source dataset path: {common_path}{RESET}")
        print(f"{BLUE}{log_tag} Runtime cache path: {cache_root}{RESET}")
        print(f"{BLUE}{log_tag} Force rebuild: {force_rebuild}{RESET}")

    # Convert one regime at a time so the cache generator keeps peak memory
    # bounded to one source matrix plus one float32 cache copy.
    for regime_index, (kappa, mu, m) in enumerate(_KAPPA_MU_SHADOWED_REGIMES, start=1):
        cache_file_path = cache_root / _build_kms_runtime_cache_filename(kappa, mu, m)
        regime_label = f"kappa={kappa:.1e}, mu={mu}, m={m}"

        if cache_file_path.exists() and not force_rebuild:
            # Validate reused cache files so the generator fails explicitly
            # instead of silently trusting an incompatible stale cache.
            _load_kms_runtime_cache_file(cache_file_path)
            if verbose:
                print(
                    f"{BLUE}{log_tag} [{regime_index}/{n_regimes}] Reusing existing cache for {regime_label}: "
                    f"{cache_file_path.name}{RESET}",
                )
            continue

        if verbose:
            print(
                f"{BLUE}{log_tag} [{regime_index}/{n_regimes}] Building cache for {regime_label}: "
                f"{cache_file_path.name}{RESET}",
            )

        source_data = _load_kms_dataset_file(common_path, kappa, mu, m)
        if source_data.ndim != 2:
            raise ValueError(
                f"Source KMS dataset must be a 2D matrix, got shape {source_data.shape} "
                f"for kappa={kappa:.1e}, mu={mu}, m={m}."
            )

        # Store the runtime cache in float32 C order because the Python-side
        # pipeline immediately casts to float32 and benefits from cheaper I/O.
        cache_data = np.array(source_data, dtype=np.float32, order="C", copy=True)
        np.save(cache_file_path, cache_data, allow_pickle=False)
        written_paths.append(cache_file_path)

        if verbose:
            print(
                f"{GREEN}{log_tag} [{regime_index}/{n_regimes}] Wrote {cache_file_path.name} "
                f"with shape {cache_data.shape}{RESET}",
            )

        del source_data
        del cache_data

    _validate_kms_runtime_cache(cache_root)

    if verbose:
        print(
            f"{GREEN}{log_tag} Cache generation finished. Files written this run: {len(written_paths)}{RESET}",
        )
    return tuple(written_paths)


def load_raw_kms_cache(
    common_path: str = None,
    cache_path: str = None,
    regime_labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    """Load all KMS regimes from the derived runtime cache.

    Args:
        common_path (str | None): Unused compatibility argument kept to mirror
            the MATLAB-backed loader signature. The cache-backed implementation
            loads only from the derived ``.npy`` cache and does not read the
            MATLAB source files at runtime.
        cache_path (str | None): Base directory containing the runtime-cache
            files. If a non-empty path is provided, that path is used directly.
            If ``None``, ``DEFAULT_KMS_RUNTIME_CACHE_PATH`` is used.
        regime_labels (list[str] | tuple[str, ...] | None): Optional subset of
            ``KMS_REGIME_LABELS`` to load. If ``None``, all regimes are loaded.
            If a list or tuple is provided, only those regimes are loaded, but
            the returned tuple still follows canonical order instead of caller
            order. This keeps downstream training splits reproducible.

    Returns:
        tuple[np.ndarray, ...]: Read-only memory-mapped ``float32`` arrays for
            the requested regimes in canonical KMS regime order.

    Raises:
        FileNotFoundError: If the runtime cache is missing.
        TypeError: If ``regime_labels`` is not ``None`` and not a list or
            tuple of strings.
        ValueError: If any cache file has an unexpected shape or dtype.
            Also raised if ``regime_labels`` contains duplicates or unknown
            regime names.
        OSError: If NumPy cannot read a cache file.

    Examples:
        >>> callable(load_raw_kms_cache)
        True
    """
    del common_path

    cache_root = _resolve_kms_runtime_cache_root(cache_path=cache_path)
    regime_indices = _resolve_selected_kms_regime_indices(regime_labels=regime_labels)
    cache_paths = _validate_kms_runtime_cache(cache_root, regime_indices=regime_indices)

    datasets = []
    for cache_file_path in cache_paths:
        datasets.append(_load_kms_runtime_cache_file(cache_file_path))

    return tuple(datasets)


# ———————————————————————————————————————————————————————————————————————————— #
#                                 Split Makers                                 #
# ———————————————————————————————————————————————————————————————————————————— #

def _limit_subset_samples_per_regime(
    data: np.ndarray,
    samples_per_regime: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select an exact per-regime sample budget before splitting.

    Args:
        data (np.ndarray): Input regime matrix with samples on axis 0. The
            returned rows are selected only from this matrix, so each regime is
            budgeted independently.
        samples_per_regime (int | None): Optional number of rows to keep.
            If ``None``, ``data`` is returned unchanged, preserving full-data
            behavior and avoiding an extra indexing allocation.
            If an integer is provided, exactly that many rows are selected by a
            seeded permutation. Smaller budgets reduce memory and runtime for
            learning-curve experiments but also increase sampling variance.
        rng (np.random.Generator): Random generator shared by the split helper.
            Reusing the same generator ties sample-budget selection and split
            assignment to the existing ``seed`` argument. Changing the seed
            changes both selected rows and train/validation/test membership.

    Returns:
        np.ndarray: Original data when ``samples_per_regime`` is ``None``;
            otherwise a row-indexed subset with exactly
            ``samples_per_regime`` samples.

    Raises:
        TypeError: If ``samples_per_regime`` is not ``None`` and not an
            integer.
        ValueError: If ``data`` is empty, if the requested budget is below
            ``5``, or if it is larger than the available regime size.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> data = np.ones((10, 2), dtype=np.float32)
        >>> _limit_subset_samples_per_regime(data, 5, rng).shape
        (5, 2)
    """
    if data.shape[0] == 0:
        raise ValueError("Cannot sample from an empty subset.")
    if samples_per_regime is None:
        return data
    if not isinstance(samples_per_regime, int):
        raise TypeError("samples_per_regime must be None or an integer.")
    if samples_per_regime < 5:
        raise ValueError("samples_per_regime must be at least 5.")
    if samples_per_regime > data.shape[0]:
        raise ValueError(
            "samples_per_regime cannot exceed the number of available samples "
            f"for a selected regime: requested={samples_per_regime}, available={data.shape[0]}."
        )

    # Select the budget before splitting so every selected regime contributes
    # the same number of samples to multi-regime learning-curve experiments.
    permutation = rng.permutation(data.shape[0])
    return data[permutation[:samples_per_regime]]


def _log_kms_split_sample_report(
    function_name: str,
    regime_labels: list[str] | tuple[str, ...] | None,
    train_parts: list[np.ndarray] | tuple[np.ndarray, ...],
    val_parts: list[np.ndarray] | tuple[np.ndarray, ...],
    test_parts: list[np.ndarray] | tuple[np.ndarray, ...],
) -> None:
    """Print a per-regime sample report for one cache-backed KMS split.

    Args:
        function_name (str): Log source identifier passed through to
            ``log_print`` so the report lines remain easy to grep alongside the
            rest of the split helper output.
        regime_labels (list[str] | tuple[str, ...] | None): Selected regime
            labels for the split.
            If ``None``, this report covers all regimes in canonical order.
            If a list or tuple is provided, only those regimes are reported,
            still ordered canonically to match the returned split parts.
        train_parts (list[np.ndarray] | tuple[np.ndarray, ...]): Per-regime
            training subsets returned by the split helper. Each element must be
            a 2D array with one row per sample.
        val_parts (list[np.ndarray] | tuple[np.ndarray, ...]): Per-regime
            validation subsets aligned positionally with ``train_parts``.
        test_parts (list[np.ndarray] | tuple[np.ndarray, ...]): Per-regime
            test subsets aligned positionally with ``train_parts``.

    Returns:
        None: This function writes a multi-line report to stdout.

    Raises:
        ValueError: If the per-regime split parts do not have matching lengths
            or if the resolved regime labels do not align with the provided
            split parts.

    Examples:
        >>> sample = np.ones((2, 3), dtype=np.float32)
        >>> _log_kms_split_sample_report(
        ...     "demo",
        ...     ["kappa5_mu5_m50"],
        ...     [sample],
        ...     [sample],
        ...     [sample],
        ... )
    """
    if not (len(train_parts) == len(val_parts) == len(test_parts)):
        raise ValueError("train_parts, val_parts, and test_parts must have matching lengths.")

    selected_regime_indices = _resolve_selected_kms_regime_indices(regime_labels)
    selected_regime_labels = tuple(KMS_REGIME_LABELS[index] for index in selected_regime_indices)
    if len(selected_regime_labels) != len(train_parts):
        raise ValueError(
            "Resolved regime label count does not match the provided split parts: "
            f"labels={len(selected_regime_labels)}, train_parts={len(train_parts)}."
        )

    log_print(
        function_name,
        f"Per-regime sample report for {len(selected_regime_labels)} selected regime(s):",
    )

    total_train_samples = 0
    total_val_samples = 0
    total_test_samples = 0

    # Print one line per selected regime so callers can confirm both the
    # selected labels and the resulting train/val/test sample counts.
    for regime_label, train_subset, val_subset, test_subset in zip(
        selected_regime_labels,
        train_parts,
        val_parts,
        test_parts,
        strict=True,
    ):
        train_count = int(train_subset.shape[0])
        val_count = int(val_subset.shape[0])
        test_count = int(test_subset.shape[0])
        total_count = train_count + val_count + test_count

        total_train_samples += train_count
        total_val_samples += val_count
        total_test_samples += test_count

        log_print(
            function_name,
            f"Regime {regime_label}: total={total_count}, train={train_count}, val={val_count}, test={test_count}",
        )

    total_selected_samples = total_train_samples + total_val_samples + total_test_samples
    n_ports = int(train_parts[0].shape[1]) if train_parts else 0
    log_print(
        function_name,
        (
            "Total data summary: "
            f"selected_regimes={len(selected_regime_labels)}, "
            f"ports_per_sample={n_ports}, "
            f"train_samples={total_train_samples}, "
            f"val_samples={total_val_samples}, "
            f"test_samples={total_test_samples}, "
            f"total_samples={total_selected_samples}"
        ),
    )


def split_kms_ratio_cache(
    common_path: str = None,
    cache_path: str = None,
    seed: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    regime_labels: list[str] | tuple[str, ...] | None = None,
    samples_per_regime: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Split cache-backed KMS regimes into concatenated train/val/test sets.

    This unified splitter applies an optional per-regime sample budget and
    then splits each selected regime according to the provided three-way
    ratios. The per-regime parts are concatenated in canonical regime order
    and returned along with the per-regime parts for auditing.

    Note: ``common_path`` is accepted for backward compatibility but ignored
    by the cache-backed implementation (the cache path is used instead).

    Args:
        common_path (str | None): Ignored compatibility argument.
        cache_path (str | None): Base directory containing the runtime-cache
            ``.npy`` files. If ``None``, ``DEFAULT_KMS_RUNTIME_CACHE_PATH`` is used.
        seed (int): RNG seed for reproducible per-regime permutations.
        train_ratio (float): Fraction assigned to the training split. Must be
            in ``[0.0, 1.0]``.
        val_ratio (float): Fraction assigned to the validation split. Must be
            in ``[0.0, 1.0]``.
        test_ratio (float): Fraction assigned to the test split. Must be
            in ``[0.0, 1.0]``. The three ratios must sum to no more than 1.0
            within a small numerical tolerance. If they sum to less than 1.0,
            the remaining shuffled rows are intentionally unused.
        regime_labels (list[str] | tuple[str, ...] | None): Optional subset of
            canonical regime labels to include.
        samples_per_regime (int | None): Optional per-regime sample budget
            applied before splitting (keeps per-regime balance).
        verbose (bool): Controls summary logging.

    Returns:
        tuple: ``(train_dataset, val_dataset, test_dataset,
            train_parts, val_parts, test_parts)`` where concatenated datasets are
            ``float32`` arrays and parts are tuples of per-regime arrays.

    Raises:
        FileNotFoundError: If the runtime cache is missing.
        ValueError: If ratios are invalid, if their sum exceeds 1.0, or if a
            subset cannot be split into the requested non-empty parts when a
            positive ratio is requested.
    """
    # Keep the compatibility parameter to avoid breaking callers that still
    # pass ``common_path``; the cache-backed loader ignores it.
    del common_path

    # Validate ratios
    for r in (train_ratio, val_ratio, test_ratio):
        if not (0.0 <= r <= 1.0):
            raise ValueError("All ratios must be in [0.0, 1.0].")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum > 1.0 and not np.isclose(ratio_sum, 1.0, atol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must be less than or equal to 1.0.")
    use_all_rows = np.isclose(ratio_sum, 1.0, atol=1e-8)

    rng = np.random.default_rng(seed)
    subsets = load_raw_kms_cache(cache_path=cache_path, regime_labels=regime_labels)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for subset in subsets:
        # Optionally cap the per-regime samples first
        budgeted_subset = _limit_subset_samples_per_regime(
            data=subset, samples_per_regime=samples_per_regime, rng=rng
        )

        n = int(budgeted_subset.shape[0])
        if n == 0:
            raise ValueError("Cannot split an empty subset.")

        # Compute integer part sizes with deterministic rounding
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        if use_all_rows:
            n_test = n - (n_train + n_val)
        else:
            n_test = int(np.floor(n * test_ratio))

        # Ensure requested positive-ratio parts are non-empty
        if train_ratio > 0.0 and n_train == 0:
            raise ValueError("train_ratio is too small for this subset size.")
        if val_ratio > 0.0 and n_val == 0:
            raise ValueError("val_ratio is too small for this subset size.")
        if test_ratio > 0.0 and n_test == 0:
            raise ValueError("test_ratio is too small for this subset size.")

        perm = rng.permutation(n)
        train_end = n_train
        val_end = train_end + n_val
        test_end = val_end + n_test

        train_subset = budgeted_subset[perm[:train_end]]
        val_subset = budgeted_subset[perm[train_end:val_end]]
        test_subset = budgeted_subset[perm[val_end:test_end]]

        train_parts.append(np.asarray(train_subset, dtype=np.float32))
        val_parts.append(np.asarray(val_subset, dtype=np.float32))
        test_parts.append(np.asarray(test_subset, dtype=np.float32))

    train_dataset = np.concatenate(train_parts, axis=0).astype(np.float32)
    val_dataset = np.concatenate(val_parts, axis=0).astype(np.float32)
    test_dataset = np.concatenate(test_parts, axis=0).astype(np.float32)

    if verbose:
        _log_kms_split_sample_report(
            function_name="split_kms_ratio_cache",
            regime_labels=regime_labels,
            train_parts=train_parts,
            val_parts=val_parts,
            test_parts=test_parts,
        )
        log_print("split_kms_ratio_cache", f"Train dataset shape: {train_dataset.shape}")
        log_print("split_kms_ratio_cache", f"Validation dataset shape: {val_dataset.shape}")
        log_print("split_kms_ratio_cache", f"Test dataset shape: {test_dataset.shape}")
        log_print("split_kms_ratio_cache", f"Train dataset parts: {len(train_parts)}")
        log_print("split_kms_ratio_cache", f"Validation dataset parts: {len(val_parts)}")
        log_print("split_kms_ratio_cache", f"Test dataset parts: {len(test_parts)}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        tuple(train_parts),
        tuple(val_parts),
        tuple(test_parts),
    )


def split_kms_60_20_20_cache(
    common_path: str = None,
    cache_path: str = None,
    seed: int = 0,
    regime_labels: list[str] | tuple[str, ...] | None = None,
    samples_per_regime: int | None = None,
    verbose: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """Split cache-backed KMS data into 60/20/20 partitions.

    The split is applied independently to each selected regime and only then
    concatenated. This preserves the repo's existing per-regime balancing
    strategy. By default, the full cache-backed dataset is used. If a sample
    budget is provided, each selected regime is first reduced to exactly that
    many samples before the 60/20/20 split is applied.

    Args:
        common_path (str | None): Unused compatibility argument kept to mirror
            the MATLAB-backed function signatures. The cache-backed
            implementation reads only from the derived ``.npy`` cache.
        cache_path (str | None): Base directory containing the runtime-cache
            files. If a non-empty path is provided, that path is used directly.
            If ``None``, ``DEFAULT_KMS_RUNTIME_CACHE_PATH`` is used.
        seed (int): Seed used to initialize ``numpy.random.default_rng``. A
            fixed value makes the 60/20/20 assignment reproducible. Changing it
            changes which rows land in train, validation, and test for every
            selected regime.
        regime_labels (list[str] | tuple[str, ...] | None): Optional subset of
            ``KMS_REGIME_LABELS`` to include. If ``None``, all regimes are
            split. If a list or tuple is provided, only those regimes are
            included, and they are resolved in canonical order rather than the
            caller's order.
        samples_per_regime (int | None): Optional sample budget applied
            independently to every selected regime before splitting.
            If ``None``, every available sample from each selected regime is
            used, preserving the existing full-data behavior and memory usage.
            If an integer is provided, each selected regime is reduced to
            exactly that many rows using the seeded random generator before the
            60/20/20 split. This keeps regimes balanced for multi-regime
            training and reduces memory use, but the selected subset changes
            when ``seed`` changes. Values below ``5`` are rejected because they
            cannot produce meaningful non-empty train, validation, and test
            partitions under the project protocol.
        verbose (bool): Controls summary logging. If ``True``, the function
            prints the concatenated split shapes plus the number of per-regime
            parts. If ``False``, it runs silently.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
            ``(train_dataset, val_dataset, test_dataset, train_parts, val_parts, test_parts)``
            where:
            - ``train_dataset`` is the concatenated 60% training matrix.
            - ``val_dataset`` is the concatenated 20% validation matrix.
            - ``test_dataset`` is the concatenated 20% test matrix.
            - ``train_parts`` contains one per-regime training matrix.
            - ``val_parts`` contains one per-regime validation matrix.
            - ``test_parts`` contains one per-regime test matrix.
            All returned arrays are materialized ``float32`` matrices.

    Raises:
        FileNotFoundError: If the runtime cache is missing for any required
            selected regime.
        TypeError: If ``regime_labels`` is not ``None`` and not a list or
            tuple of strings.
        TypeError: If ``samples_per_regime`` is not ``None`` and not an
            integer.
        ValueError: If any subset is empty, too small for a non-empty 60/20/20
            split, if ``samples_per_regime`` is smaller than ``5`` or larger
            than a selected regime, or if ``regime_labels`` is empty,
            duplicated, or unknown.

    Examples:
        >>> callable(split_kms_60_20_20_cache)
        True
    """
    return split_kms_ratio_cache(
        common_path=common_path,
        cache_path=cache_path,
        seed=seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        regime_labels=regime_labels,
        samples_per_regime=samples_per_regime,
        verbose=verbose,
    )


# ———————————————————————————————————————————————————————————————————————————— #
#                       Observed Ports Sampling and Utils                      #
# ———————————————————————————————————————————————————————————————————————————— #
def get_random_observed_ports(
    sinr_data: np.ndarray,
    num_observed_ports: int,
    total_ports: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select uniformly random distinct observed ports from a full SINR matrix.

    The sampling is performed without replacement so every selected index is unique.
    All index combinations of size ``num_observed_ports`` are equiprobable under
    ``numpy.random.Generator.choice(..., replace=False)``.

    Args:
        sinr_data (np.ndarray): Full SINR matrix with shape
            ``(n_samples, total_ports)``.
        num_observed_ports (int): Number of ports to sample and observe.
            If this value is higher, more channel information is available
            to downstream OP estimators at the cost of observing more ports.
            If this value is lower, sensing cost is reduced but observed-only
            OP can degrade because fewer ports are probed.
        total_ports (int): Total number of candidate ports represented in
            ``sinr_data``.
        rng (np.random.Generator): Random number generator used to sample
            unique indices. Passing a generator with a fixed seed enables
            reproducible selections; passing a fresh/unseeded generator yields
            non-deterministic samples.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(observed_sinr, observed_indices)`` where:
            - ``observed_sinr`` has shape ``(n_samples, num_observed_ports)``.
            - ``observed_indices`` contains sorted sampled column indices.

    Raises:
        ValueError: If ``num_observed_ports`` is not in ``[1, total_ports]``.
        ValueError: If ``sinr_data`` does not have ``total_ports`` columns.

    Examples:
        >>> rng = np.random.default_rng(111)
        >>> sample = np.arange(20, dtype=float).reshape(2, 10)
        >>> observed, idxs = get_random_observed_ports(sample, 3, 10, rng)
        >>> observed.shape
        (2, 3)
    """
    # Validate input dimensions early to fail fast with a clear message.
    if sinr_data.ndim != 2 or sinr_data.shape[1] != total_ports:
        raise ValueError("sinr_data must be a 2D array with shape (n_samples, total_ports).")

    # Sampling without replacement requires the requested budget to be within
    # the available port range and strictly positive.
    if num_observed_ports < 1 or num_observed_ports > total_ports:
        raise ValueError("num_observed_ports must satisfy 1 <= num_observed_ports <= total_ports.")

    # Draw a uniform subset of distinct indices and sort it so downstream
    # processing remains stable and easier to inspect.
    observed_indices = np.sort(rng.choice(total_ports, size=num_observed_ports, replace=False))
    observed_sinr = sinr_data[:, observed_indices]
    return observed_sinr, observed_indices


def get_observed_ports(sinr_data, num_observed_ports, total_ports):
    """Select equally spaced observed ports from a full SINR matrix.

    Args:
        sinr_data (np.ndarray): Full SINR matrix with shape
            ``(n_samples, total_ports)``.
        num_observed_ports (int): Number of ports to keep as observed features.
            Larger values preserve more information but increase input size.
            Smaller values reduce sensing/feature cost but may reduce accuracy.
        total_ports (int): Total number of ports represented in ``sinr_data``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(observed_sinr, observed_indices)`` where:
            - ``observed_sinr`` has shape ``(n_samples, num_observed_ports)``.
            - ``observed_indices`` contains the selected column indices.

    Raises:
        IndexError: If ``total_ports`` is incompatible with ``sinr_data`` width.
        ValueError: If NumPy cannot generate valid evenly spaced indices.
    """
    observed_indices = np.linspace(0, total_ports - 1, num_observed_ports, dtype=int)
    observed_sinr = sinr_data[:, observed_indices]

    return observed_sinr, observed_indices


def getOP(
    observed_indices: np.ndarray,
    predicted_values: np.ndarray,
    true_values: np.ndarray,
    threshold: float,
    snr_linear: float,
    total_ports: int,
    combine_observed_and_predicted: bool = True,
) -> float:
    """Estimate outage probability using observed and model-predicted port choices.

     For each sample, this function supports two modes:
     1. If ``combine_observed_and_predicted=True`` (default), it identifies the
         best observed port, identifies the model-predicted best port, and uses
         the better of the two.
     2. If ``combine_observed_and_predicted=False``, it uses only the model-
         predicted best port.

     In both modes, outage probability is computed as
     ``1 - P(best_selected_value > threshold / snr_linear)``.

    Args:
        observed_indices (np.ndarray): 1D array of port indices that are physically
            observable at runtime. Must be sorted for consistency.
        predicted_values (np.ndarray): 2D array of shape (n_samples, total_ports)
            containing model predictions/scores for every port. The argmax of
            each row selects the predicted best port.
        true_values (np.ndarray): 2D array of shape (n_samples, total_ports)
            containing ground-truth SINR values for all ports. Used to evaluate
            the actual quality of predicted and observed selections.
        threshold (float): Outage threshold in linear domain (numerator).
        snr_linear (float): SNR scaling in linear domain (denominator).
            The actual comparison is: value > (threshold / snr_linear).
        total_ports (int): Total number of ports in the channel vector.
            Defines the matrix width for masking.
        combine_observed_and_predicted (bool): Selection mode switch.
            If ``True`` (default), combine observed and predicted selections by
            taking the better of both per sample, which uses observed-port
            information and model output jointly.
            If ``False``, ignore observed-port assistance and use model-only
            selection based on ``argmax(predicted_values, axis=1)``.

    Returns:
        float: Outage probability in [0, 1]. Represents the fraction of samples
            where the selected value fell below the threshold.

    Raises:
        ValueError: If input matrix dimensions are inconsistent.
        IndexError: If any observed index is out of bounds.

    Examples:
        >>> true_vals = np.random.randn(100, 10)
        >>> pred_vals = np.random.randn(100, 10)
        >>> observed_idx = np.array([0, 2, 5, 7])
        >>> op = getOP(observed_idx, pred_vals, true_vals, 2.0, 1.0, 10)
        >>> 0 <= op <= 1
        True
    """
    n_samples = true_values.shape[0]

    # Find the best port according to model predictions
    best_predicted_indices = np.argmax(predicted_values, axis=1)

    if combine_observed_and_predicted:
        # Build a matrix with only observed port values visible; mask others with -inf
        # so argmax will never select them.
        observed_values_matrix = np.full((n_samples, total_ports), -np.inf, dtype=np.float64)
        observed_values_matrix[:, observed_indices] = true_values[:, observed_indices]

        predicted_values_matrix = np.full((n_samples, total_ports), -np.inf, dtype=np.float64)
        predicted_values_matrix[np.arange(n_samples), best_predicted_indices] = true_values[
            np.arange(n_samples), best_predicted_indices
        ]

        # Combine observed and predicted selections: pick the better of the two for each sample.
        best_value_matrix = np.maximum(observed_values_matrix, predicted_values_matrix)
        best_selected_ports = np.argmax(best_value_matrix, axis=1)
        selected_values = best_value_matrix[np.arange(n_samples), best_selected_ports]
    else:
        # Model-only mode: evaluate only the model-chosen best port.
        selected_values = true_values[np.arange(n_samples), best_predicted_indices]

    # Compute outage: fraction of samples below threshold
    above_threshold = selected_values > (threshold / snr_linear)
    outage_probability = 1.0 - (np.sum(above_threshold) / n_samples)

    return outage_probability


def getObservedOP(
    observed_indices: np.ndarray, true_values: np.ndarray, threshold: float, snr_linear: float
) -> float:
    """Compute outage probability using only observed ports.

    Args:
        observed_indices (np.ndarray): Indices of observed ports.
        true_values (np.ndarray): Ground-truth SINR matrix for all ports.
        threshold (float): Outage threshold in linear domain numerator.
        snr_linear (float): SNR scaling in linear domain denominator.

    Returns:
        float: Outage probability in ``[0, 1]`` defined as
            ``1 - P(max_observed > threshold / snr_linear)``.

    Raises:
        IndexError: If any observed index is out of bounds.
        ValueError: If ``true_values`` is empty.
    """
    # extract only observed-port SINRs
    observed_sinr = true_values[:, observed_indices]
    # best per sample
    best_obs = np.max(observed_sinr, axis=1)
    # fraction above threshold
    p_above = np.mean(best_obs > (threshold / snr_linear))
    return 1.0 - p_above


def getIdealOP(true_values: np.ndarray, threshold: float, snr_linear: float) -> float:
    """Compute genie-aided outage probability assuming all ports are known.

    Args:
        true_values (np.ndarray): Ground-truth SINR matrix for all ports.
        threshold (float): Outage threshold in linear domain numerator.
        snr_linear (float): SNR scaling in linear domain denominator.

    Returns:
        float: Ideal outage probability in ``[0, 1]`` defined as
            ``1 - P(max_all_ports > threshold / snr_linear)``.

    Raises:
        ValueError: If ``true_values`` is empty.
    """
    best_all = np.max(true_values, axis=1)
    p_above = np.mean(best_all > (threshold / snr_linear))
    return 1.0 - p_above
