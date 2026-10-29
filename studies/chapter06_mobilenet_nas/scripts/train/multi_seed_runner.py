"""Helpers for running MobileNet training scripts across multiple seeds."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path


TRAIN_SEED_ENV_VAR = "MOBILENET_TRAIN_SEED"
RUN_ROOT_ENV_VAR = "MOBILENET_RUN_ROOT"
TRAIN_SEEDS = (0, 1, 2, 3, 4)
YELLOW = "\033[33m"
RESET = "\033[0m"
OBSERVED_PORTS_LIST = (3, 4, 5, 6, 7)
TRAIN_REGIME_LABELS = (
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


@dataclass(frozen=True)
class TrainingCaseSpec:
    """Describe one concrete MobileNet training case scheduled by a wrapper.

    Args:
        run_dir (Path): Final output directory for this case. The wrapper and
            child script both use this path to decide whether the case is
            already complete and can be skipped.
        num_observed_ports (int): Number of observed ports used for this case.
            Different values change the input width seen by the model and
            produce distinct run-directory names.
        regime_label (str | None): Selected regime for per-regime runs.
            If ``None``, the case belongs to an all-regimes script and the
            child script must train on the full predefined regime set. If a
            string is provided, only that regime should be used.

    Returns:
        None: This dataclass only stores immutable scheduling metadata.

    Raises:
        None.
    """

    run_dir: Path
    num_observed_ports: int
    regime_label: str | None


def build_seed_run_root(run_root: Path, train_seed: int) -> Path:
    """Return the seed-specific output root for one wrapper execution.

    Args:
        run_root (Path): Wrapper execution directory such as
            ``runs/train_per_regime`` or ``runs/train_all_regimes``. This path
            groups all seeds launched by one wrapper invocation mode.
        train_seed (int): Weight-initialization seed for the current execution.
            Different values produce different ``seed_<seed>`` subdirectories.

    Returns:
        Path: Seed-specific root path such as ``runs/run3/seed_2``.

    Raises:
        ValueError: If ``train_seed`` is negative.

    Examples:
        >>> build_seed_run_root(Path("runs/train_per_regime"), 2).as_posix()
        'runs/train_per_regime/seed_2'
    """
    if train_seed < 0:
        raise ValueError("train_seed must be non-negative.")
    return run_root / f"seed_{train_seed}"


def build_all_regimes_case_specs(
    run_root: Path,
    train_seed: int,
    variant_name: str,
    observed_ports_list: tuple[int, ...] = OBSERVED_PORTS_LIST,
) -> list[TrainingCaseSpec]:
    """Build ordered case metadata for one all-regimes training script.

    Args:
        run_root (Path): Wrapper output root such as ``runs/train_all_regimes``.
            The function places all generated case directories under the
            corresponding ``seed_<n>`` subdirectory.
        train_seed (int): Training seed used to isolate outputs. Different
            seeds produce different ``seed_<n>`` parents but keep the same case
            order and case names.
        variant_name (str): MobileNet variant suffix such as
            ``"mobv1_alpha02"``. The value is embedded directly in each
            generated run-directory name.
        observed_ports_list (tuple[int, ...]): Ordered observed-port sweep for
            this script. Reordering the tuple changes case execution order. A
            smaller tuple reduces the total case count, while a larger one adds
            more case directories to schedule.

    Returns:
        list[TrainingCaseSpec]: Ordered case list for the all-regimes script.

    Raises:
        ValueError: If ``variant_name`` is empty or ``train_seed`` is negative.

    Examples:
        >>> specs = build_all_regimes_case_specs(Path("runs/train_all_regimes"), 1, "mobv1_alpha02")
        >>> specs[0].run_dir.name
        'train_mobv1_alpha02_60_20_20_all_regimes_3_ports'
    """
    if not variant_name:
        raise ValueError("variant_name must be non-empty.")

    seed_run_root = build_seed_run_root(run_root, train_seed)
    return [
        TrainingCaseSpec(
            run_dir=seed_run_root / f"train_{variant_name}_60_20_20_all_regimes_{num_observed_ports}_ports",
            num_observed_ports=num_observed_ports,
            regime_label=None,
        )
        for num_observed_ports in observed_ports_list
    ]


def build_per_regime_case_specs(
    run_root: Path,
    train_seed: int,
    variant_name: str,
    regime_labels: tuple[str, ...] = TRAIN_REGIME_LABELS,
    observed_ports_list: tuple[int, ...] = OBSERVED_PORTS_LIST,
) -> list[TrainingCaseSpec]:
    """Build ordered case metadata for one per-regime training script.

    Args:
        run_root (Path): Wrapper output root such as ``runs/train_per_regime``.
            The function places all generated case directories under the
            corresponding ``seed_<n>`` subdirectory.
        train_seed (int): Training seed used to isolate outputs. Different
            seeds produce different ``seed_<n>`` parents but keep the same case
            order and case names.
        variant_name (str): MobileNet variant suffix such as
            ``"mobv2_alpha02"``. The value is embedded directly in each
            generated run-directory name.
        regime_labels (tuple[str, ...]): Ordered regime sweep for this script.
            Reordering the tuple changes case execution order. Removing labels
            reduces the total case count, while adding labels schedules more
            per-regime runs.
        observed_ports_list (tuple[int, ...]): Ordered observed-port sweep for
            each regime. Different tuples change both order and total count.

    Returns:
        list[TrainingCaseSpec]: Ordered case list for the per-regime script.

    Raises:
        ValueError: If ``variant_name`` is empty or ``train_seed`` is negative.

    Examples:
        >>> specs = build_per_regime_case_specs(Path("runs/train_per_regime"), 2, "mobv3_alpha02")
        >>> specs[0].run_dir.name
        'train_mobv3_alpha02_60_20_20_regime_kappa0_mu1_m0_3_ports'
    """
    if not variant_name:
        raise ValueError("variant_name must be non-empty.")

    seed_run_root = build_seed_run_root(run_root, train_seed)
    case_specs: list[TrainingCaseSpec] = []

    # Preserve the child-script execution order so wrapper summaries and
    # restart behavior line up with the actual case loop inside each script.
    for regime_label in regime_labels:
        for num_observed_ports in observed_ports_list:
            case_specs.append(
                TrainingCaseSpec(
                    run_dir=seed_run_root
                    / (
                        f"train_{variant_name}_60_20_20_"
                        f"regime_{regime_label}_{num_observed_ports}_ports"
                    ),
                    num_observed_ports=num_observed_ports,
                    regime_label=regime_label,
                )
            )
    return case_specs


def build_script_case_specs(
    script_name: str,
    run_root: Path,
    train_seed: int,
) -> list[TrainingCaseSpec]:
    """Build the expected ordered case list for one wrapper child script.

    Args:
        script_name (str): File name of the child training script.
            Names starting with ``"train_all_regimes_"`` are treated as
            all-regimes scripts. Names starting with ``"train_per_regime_"``
            are treated as per-regime scripts. Any other name is rejected
            explicitly to avoid scheduling the wrong directory pattern.
        run_root (Path): Wrapper output root used by the parent launcher. The
            resulting cases are nested under ``seed_<train_seed>``.
        train_seed (int): Training seed used for output isolation.

    Returns:
        list[TrainingCaseSpec]: Ordered case metadata for the named script.

    Raises:
        ValueError: If ``script_name`` does not match a supported MobileNet
            training-script naming convention.

    Examples:
        >>> specs = build_script_case_specs('train_all_regimes_mobv1_alpha02.py', Path('runs/train_all_regimes'), 1)
        >>> len(specs)
        5
    """
    script_match = re.fullmatch(r"train_(all_regimes|per_regime)_(.+)\.py", script_name)
    if script_match is None:
        raise ValueError(f"Unsupported training script name: {script_name}")

    mode, variant_name = script_match.groups()
    if mode == "all_regimes":
        return build_all_regimes_case_specs(run_root=run_root, train_seed=train_seed, variant_name=variant_name)
    return build_per_regime_case_specs(run_root=run_root, train_seed=train_seed, variant_name=variant_name)


def _is_non_empty_file(path: Path) -> bool:
    """Return whether a path is a regular file with content.

    Args:
        path (Path): Artifact path to inspect. Missing paths, directories, and
            zero-byte files are treated as incomplete training outputs.

    Returns:
        bool: ``True`` only when ``path`` is a file and its size is greater
            than zero bytes.

    Raises:
        None.
    """
    return path.is_file() and path.stat().st_size > 0


def is_finished_training_run(run_dir: Path) -> bool:
    """Return whether a run directory contains all required non-empty artifacts.

    Args:
        run_dir (Path): Top-level case directory to inspect. The run is
            considered complete only when the final config, model, scalers,
            history, and model-stats artifacts all exist and are non-empty.
            Partial or zero-byte outputs keep the run marked unfinished so the
            wrapper reruns it explicitly.

    Returns:
        bool: ``True`` when every required artifact exists and is non-empty,
            otherwise ``False``.

    Raises:
        None.

    Examples:
        >>> is_finished_training_run(Path('missing_case'))
        False
    """
    required_paths = (
        run_dir / "model_training" / "args" / "mobilenet_config.json",
        run_dir / "model_training" / "model" / "model.keras",
        run_dir / "model_training" / "scaler" / "scaler_X.pkl",
        run_dir / "model_training" / "scaler" / "scaler_y.pkl",
        run_dir / "model_training" / "history" / "history.csv",
        run_dir / "model_training" / "history" / "selection_history.csv",
        run_dir / "model_training" / "args" / "model_stats.txt",
    )
    return all(_is_non_empty_file(path) for path in required_paths)


def get_case_progress(case_specs: list[TrainingCaseSpec]) -> dict[str, int | str | None]:
    """Summarize completion progress for an ordered case list.

    Args:
        case_specs (list[TrainingCaseSpec]): Ordered case list for one child
            script and one seed. The function preserves the given order when it
            determines the most recent finished case.

    Returns:
        dict[str, int | str | None]: Progress summary containing ``total``,
            ``finished``, ``remaining``, and ``last_finished_case``. The last
            finished case is ``None`` when no case artifacts are complete yet.

    Raises:
        None.

    Examples:
        >>> progress = get_case_progress([])
        >>> progress['remaining']
        0
    """
    finished_case_names = [spec.run_dir.name for spec in case_specs if is_finished_training_run(spec.run_dir)]
    finished_count = len(finished_case_names)
    total_count = len(case_specs)
    return {
        "total": total_count,
        "finished": finished_count,
        "remaining": total_count - finished_count,
        "last_finished_case": finished_case_names[-1] if finished_case_names else None,
    }


def format_case_progress_message(
    script_name: str,
    train_seed: int,
    script_progress: dict[str, int | str | None],
    global_remaining: int,
    global_total: int,
) -> str:
    """Format the yellow progress line requested for wrapper scheduling.

    Args:
        script_name (str): Child training-script file name being summarized.
        train_seed (int): Current training seed shown in the progress line.
        script_progress (dict[str, int | str | None]): Progress summary from
            ``get_case_progress``. The wrapper must supply the exact dictionary
            returned by that helper so field names remain consistent.
        global_remaining (int): Number of unfinished cases left across the full
            wrapper run, including the current script and seed.
        global_total (int): Total scheduled cases across the full wrapper run.

    Returns:
        str: ANSI-colored progress message for terminal printing.

    Raises:
        KeyError: If ``script_progress`` is missing a required field.

    Examples:
        >>> message = format_case_progress_message('train_all_regimes_mobv1_alpha02.py', 1, {'total': 5, 'finished': 0, 'remaining': 5, 'last_finished_case': None}, 5, 5)
        >>> '[PROGRESS]' in message
        True
    """
    last_finished_case = script_progress["last_finished_case"]
    last_case_text = "none" if last_finished_case is None else str(last_finished_case)
    return (
        f"{YELLOW}[PROGRESS] "
        f"seed={train_seed} "
        f"script={script_name} "
        f"last_done={last_case_text} "
        f"script_done={script_progress['finished']}/{script_progress['total']} "
        f"remaining_total={global_remaining}/{global_total}"
        f"{RESET}"
    )


def release_training_case_resources(clear_session_callback) -> None:
    """Release TensorFlow and Python objects after one training case finishes.

    Args:
        clear_session_callback: Zero-argument callback that clears framework
            session state. In the Keras path this is
            ``tf.keras.backend.clear_session``. Passing a different callback
            changes which framework state is released, while passing the wrong
            callable can leave memory resident across cases.

    Returns:
        None: The function triggers cleanup side effects only.

    Raises:
        TypeError: If ``clear_session_callback`` is not callable.

    Examples:
        >>> calls = []
        >>> release_training_case_resources(lambda: calls.append('clear'))
        >>> calls
        ['clear']
    """
    if not callable(clear_session_callback):
        raise TypeError("clear_session_callback must be callable.")

    # Clearing the framework session releases graph, optimizer, and compiled
    # function state that otherwise accumulates across many case runs.
    clear_session_callback()

    # The explicit GC pass complements the session reset by reclaiming Python
    # objects that became unreachable only after TensorFlow released them.
    gc.collect()


def read_required_train_seed() -> int:
    """Read the wrapper-provided training seed from the environment.

    Args:
        None.

    Returns:
        int: Parsed integer value from ``MOBILENET_TRAIN_SEED``.

    Raises:
        RuntimeError: If the environment variable is missing or not an integer.

    Examples:
        >>> previous = os.environ.get(TRAIN_SEED_ENV_VAR)
        >>> os.environ[TRAIN_SEED_ENV_VAR] = '3'
        >>> read_required_train_seed()
        3
        >>> if previous is None:
        ...     del os.environ[TRAIN_SEED_ENV_VAR]
        ... else:
        ...     os.environ[TRAIN_SEED_ENV_VAR] = previous
    """
    train_seed_value = os.environ.get(TRAIN_SEED_ENV_VAR)
    if train_seed_value is None:
        raise RuntimeError(
            f"Missing required environment variable {TRAIN_SEED_ENV_VAR}. "
            "Run this script through the multi-seed wrapper."
        )

    try:
        return int(train_seed_value)
    except ValueError as error:
        raise RuntimeError(
            f"Environment variable {TRAIN_SEED_ENV_VAR} must be an integer. "
            f"Received: {train_seed_value!r}."
        ) from error


def read_required_wrapper_run_root() -> Path:
    """Read the wrapper-selected fixed run root from the environment.

    Args:
        None.

    Returns:
        Path: Parsed path stored in ``MOBILENET_RUN_ROOT``.

    Raises:
        RuntimeError: If the environment variable is missing.

    Examples:
        >>> previous = os.environ.get(RUN_ROOT_ENV_VAR)
        >>> os.environ[RUN_ROOT_ENV_VAR] = 'runs/train_per_regime'
        >>> read_required_wrapper_run_root()
        PosixPath('runs/train_per_regime')
        >>> if previous is None:
        ...     del os.environ[RUN_ROOT_ENV_VAR]
        ... else:
        ...     os.environ[RUN_ROOT_ENV_VAR] = previous
    """
    run_root_value = os.environ.get(RUN_ROOT_ENV_VAR)
    if run_root_value is None:
        raise RuntimeError(
            f"Missing required environment variable {RUN_ROOT_ENV_VAR}. "
            "Run this script through the multi-seed wrapper."
        )
    return Path(run_root_value)
