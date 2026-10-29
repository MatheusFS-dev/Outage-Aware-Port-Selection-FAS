"""Run all MobileNet Keras training per regime scripts for the 60/20/20 split."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import os
import subprocess
import sys
from pathlib import Path

from multi_seed_runner import (
    RUN_ROOT_ENV_VAR,
    TRAIN_SEED_ENV_VAR,
    TRAIN_SEEDS,
    build_script_case_specs,
    format_case_progress_message,
    get_case_progress,
    is_finished_training_run,
)


RUN_ROOT_NAME = "train_per_regime"


def list_training_scripts(base_dir: Path, self_name: str) -> list[Path]:
    """Collect all training scripts in a split directory.

    Args:
        base_dir (Path): Directory that contains split-specific training scripts.
            The directory must exist and be readable.
        self_name (str): File name of the current runner script. Any path whose
            file name matches this value is excluded from the results.

    Returns:
        list[Path]: Sorted list of paths matching ``train_per_regime_*.py`` in ``base_dir``
            excluding the current runner script.

    Raises:
        FileNotFoundError: If ``base_dir`` does not exist.
        NotADirectoryError: If ``base_dir`` is not a directory.

    Examples:
        >>> scripts = list_training_scripts(Path("."), "run_all_train_per_regime_scripts_60_20_20.py")
        >>> all(path.name.startswith("train_per_regime_") for path in scripts)
        True
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_dir}")

    return sorted(
        path for path in base_dir.glob("train_per_regime_*.py") if path.name != self_name
    )


def run_all_training_scripts() -> None:
    """Execute every Keras training script for the 60/20/20 split.

    The scripts run sequentially in deterministic file-name order for each
    additional training seed. Execution stops immediately if one script exits
    with a non-zero status code.

    Args:
        None: This function does not accept arguments.

    Returns:
        None: The function returns after all scripts finish successfully.

    Raises:
        RuntimeError: If no training scripts are found in the current folder.
        subprocess.CalledProcessError: If any launched training script fails.
    """
    current_dir = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train'
    scripts = list_training_scripts(current_dir, _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/run_all_train_per_regime_scripts.py'.name)

    if not scripts:
        raise RuntimeError(f"No training scripts found in: {current_dir}")

    # Use a fixed wrapper root so reruns target the same seed subdirectories and
    # can reuse the inner training backup directories created by each script.
    run_root = current_dir / "runs" / RUN_ROOT_NAME
    run_root.mkdir(parents=True, exist_ok=True)

    all_case_specs = [
        case_spec
        for train_seed in TRAIN_SEEDS
        for script_path in scripts
        for case_spec in build_script_case_specs(script_path.name, run_root, train_seed)
    ]
    global_total = len(all_case_specs)
    global_finished = sum(1 for case_spec in all_case_specs if is_finished_training_run(case_spec.run_dir))

    # Keep the existing seed_<n> layout so each training seed still writes to a
    # stable, isolated subtree inside the fixed wrapper root.
    for train_seed in TRAIN_SEEDS:
        print(f"[SEED] train_seed={train_seed}, run_root={run_root}")
        for script_path in scripts:
            case_specs = build_script_case_specs(script_path.name, run_root, train_seed)
            script_progress = get_case_progress(case_specs)
            print(
                format_case_progress_message(
                    script_name=script_path.name,
                    train_seed=train_seed,
                    script_progress=script_progress,
                    global_remaining=global_total - global_finished,
                    global_total=global_total,
                )
            )
            if script_progress["remaining"] == 0:
                print(f"[SKIP] seed={train_seed} script={script_path.name} already complete.")
                continue

            print(f"[RUN] seed={train_seed} script={script_path.name}")
            env = os.environ.copy()
            env[TRAIN_SEED_ENV_VAR] = str(train_seed)
            env[RUN_ROOT_ENV_VAR] = str(run_root)
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=current_dir,
                env=env,
                check=True,
            )
            global_finished += int(script_progress["remaining"])

    print(f"[DONE] Accounted for {global_total} training cases in {run_root}.")


if __name__ == "__main__":
    run_all_training_scripts()
