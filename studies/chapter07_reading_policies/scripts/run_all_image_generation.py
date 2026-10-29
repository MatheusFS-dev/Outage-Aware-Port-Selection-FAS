"""Run all KMS reading-pattern image-generation scripts."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import subprocess
import sys
from pathlib import Path


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/run_all_image_generation.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
IMAGE_SCRIPT_DIR = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation'

IMAGE_GENERATION_SCRIPTS: tuple[Path, ...] = (
    IMAGE_SCRIPT_DIR / "plot_best_policy_heatmap.py",
    IMAGE_SCRIPT_DIR / "plot_ci_analysis.py",
    IMAGE_SCRIPT_DIR / "plot_forest_difference.py",
    IMAGE_SCRIPT_DIR / "plot_mean_regret_vs_budget.py",
    IMAGE_SCRIPT_DIR / "plot_ports_histograms.py",
    IMAGE_SCRIPT_DIR / "plot_regime_op_vs_budget.py",
)


def run_scripts_in_order(script_paths: tuple[Path, ...]) -> None:
    """Run Python image-generation scripts sequentially.

    Args:
        script_paths (tuple[Path, ...]): Ordered absolute or relative Python
            script paths to execute. Each path must exist before execution
            starts. Scripts run with the repository root as their working
            directory so they can find saved analysis outputs and shared
            reading-pattern helpers through their existing path bootstrap code.

    Returns:
        None: The function streams child process output directly to the
            terminal and stops after the final image-generation script
            succeeds.

    Raises:
        FileNotFoundError: If any script in ``script_paths`` does not exist.
        subprocess.CalledProcessError: If any child script exits with a
            non-zero status. Later scripts are not run after a failure.

    Examples:
        >>> run_scripts_in_order((IMAGE_SCRIPT_DIR / "plot_best_policy_heatmap.py",))
    """
    for script_path in script_paths:
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

    for script_path in script_paths:
        # Keep plot generation sequential so output logs identify the script
        # that produced each figure set.
        print(f"\n=== Running {script_path.relative_to(REPO_ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    run_scripts_in_order(IMAGE_GENERATION_SCRIPTS)
