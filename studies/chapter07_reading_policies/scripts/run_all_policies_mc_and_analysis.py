"""Run all KMS reading-pattern policy Monte Carlo scripts, then analysis."""

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


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/run_all_policies_mc_and_analysis.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
SCRIPT_DIR = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts'

POLICY_AND_ANALYSIS_SCRIPTS: tuple[Path, ...] = (
    SCRIPT_DIR / "policy_uni_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_mar_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_gc_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_suni_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_swuni_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_rag_kms_per_regime_mc.py",
    SCRIPT_DIR / "policy_rnd_kms_per_regime_mc.py",
    SCRIPT_DIR / "analyze_all_policies_kms.py",
)


def run_scripts_in_order(script_paths: tuple[Path, ...]) -> None:
    """Run Python scripts sequentially with the current interpreter.

    Args:
        script_paths (tuple[Path, ...]): Ordered absolute or relative Python
            script paths to execute. Each path must exist before execution
            starts. Scripts run with the repository root as their working
            directory, which matches the existing reading-pattern path
            bootstrap assumptions.

    Returns:
        None: The function streams child process output directly to the
            terminal and stops after the final script succeeds.

    Raises:
        FileNotFoundError: If any script in ``script_paths`` does not exist.
        subprocess.CalledProcessError: If any child script exits with a
            non-zero status. Later scripts are not run after a failure.

    Examples:
        >>> run_scripts_in_order((SCRIPT_DIR / "policy_uni_kms_per_regime_mc.py",))
    """
    for script_path in script_paths:
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

    for script_path in script_paths:
        # Run policies sequentially because the analysis step depends on all
        # policy result folders being complete and configuration-compatible.
        print(f"\n=== Running {script_path.relative_to(REPO_ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    run_scripts_in_order(POLICY_AND_ANALYSIS_SCRIPTS)
