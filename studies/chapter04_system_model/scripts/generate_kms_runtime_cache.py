"""Generate the derived KMS runtime cache used by Python examples.

This script converts the checked-in MATLAB ``SNR_events`` files into an
uncompressed ``float32`` per-regime ``.npy`` cache under ``src/data``. The
runtime cache is meant for faster Python-side loading while the original
MATLAB files remain the source of truth.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import sys
from pathlib import Path


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter04_system_model/scripts/generate_kms_runtime_cache.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from utils.data import (
    DEFAULT_DATASET_ROOT_PATH,
    DEFAULT_KMS_RUNTIME_CACHE_PATH,
    generate_kms_runtime_cache,
)
from utils.constants import BLUE, GREEN, RED, RESET


COMMON_DATASET_PATH = DEFAULT_DATASET_ROOT_PATH
CACHE_PATH = DEFAULT_KMS_RUNTIME_CACHE_PATH
FORCE_REBUILD = False


def run_generate_kms_runtime_cache() -> dict[str, object]:
    """Generate the KMS runtime cache and print a short summary.

    Args:
        None.

    Returns:
        dict[str, object]: Summary dictionary with keys ``cache_path`` and
            ``written_files``. ``cache_path`` is the destination directory used
            for the cache and ``written_files`` is a tuple of the files created
            or overwritten during this run.

    Raises:
        FileNotFoundError: If the source MATLAB dataset files are missing.
        KeyError: If a source MATLAB file does not contain ``SNR_events``.
        ValueError: If a source dataset is not a 2D matrix.
        OSError: If cache files cannot be written.

    Examples:
        >>> callable(run_generate_kms_runtime_cache)
        True
    """
    print(f"{BLUE}Starting KMS runtime cache generation...{RESET}")
    print(f"{BLUE}Source dataset path: {COMMON_DATASET_PATH}{RESET}")
    print(f"{BLUE}Runtime cache path: {CACHE_PATH}{RESET}")
    print(f"{BLUE}Force rebuild: {FORCE_REBUILD}{RESET}")

    try:
        written_files = generate_kms_runtime_cache(
            common_path=COMMON_DATASET_PATH,
            cache_path=CACHE_PATH,
            force_rebuild=FORCE_REBUILD,
            verbose=True,
        )
    except Exception as exc:
        print(f"{RED}KMS runtime cache generation failed: {exc}{RESET}")
        raise

    print(f"{GREEN}KMS runtime cache generation finished.{RESET}")
    print(f"{GREEN}Cache path: {CACHE_PATH}{RESET}")
    print(f"{GREEN}Files written this run: {len(written_files)}{RESET}")

    return {
        "cache_path": Path(CACHE_PATH),
        "written_files": written_files,
    }


if __name__ == "__main__":
    run_generate_kms_runtime_cache()
