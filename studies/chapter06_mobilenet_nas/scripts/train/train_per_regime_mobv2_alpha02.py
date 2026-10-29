"""Train the validated MobileNetV2 safe selector per 60/20/20 regime."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from per_regime_training import run_per_regime_campaign


if __name__ == "__main__":
    run_per_regime_campaign("mobv2_alpha02")
