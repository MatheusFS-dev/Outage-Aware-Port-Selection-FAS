
# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))
from pathlib import Path

MODEL_PATH = Path("model.tflite")
HEADER_PATH = Path("model_data.h")
SOURCE_PATH = Path("model_data.cc")
BYTES_PER_LINE = 12

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

model_bytes = MODEL_PATH.read_bytes()
byte_literals = [f"0x{value:02x}" for value in model_bytes]

lines = []
for start in range(0, len(byte_literals), BYTES_PER_LINE):
    chunk = byte_literals[start : start + BYTES_PER_LINE]
    lines.append("    " + ", ".join(chunk) + ",")

HEADER_PATH.write_text(
    """#pragma once

#include <cstdint>

extern const unsigned char g_model[];
extern const int g_model_len;
""",
    encoding="utf-8",
)

SOURCE_PATH.write_text(
    """#include "model_data.h"

alignas(16) const unsigned char g_model[] = {
"""
    + "\n".join(lines)
    + f"""
}};

const int g_model_len = {len(model_bytes)};
""",
    encoding="utf-8",
)

print(f"Wrote {HEADER_PATH}")
print(f"Wrote {SOURCE_PATH}")
print(f"Model bytes: {len(model_bytes)}")
