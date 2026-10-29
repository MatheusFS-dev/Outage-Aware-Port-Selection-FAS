"""Compare inference time between policies."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import time
import sys
from pathlib import Path
import numpy as np

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/benchmark_policy_construction_time.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import (
    build_uni_indices,
    build_mar_order,
    build_gc_order,
    build_rag_order,
    build_suni_indices,
    build_swuni_indices,
    format_threshold_folder,
    load_kms_regime,
    THRESHOLD,
    threshold_to_binary,
)


def measure_inference_time():
    regime_label = "kappa0_mu1_m50"
    print(f"Loading regime {regime_label} for inference benchmarking...")
    
    # We will measure the time it takes to select 7 ports (max budget)
    n_ports = 7
    total_ports = 100
    
    # UNI
    start = time.perf_counter()
    for _ in range(100):
        _ = build_uni_indices(total_ports, n_ports)
    time_uni = (time.perf_counter() - start) / 100
    print(f"UNI: {time_uni * 1e6:.2f} microseconds per selection")
    
    # RND
    rng = np.random.default_rng(42)
    start = time.perf_counter()
    for _ in range(100):
        _ = rng.choice(total_ports, size=n_ports, replace=False)
    time_rnd = (time.perf_counter() - start) / 100
    print(f"RND: {time_rnd * 1e6:.2f} microseconds per selection")
    
    regime_data = load_kms_regime(regime_label, common_path=None)
    out_dir = _COMPANION_ROOT / 'studies/chapter07_reading_policies/results/inference_time'
    out_dir.mkdir(parents=True, exist_ok=True)

    for threshold in THRESHOLD:
        # MAR requires training data to get exceedance scores. The
        # inference benchmark intentionally includes that thresholded ranking.
        binary = threshold_to_binary(regime_data[:1000], threshold=float(threshold), snr_linear=1.0)
        means = np.mean(regime_data[:1000], axis=0)

        start = time.perf_counter()
        for _ in range(100):
            order, scores = build_mar_order(binary, means)
            _ = order[:n_ports]
        time_mar = (time.perf_counter() - start) / 100
        print(f"MAR at threshold {float(threshold):.8f}: {time_mar * 1e6:.2f} microseconds per selection")

        start = time.perf_counter()
        for _ in range(10):
            _ = build_suni_indices(binary, means, n_ports)
        time_suni = (time.perf_counter() - start) / 10
        print(f"SUNI at threshold {float(threshold):.8f}: {time_suni * 1e6:.2f} microseconds per selection")

        start = time.perf_counter()
        for _ in range(10):
            _ = build_swuni_indices(binary, means, n_ports)
        time_swuni = (time.perf_counter() - start) / 10
        print(f"SWUNI at threshold {float(threshold):.8f}: {time_swuni * 1e6:.2f} microseconds per selection")

        start = time.perf_counter()
        for _ in range(10):
            order = build_gc_order(binary, means, scores, n_ports)
        time_gc = (time.perf_counter() - start) / 10
        print(f"GC at threshold {float(threshold):.8f}: {time_gc * 1e6:.2f} microseconds per selection")

        start = time.perf_counter()
        for _ in range(10):
            order = build_rag_order(binary, means, scores, n_ports, redundancy_weight=0.1)
        time_rag = (time.perf_counter() - start) / 10
        print(f"RAG at threshold {float(threshold):.8f}: {time_rag * 1e6:.2f} microseconds per selection")

        threshold_dir = out_dir / format_threshold_folder(float(threshold))
        threshold_dir.mkdir(parents=True, exist_ok=True)
        output_path = threshold_dir / "inference_times.txt"
        with open(output_path, "w") as f:
            f.write(f"Threshold: {float(threshold):.8f}\n")
            f.write(f"UNI: {time_uni * 1e6:.2f} us\n")
            f.write(f"RND: {time_rnd * 1e6:.2f} us\n")
            f.write(f"MAR: {time_mar * 1e6:.2f} us\n")
            f.write(f"GC: {time_gc * 1e6:.2f} us\n")
            f.write(f"SUNI: {time_suni * 1e6:.2f} us\n")
            f.write(f"SWUNI: {time_swuni * 1e6:.2f} us\n")
            f.write(f"RAG: {time_rag * 1e6:.2f} us\n")
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    measure_inference_time()
