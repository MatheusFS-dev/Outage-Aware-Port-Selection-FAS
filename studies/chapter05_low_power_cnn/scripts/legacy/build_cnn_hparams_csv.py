from pathlib import Path
_CHAPTER = Path(__file__).resolve().parents[2]

import csv
import glob
import os
import re
from typing import Dict, List


def parse_parameters_block(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    params_started = False
    params: Dict[str, str] = {}
    for line in lines:
        if not params_started:
            if line.strip() == "Parameters:":
                params_started = True
            continue
        # stop when we hit a non-indented section or an empty line following params
        if line.strip() == "":
            break
        if not line.startswith(" ") and not line.startswith("\t"):
            break
        # expected format: two spaces then key: value
        stripped = line.strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            params[key.strip()] = value.strip()
    return params


def main() -> None:
    pattern = os.path.join(
        str(_CHAPTER), "results", "models", "cnn", "optuna_study_*_ports", "args", "top_1_trial.txt"
    )
    files = sorted(glob.glob(pattern))
    rows: List[Dict[str, str]] = []
    all_keys = set()
    for fpath in files:
        m = re.search(r"optuna_study_(\d+)_ports", fpath)
        if not m:
            continue
        ports = m.group(1)
        params = parse_parameters_block(fpath)
        row: Dict[str, str] = {"ports": ports}
        row.update(params)
        rows.append(row)
        all_keys.update(params.keys())

    # Desired column order
    preferred_prefix_order = [
        "scaler",
        "num_layers",
    ]

    # cnn1d_* keys sorted by base key then numeric suffix
    def cnn_sort_key(k: str):
        # Split into name and optional numeric suffix
        m = re.match(r"(cnn1d_[a-z_]+)_(\d+)$", k)
        if m:
            base, idx = m.group(1), int(m.group(2))
            return (base, idx)
        return (k, -1)

    cnn_keys = sorted([k for k in all_keys if k.startswith("cnn1d_")], key=cnn_sort_key)

    tail_order = [
        "num_dense_layers",
        "optimizer",
        "learning_rate",
        "batch_size",
    ]

    # Remaining keys (rare/unexpected) not covered by the above
    covered = set(preferred_prefix_order) | set(cnn_keys) | set(tail_order)
    remaining = sorted([k for k in all_keys if k not in covered])

    header = ["ports"] + preferred_prefix_order + cnn_keys + tail_order + remaining

    # Sort rows by integer ports ascending
    rows.sort(key=lambda r: int(r["ports"]))

    out_dir = os.path.join(str(_CHAPTER), "results", "models", "cnn")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hparams_top1_by_ports.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(out_path)


if __name__ == "__main__":
    main()

