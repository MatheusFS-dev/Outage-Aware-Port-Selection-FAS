"""Generate confidence interval width analysis plots."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

import sys
CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/plot_ci_analysis.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.analyze_all_policies_kms import read_json_file

DEFAULT_OUTPUT_FORMATS = ("pdf", "png")


def list_threshold_analysis_dirs(analysis_root: Path) -> list[Path]:
    """List threshold-specific analysis folders that contain run configs.

    Args:
        analysis_root (Path): Root analysis directory expected to contain
            threshold-specific result folders.

    Returns:
        list[Path]: Sorted threshold analysis directories.

    Raises:
        OSError: If the directory scan fails.
    """
    return sorted(
        path for path in analysis_root.glob("threshold_*") if path.is_dir() and (path / "run_config.json").exists()
    )


def plot_ci_analysis_for_analysis_dir(analysis_dir: Path, output_base: Path) -> None:
    """Plot CI half-widths for one representative regime and threshold.

    Args:
        analysis_dir (Path): Threshold-specific analysis folder containing
            per-regime summary CSV files.
        output_base (Path): Output path without suffix. One figure is written
            for every configured format.

    Returns:
        None: Figure files are written to disk.

    Raises:
        FileNotFoundError: If expected analysis files are missing.
        OSError: If a figure cannot be written.
    """
    config = read_json_file(analysis_dir / "run_config.json")
    regime_labels = config["regime_labels"]
    target_ports = config["target_ports"]
    
    # We will plot the 95% CI half-width for a representative regime
    regime_label = "kappa0_mu1_m50"
    csv_path = analysis_dir / regime_label / "summary_by_budget.csv"
    if not csv_path.exists():
        print(f"Summary CSV for {regime_label} not found.")
        return
        
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    policy_styles = {
        "uni": {"label": "UNI", "color": "tab:blue", "marker": "o"},
        "mar": {"label": "MAR", "color": "tab:orange", "marker": "s"},
        "gc": {"label": "GC", "color": "tab:green", "marker": "^"},
        "suni": {"label": "SUNI", "color": "tab:purple", "marker": "v"},
        "swuni": {"label": "SWUNI", "color": "tab:red", "marker": "P"},
        "rag": {"label": "RAG", "color": "tab:brown", "marker": "X"},
    }
    _rnd_colors = ["#888888", "#555555", "#222222"]
    rnd_policy_styles = {
        f"rnd_{n}x": {"label": f"RND ({n}x)", "color": _rnd_colors[i], "marker": "D"}
        for i, n in enumerate((1, 3, 10))
    }

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    for policy_name, style in {**policy_styles, **rnd_policy_styles}.items():
        policy_rows = [r for r in rows if r["policy"] == policy_name]
        if not policy_rows:
            continue
        policy_rows.sort(key=lambda x: int(x["n_ports"]))
        
        x_vals = [int(r["n_ports"]) for r in policy_rows]
        y_vals = [float(r["ci95_half_width"]) for r in policy_rows]
        
        ls = "-" if not policy_name.startswith("rnd_") else "--"
        ax.plot(x_vals, y_vals, color=style["color"], marker=style["marker"], linestyle=ls, label=style["label"])
        
    ax.set_xlabel("Observed ports $N$")
    ax.set_ylabel("95% CI Half-width")
    ax.set_xticks(target_ports)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.0)
    
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in DEFAULT_OUTPUT_FORMATS:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    analysis_root = _COMPANION_ROOT / 'studies/chapter07_reading_policies/results/analysis_all_policies_kms'
    if not analysis_root.exists():
        print(f"Analysis directory not found: {analysis_root}")
        return

    threshold_dirs = list_threshold_analysis_dirs(analysis_root)
    if not threshold_dirs:
        print(f"No threshold analysis directories found under: {analysis_root}")
        return

    regime_label = "kappa0_mu1_m50"
    for analysis_dir in threshold_dirs:
        output_base = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/ci_analysis' / analysis_dir.name / f"ci_analysis_{regime_label}"
        plot_ci_analysis_for_analysis_dir(analysis_dir=analysis_dir, output_base=output_base)
    print("Saved CI analysis plot")

if __name__ == "__main__":
    main()
