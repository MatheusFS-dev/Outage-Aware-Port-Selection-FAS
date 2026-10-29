"""Generate forest plot of paired differences."""

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

import sys
CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/plot_forest_difference.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.analyze_all_policies_kms import read_json_file, read_csv_rows

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

def format_regime_label_mathtext(regime_label: str) -> str:
    parts = regime_label.split("_")
    kappa = int(parts[0][5:])
    mu = int(parts[1][2:])
    m = int(parts[2][1:])
    kappa_part = r"\kappa \to 0" if kappa == 0 else rf"\kappa = {kappa}"
    if m == 0:
        m_part = r"m \to 0"
    elif m >= 50:
        m_part = r"m \to \infty"
    else:
        m_part = rf"m = {m}"
    mu_part = rf"\mu = {mu}"
    return rf"${kappa_part},\, {mu_part},\, {m_part}$"

def plot_forest_difference_for_analysis_dir(analysis_dir: Path, output_base: Path) -> None:
    """Plot paired policy differences for one threshold analysis folder.

    Args:
        analysis_dir (Path): Threshold-specific analysis folder containing
            regime summary CSV files.
        output_base (Path): Output path without suffix. One file is written for
            each configured figure format.

    Returns:
        None: Figure files are written to disk.

    Raises:
        FileNotFoundError: If expected analysis files are missing.
        OSError: If a figure cannot be written.
    """
    config = read_json_file(analysis_dir / "run_config.json")
    regime_labels = config["regime_labels"]
    target_ports = config["target_ports"]
    
    # Plot the paired policy-minus-UNI difference at the largest budget.
    # Negative values mean the policy has lower observed outage than UNI.
    budget = target_ports[-1]
    
    fig, ax = plt.subplots(figsize=(6.0, len(regime_labels) * 0.4 + 1.0))
    y_positions = np.arange(len(regime_labels))
    
    colors = {
        "mar": "tab:orange",
        "gc": "tab:green",
        "suni": "tab:purple",
        "swuni": "tab:red",
        "rag": "tab:brown",
        "rnd_10x": "#333333",
    }
    markers = {"mar": "s", "gc": "^", "suni": "v", "swuni": "P", "rag": "X", "rnd_10x": "h"}
    labels = {
        "mar": "MAR",
        "gc": "GC",
        "suni": "SUNI",
        "swuni": "SWUNI",
        "rag": "RAG",
        "rnd_10x": "RND (10x)",
    }
    
    offsets = {"mar": -0.3, "gc": -0.2, "suni": -0.1, "swuni": 0.0, "rag": 0.1, "rnd_10x": 0.2}

    for regime_index, regime_label in enumerate(regime_labels):
        csv_path = analysis_dir / regime_label / "summary_by_budget.csv"
        if not csv_path.exists():
            continue
        
        rows = read_csv_rows(csv_path)
            
        for policy in ["mar", "gc", "suni", "swuni", "rag", "rnd_10x"]:
            row = next((r for r in rows if r["policy"] == policy and int(r["n_ports"]) == budget), None)
            if not row:
                continue
                
            mean_delta = float(row["mean_delta_vs_uni"])
            ci_half = float(row["delta_ci95_half_width"])
            
            y_pos = y_positions[regime_index] + offsets[policy]
            ax.errorbar(mean_delta, y_pos, xerr=ci_half, fmt=markers[policy], color=colors[policy], ecolor="#B24C63", elinewidth=1.4, capsize=3.5, capthick=1.4, label=labels[policy] if regime_index == 0 else "_nolegend_")
            
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([format_regime_label_mathtext(r) for r in regime_labels])
    ax.set_xlabel(f"Observed OP Difference vs UNI (Budget N={budget})")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    
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

    for analysis_dir in threshold_dirs:
        output_base = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/forest_difference_plot' / analysis_dir.name / "forest_difference_plot"
        plot_forest_difference_for_analysis_dir(analysis_dir=analysis_dir, output_base=output_base)
    print("Saved forest plot")

if __name__ == "__main__":
    main()
