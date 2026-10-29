"""Generate observed outage probability versus budget plots for all regimes."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sys
CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/plot_regime_op_vs_budget.py'
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
            folders named like ``threshold_0.37673008``.

    Returns:
        list[Path]: Sorted threshold analysis directories.

    Raises:
        OSError: If the analysis root cannot be scanned.
    """
    return sorted(
        path for path in analysis_root.glob("threshold_*") if path.is_dir() and (path / "run_config.json").exists()
    )

def plot_regime(regime_label: str, cross_regime_rows: list, target_ports: list, output_dir: Path):
    regime_rows = [row for row in cross_regime_rows if str(row["dataset"]) == regime_label]
    if not regime_rows:
        return

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

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    plotted_policies = []

    for policy_name, style in policy_styles.items():
        policy_rows = [row for row in regime_rows if str(row["policy"]) == policy_name]
        if not policy_rows:
            continue
        plotted_policies.append(policy_name)
        policy_rows = sorted(policy_rows, key=lambda x: int(x["n_ports"]))
        x_vals = [int(r["n_ports"]) for r in policy_rows]
        y_vals = [float(r["mean_observed_op"]) for r in policy_rows]
        y_errs = [float(r["ci95_half_width"]) for r in policy_rows]

        ax.errorbar(x_vals, y_vals, yerr=y_errs, color=style["color"], marker=style["marker"],
                    linestyle="-", linewidth=1.2, ecolor="#B24C63", elinewidth=1.4,
                    capsize=3.5, capthick=1.4, label="_nolegend_")

    uni_rows = sorted([r for r in regime_rows if str(r["policy"]) == "uni"], key=lambda x: int(x["n_ports"]))
    has_ideal = len(uni_rows) > 0
    if has_ideal:
        ideal_vals = [float(r["mean_ideal_op"]) for r in uni_rows]
        ax.plot([int(r["n_ports"]) for r in uni_rows], ideal_vals, color="black", linestyle="--", linewidth=1.0, label="_nolegend_")
        ax.set_ylim(bottom=0.0)

    for policy_name, style in rnd_policy_styles.items():
        policy_rows = sorted([r for r in regime_rows if str(r["policy"]) == policy_name], key=lambda x: int(x["n_ports"]))
        if not policy_rows:
            continue
        plotted_policies.append(policy_name)
        x_vals = [int(r["n_ports"]) for r in policy_rows]
        y_vals = [float(r["mean_observed_op"]) for r in policy_rows]
        y_errs = [float(r["ci95_half_width"]) for r in policy_rows]

        ax.errorbar(x_vals, y_vals, yerr=y_errs, color=style["color"], marker=style["marker"],
                    linestyle="--", linewidth=0.9, ecolor="#B24C63", elinewidth=1.4,
                    capsize=3.5, capthick=1.4, markersize=3.5, label="_nolegend_")

    ax.set_xlabel("Observed ports $N$")
    ax.set_ylabel("Outage probability")
    ax.set_xticks(target_ports)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    legend_handles = []
    if has_ideal:
        legend_handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Ideal OP"))

    for p in plotted_policies:
        style = policy_styles.get(p) or rnd_policy_styles.get(p)
        ls = "-" if p in policy_styles else "--"
        legend_handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"], linestyle=ls, linewidth=1.2, markersize=6.0, label=style["label"]))

    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=8.5)
    fig.tight_layout(pad=0.2)
    
    output_base = output_dir / f"op_vs_budget_{regime_label}"
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
        config = read_json_file(analysis_dir / "run_config.json")
        regime_labels = config["regime_labels"]
        target_ports = config["target_ports"]
        cross_regime_rows = read_csv_rows(analysis_dir / "cross_regime_summary.csv")

        output_dir = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/regime_op_plots' / analysis_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for regime_label in regime_labels:
            plot_regime(regime_label, cross_regime_rows, target_ports, output_dir)
    print("Saved op vs budget plots")

if __name__ == "__main__":
    main()
