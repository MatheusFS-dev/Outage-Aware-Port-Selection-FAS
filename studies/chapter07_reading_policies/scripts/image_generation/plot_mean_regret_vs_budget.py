"""Generate mean regret versus budget plot across regimes."""

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
CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/plot_mean_regret_vs_budget.py'
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
        list[Path]: Sorted folders such as ``threshold_0.37673008``.

    Raises:
        OSError: If the directory scan fails.
    """
    return sorted(
        path for path in analysis_root.glob("threshold_*") if path.is_dir() and (path / "run_config.json").exists()
    )


def plot_mean_regret_for_analysis_dir(analysis_dir: Path, output_base: Path) -> None:
    """Plot mean regret versus budget for one threshold analysis folder.

    Args:
        analysis_dir (Path): Threshold-specific analysis folder containing
            ``run_config.json`` and ``cross_regime_summary.csv``.
        output_base (Path): Output path without suffix. One file is written for
            every configured format.

    Returns:
        None: Figures are written to disk.

    Raises:
        FileNotFoundError: If expected analysis files are missing.
        OSError: If a figure cannot be written.
    """
    config = read_json_file(analysis_dir / "run_config.json")
    regime_labels = config["regime_labels"]
    target_ports = config["target_ports"]
    cross_regime_rows = read_csv_rows(analysis_dir / "cross_regime_summary.csv")

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

    regimes_set = {str(label) for label in regime_labels}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for policy_name, style in policy_styles.items():
        means, stds = [], []
        for n_ports in target_ports:
            values = [
                float(r["mean_gap_to_ideal"]) for r in cross_regime_rows 
                if str(r.get("policy")) == policy_name and int(r.get("n_ports")) == n_ports and str(r.get("dataset")) in regimes_set
            ]
            if not values:
                continue
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            
        if len(means) == len(target_ports):
            ax.errorbar(np.array(target_ports), np.array(means), yerr=np.array(stds), color=style["color"], marker=style["marker"], linestyle="-", linewidth=1.2, ecolor="#B24C63", elinewidth=1.4, capsize=3.5, capthick=1.4, label=style["label"])

    for policy_name, style in rnd_policy_styles.items():
        means, stds = [], []
        for n_ports in target_ports:
            values = [
                float(r["mean_gap_to_ideal"]) for r in cross_regime_rows 
                if str(r.get("policy")) == policy_name and int(r.get("n_ports")) == n_ports and str(r.get("dataset")) in regimes_set
            ]
            if not values:
                break
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            
        if len(means) == len(target_ports):
            ax.errorbar(np.array(target_ports), np.array(means), yerr=np.array(stds), color=style["color"], marker=style["marker"], linestyle="--", linewidth=0.9, ecolor="#B24C63", elinewidth=1.4, capsize=3.5, capthick=1.4, markersize=3.5, label=style["label"])

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Observed ports $N$")
    ax.set_ylabel("Regret to ideal OP")
    ax.set_xticks(target_ports)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", frameon=False, fontsize=8.5)

    fig.tight_layout(pad=0.2)
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
        output_base = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/mean_regret_vs_budget' / analysis_dir.name / "mean_regret_vs_budget"
        plot_mean_regret_for_analysis_dir(analysis_dir=analysis_dir, output_base=output_base)
    print("Saved mean regret plot")

if __name__ == "__main__":
    main()
