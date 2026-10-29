"""Generate best policy heatmap figure."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

import sys
CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/plot_best_policy_heatmap.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter07_reading_policies.scripts.policy_mc_kms_common import KMS_REGIME_LABELS
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

def plot_best_policy_heatmap_for_analysis_dir(analysis_dir: Path, output_base: Path) -> None:
    """Plot the best-policy heatmap for one threshold analysis folder.

    Args:
        analysis_dir (Path): Threshold-specific analysis folder containing the
            best-policy CSV and run config.
        output_base (Path): Output path without suffix. One file is written for
            every configured figure format.

    Returns:
        None: Figure files are written to disk.

    Raises:
        FileNotFoundError: If expected analysis files are missing.
        OSError: If any figure cannot be written.
    """
    config = read_json_file(analysis_dir / "run_config.json")
    regime_labels = config["regime_labels"]
    target_ports = config["target_ports"]
    best_policy_rows = read_csv_rows(analysis_dir / "best_mary_regime_and_budget.csv")
    
    policy_to_value = {
        "uni": 0.0,
        "mar": 1.0,
        "gc": 2.0,
        "suni": 3.0,
        "swuni": 4.0,
        "rag": 5.0,
        "rnd_1x": 6.0,
        "rnd_3x": 6.0,
        "rnd_10x": 6.0,
    }
    
    best_mary_key = {}
    for row in best_policy_rows:
        best_mary_key[(str(row["dataset"]), int(row["n_ports"]))] = str(row["best_policy"])

    matrix = np.full((len(regime_labels), len(target_ports)), np.nan, dtype=np.float64)
    for regime_index, regime_label in enumerate(regime_labels):
        for budget_index, n_ports in enumerate(target_ports):
            policy_name = best_mary_key.get((regime_label, int(n_ports)))
            if policy_name is not None and policy_name in policy_to_value:
                matrix[regime_index, budget_index] = policy_to_value[policy_name]

    masked = np.ma.masked_invalid(matrix)

    cmap = ListedColormap([
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:red",
        "tab:brown",
        "#999999",
    ])
    cmap.set_bad(color="white")
    boundaries = np.arange(-0.5, 7.5, 1.0, dtype=np.float64)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(5.4, max(3.8, 0.22 * len(regime_labels) + 1.1)))
    x_edges = np.arange(len(target_ports) + 1, dtype=np.float64) - 0.5
    y_edges = np.arange(len(regime_labels) + 1, dtype=np.float64) - 0.5
    
    image = ax.pcolormesh(
        x_edges, y_edges, masked, cmap=cmap, norm=norm, shading="flat", linewidth=1.0, antialiased=False
    )
    
    ax.set_xlim(-0.5, len(target_ports) - 0.5)
    ax.set_ylim(len(regime_labels) - 0.5, -0.5)

    ax.set_xlabel("Observed ports $N$")
    ax.set_ylabel("Fading regime")

    ax.set_xticks(np.arange(len(target_ports)))
    ax.set_xticklabels([str(value) for value in target_ports])
    ax.set_yticks(np.arange(len(regime_labels)))
    ax.set_yticklabels([format_regime_label_mathtext(l) for l in regime_labels])

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=np.arange(7, dtype=np.float64),
        boundaries=boundaries,
        orientation="horizontal",
        fraction=0.08,
        pad=0.14,
        drawedges=True,
    )
    colorbar.ax.set_xticklabels(["UNI", "MAR", "GC", "SUNI", "SWUNI", "RAG", "RND"])

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
        output_base = _COMPANION_ROOT / 'studies/chapter07_reading_policies/scripts/image_generation/best_policy_heatmap' / analysis_dir.name / "best_policy_heatmap"
        plot_best_policy_heatmap_for_analysis_dir(analysis_dir=analysis_dir, output_base=output_base)
    print("Saved heatmap")

if __name__ == "__main__":
    main()
