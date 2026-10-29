"""Shared plotting and output helpers for NAS analysis scripts."""

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
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/common.py'
DEFAULT_OUTPUT_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/outputs/nas_analysis'
DEFAULT_PLOT_BIN_COUNT = 7
PALETTE = {
    "background": "#ffffff",
    "grid": "#d7dbe2",
    "axis": "#20242b",
    "muted": "#9aa4b2",
    "primary": "#1f4e79",
    "secondary": "#5b6c5d",
    "highlight": "#b54708",
    "selection": "#7a1f3d",
    "reference": "#3f3f46",
    "box_fill": "#d9e6f2",
}


def build_output_directories(output_root_dir: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Path]:
    """Create and return the standard NAS analysis output directories.

    Args:
        output_root_dir (Path): Root directory for NAS analysis outputs. The
            function always creates the root plus the required ``logs``,
            ``tables``, ``plots/png``, and ``plots/csv`` subdirectories. When a
            caller passes a different root, the output layout stays the same
            and only the base location changes.

    Returns:
        dict[str, Path]: Mapping containing the created directories under the
            keys ``root``, ``logs``, ``tables``, ``plot_png``, and
            ``plot_csv``.

    Raises:
        OSError: If any directory cannot be created.

    Examples:
        >>> directories = build_output_directories(DEFAULT_OUTPUT_ROOT)
        >>> sorted(directories)
        ['logs', 'plot_csv', 'plot_png', 'root', 'tables']
    """
    directories = {
        "root": output_root_dir,
        "logs": output_root_dir / "logs",
        "tables": output_root_dir / "tables",
        "plot_png": output_root_dir / "plots" / "png",
        "plot_csv": output_root_dir / "plots" / "csv",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def apply_publication_style() -> None:
    """Apply the preview-matched Matplotlib style used by NAS plots.

    Args:
        None.

    Returns:
        None: Updates Matplotlib ``rcParams`` in-place for the current Python
        process so all subsequent figures use the same journal-style theme.

    Raises:
        None.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["background"],
            "axes.facecolor": PALETTE["background"],
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["axis"],
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "axes.titleweight": "semibold",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 16,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "savefig.facecolor": PALETTE["background"],
        }
    )


def create_figure(width: float = 6.4, height: float = 4.4) -> tuple[plt.Figure, plt.Axes]:
    """Create one styled Matplotlib figure for a single NAS plot panel.

    Args:
        width (float): Figure width in inches. Wider values are appropriate for
            plots with denser categorical labels or longer legends.
        height (float): Figure height in inches. Slightly taller figures help
            when categorical plots or legends need more vertical space.

    Returns:
        tuple[plt.Figure, plt.Axes]: One figure and one axes object ready for
        drawing a publication-style plot.

    Raises:
        None.
    """
    apply_publication_style()
    return plt.subplots(figsize=(width, height), constrained_layout=True)


def create_two_panel_figure(
    width: float = 10.2,
    height: float = 4.2,
) -> tuple[plt.Figure, np.ndarray]:
    """Create one styled two-panel Matplotlib figure.

    Args:
        width (float): Figure width in inches. The default is intentionally
            wide so both panels remain legible side by side.
        height (float): Figure height in inches. The default keeps the figure
            compact enough to resemble a journal multi-panel layout.

    Returns:
        tuple[plt.Figure, np.ndarray]: Figure plus a two-element axes array for
        side-by-side NAS objective plots.

    Raises:
        None.
    """
    apply_publication_style()
    return plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)


def style_axes(
    axes: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    grid_axis: str = "both",
) -> None:
    """Apply the preview-matched axis styling to one axes object.

    Args:
        axes (plt.Axes): Axes to style in-place.
        xlabel (str): X-axis label shown below the plot.
        ylabel (str): Y-axis label shown to the left of the plot.
        title (str | None): Optional plot title. When ``None``, the plot stays
            untitled so it can be embedded as an in-paper panel.
        grid_axis (str): Grid direction. Accepted values are ``"x"``,
            ``"y"``, and ``"both"``. Histograms and boxplots typically use
            ``"y"`` so the category axis remains visually quiet.

    Returns:
        None: Mutates the provided axes object.

    Raises:
        None.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title, pad=10.0)
    axes.grid(True, axis=grid_axis, linestyle="--", dashes=(2.0, 2.4))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.tick_params(length=4.0, width=0.8)


def add_reference_legend(axes: plt.Axes, location: str = "best") -> None:
    """Add a legend only when the plotted artists expose labels.

    Args:
        axes (plt.Axes): Axes from which labeled artists are collected.
        location (str): Matplotlib legend location string. ``"best"`` lets
            Matplotlib place the legend automatically, while fixed corners can
            be used when overlap is predictable.

    Returns:
        None: Adds a legend in-place when there is at least one labeled artist.

    Raises:
        None.
    """
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels, loc=location)


def add_boxplot_reference_legend(axes: plt.Axes, location: str = "upper right") -> None:
    """Add the explanatory legend used by categorical NAS boxplots.

    Args:
        axes (plt.Axes): Axes containing one styled NAS hyperparameter boxplot.
        location (str): Matplotlib legend location string. A fixed corner keeps
            the visual explanation stable across regenerated figures.

    Returns:
        None: Adds an in-place legend that explains the box fill, median line,
        and outlier markers used in the plot.

    Raises:
        None.
    """
    handles = [
        Patch(
            facecolor=PALETTE["box_fill"],
            edgecolor=PALETTE["axis"],
            linewidth=0.9,
            label="Interquartile range",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["highlight"],
            linewidth=1.4,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=4.2,
            markerfacecolor=PALETTE["muted"],
            markeredgecolor=PALETTE["muted"],
            alpha=0.7,
            label="Outlier",
        ),
    ]
    axes.legend(handles=handles, loc=location)


def draw_publication_boxplot(
    axes: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
) -> None:
    """Draw one styled boxplot that matches the preview scripts.

    Args:
        axes (plt.Axes): Axes on which to render the boxplot.
        groups (list[np.ndarray]): Ordered grouped values where each array
            contains the trial-level metric values for one category. Each group
            contributes one box in the rendered figure.
        labels (list[str]): Category labels displayed on the x-axis. The label
            order must match ``groups`` exactly or the resulting plot would
            mislabel the distributions.

    Returns:
        None: Draws the boxplot in-place on ``axes``.

    Raises:
        ValueError: If the numbers of groups and labels differ.

    Examples:
        >>> axis_figure, axis = create_figure()
        >>> draw_publication_boxplot(axis, [np.array([1.0, 2.0])], ['1'])
    """
    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length.")

    axes.boxplot(
        groups,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": PALETTE["highlight"], "linewidth": 1.4},
        whiskerprops={"color": PALETTE["axis"], "linewidth": 0.9},
        capprops={"color": PALETTE["axis"], "linewidth": 0.9},
        boxprops={"facecolor": PALETTE["box_fill"], "edgecolor": PALETTE["axis"], "linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": PALETTE["muted"],
            "markeredgecolor": PALETTE["muted"],
            "alpha": 0.7,
        },
    )
    axes.set_xticks(np.arange(1, len(labels) + 1, dtype=int))
    axes.set_xticklabels(labels)


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Write one DataFrame to CSV after ensuring the parent directory exists.

    Args:
        dataframe (pd.DataFrame): Table to serialize exactly as CSV. The caller
            is responsible for preparing final column order and values before
            writing.
        output_path (Path): Destination CSV path. The parent directory is
            created automatically when it does not already exist.

    Returns:
        Path: The written CSV path so callers can include it in logs.

    Raises:
        OSError: If the parent directory cannot be created or the CSV cannot be
            written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_figure(figure: plt.Figure, output_path: Path) -> Path:
    """Write one Matplotlib figure to PNG and then release the figure object.

    Args:
        figure (plt.Figure): Figure to write. The caller should finish all
        plotting before passing it here because the figure is closed after the
        PNG is saved.
        output_path (Path): Destination PNG path. The parent directory is
            created automatically if it does not exist.

    Returns:
        Path: The written PNG path so callers can include it in logs.

    Raises:
        OSError: If the PNG cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path

