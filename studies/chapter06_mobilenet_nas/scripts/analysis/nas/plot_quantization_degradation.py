"""Generate the quantization-degradation plot and reconstruction CSV.

Expected input:
    Analyze one NAS study output folder containing:
    - ``optuna_study/config.json``
    - ``optuna_study/args/top_*_trial.txt``
    - one supported Optuna study storage artifact inside ``optuna_study/``

Folder example:
    - ``STUDY_RUN_DIR`` points to one completed NAS study folder
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_quantization_degradation.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.common import (
    DEFAULT_OUTPUT_ROOT,
    PALETTE,
    add_reference_legend,
    create_figure,
    save_dataframe,
    save_figure,
    style_axes,
)
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import NasStudyContext, load_nas_study_context


STUDY_RUN_DIR = Path("./runs")


def write_quantization_degradation(context: NasStudyContext) -> tuple[Path, Path]:
    """Render and save the quantization-degradation plot for one study.

    Args:
        context (NasStudyContext): Prepared NAS study context containing the
            filtered valid-trial dataframe and the derived
            ``quantization_degradation`` column.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the plot and
        the exact plotted point series.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    trials_df = context.filtered_trials_df.sort_values("trial_id", kind="mergesort")
    figure, axes = create_figure()

    axes.scatter(
        trials_df["best_not_qt_val_metric"],
        trials_df["best_qt_val_metric"],
        s=42,
        color=PALETTE["primary"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Trials",
    )
    diagonal_min = float(np.min(trials_df["best_not_qt_val_metric"]))
    diagonal_max = float(np.max(trials_df["best_qt_val_metric"]))
    axes.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color=PALETTE["reference"],
        linestyle="--",
        linewidth=1.0,
        label="Identity",
        zorder=2,
    )
    style_axes(
        axes,
        xlabel=f"FP32 {context.objective_label}",
        ylabel=f"INT8 {context.objective_label}",
        title="Quantization degradation",
    )
    add_reference_legend(axes, location="upper left")

    csv_df = trials_df.loc[
        :,
        ["trial_id", "best_not_qt_val_metric", "best_qt_val_metric", "quantization_degradation", "is_top_k", "is_selected_trial"],
    ].rename(
        columns={
            "best_not_qt_val_metric": "x",
            "best_qt_val_metric": "y",
            "is_selected_trial": "is_selected",
        }
    )

    png_path = save_figure(figure, context.directories["plot_png"] / "quantization_degradation.png")
    csv_path = save_dataframe(csv_df, context.directories["plot_csv"] / "quantization_degradation.csv")
    return png_path, csv_path


def run_quantization_degradation_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Load one NAS study and generate only the quantization-degradation outputs.

    Args:
        study_run_dir (Path): NAS study output directory containing the
            ``optuna_study`` artifacts required by the shared loader.
        output_root_dir (Path): Root output directory for the generated PNG and
            CSV.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the plot.

    Raises:
        FileNotFoundError: If the study inputs are missing.
        ValueError: If the study data cannot satisfy the filtering contract.
        OSError: If the outputs cannot be written.
    """
    return write_quantization_degradation(
        load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    )


if __name__ == "__main__":
    run_quantization_degradation_analysis(study_run_dir=STUDY_RUN_DIR)
