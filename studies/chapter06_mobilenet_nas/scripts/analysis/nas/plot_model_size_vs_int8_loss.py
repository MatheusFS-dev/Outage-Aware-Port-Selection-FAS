"""Generate the model-size versus INT8-loss plot and reconstruction CSV.

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

import pandas as pd

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_model_size_vs_int8_loss.py'
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


def write_model_size_vs_int8_loss(context: NasStudyContext) -> tuple[Path, Path]:
    """Render and save the model-size versus INT8-loss plot for one study.

    Args:
        context (NasStudyContext): Prepared NAS study context with filtered
            valid trials and derived KiB columns.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the plot and
        the exact point series used to render it.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    figure, axes = create_figure()
    trials_df = context.filtered_trials_df.sort_values("trial_id", kind="mergesort")

    axes.scatter(
        trials_df["qt_model_size_kib"],
        trials_df["best_qt_val_metric"],
        s=42,
        color=PALETTE["primary"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
        label="Valid trials",
    )
    style_axes(
        axes,
        xlabel="INT8 model size (KiB)",
        ylabel=f"INT8 {context.objective_label}",
        title="Model size versus INT8 validation objective",
    )
    add_reference_legend(axes, location="upper right")

    csv_df = trials_df.loc[:, ["trial_id", "qt_model_size_kib", "best_qt_val_metric", "is_top_k", "is_selected_trial"]]
    csv_df = csv_df.rename(columns={"qt_model_size_kib": "x", "best_qt_val_metric": "y", "is_selected_trial": "is_selected"})

    png_path = save_figure(figure, context.directories["plot_png"] / "model_size_vs_int8_loss.png")
    csv_path = save_dataframe(csv_df, context.directories["plot_csv"] / "model_size_vs_int8_loss.csv")
    return png_path, csv_path


def run_model_size_vs_int8_loss_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Load one NAS study and generate only the model-size plot outputs.

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
    return write_model_size_vs_int8_loss(
        load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    )


if __name__ == "__main__":
    run_model_size_vs_int8_loss_analysis(study_run_dir=STUDY_RUN_DIR)
