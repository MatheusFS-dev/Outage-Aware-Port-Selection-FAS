"""Generate the NAS optimization-history plot and reconstruction CSV.

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

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_optimization_history.py'
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


def build_optimization_history_series(context: NasStudyContext) -> pd.DataFrame:
    """Build the best-so-far valid-trial loss series across all trial indices.

    Args:
        context (NasStudyContext): Prepared NAS study context containing both
            the raw all-trial dataframe and the filtered valid-trial subset.

    Returns:
        pd.DataFrame: One row per trial index with the columns ``trial_id``,
        ``best_so_far_qt_val_metric``, and ``is_valid_trial``.

    Raises:
        None.
    """
    raw_df = context.raw_trials_df.sort_values("trial_id", kind="mergesort").reset_index(drop=True)
    valid_trial_ids = set(context.filtered_trials_df["trial_id"].astype(int).tolist())
    best_so_far = np.nan
    history_rows = []

    # Carry forward the best valid value only when the current trial survives
    # the strict analysis filter so the curve matches the documented contract.
    for trial_row in raw_df.itertuples(index=False):
        is_valid_trial = int(trial_row.trial_id) in valid_trial_ids
        if is_valid_trial:
            current_loss = float(trial_row.best_qt_val_metric)
            best_so_far = current_loss if np.isnan(best_so_far) else min(best_so_far, current_loss)
        history_rows.append(
            {
                "trial_id": int(trial_row.trial_id),
                "best_so_far_qt_val_metric": best_so_far,
                "is_valid_trial": is_valid_trial,
            }
        )
    return pd.DataFrame(history_rows)


def write_optimization_history(context: NasStudyContext) -> tuple[Path, Path]:
    """Render and save the optimization-history plot for one NAS study.

    Args:
        context (NasStudyContext): Prepared NAS study context returned by the
            shared loader.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the plot and
        its reconstruction series.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    history_df = build_optimization_history_series(context)
    figure, axes = create_figure(width=6.8, height=4.4)

    axes.plot(
        history_df["trial_id"],
        history_df["best_so_far_qt_val_metric"],
        color=PALETTE["primary"],
        linewidth=1.7,
        label="Best so far (valid trials)",
    )
    style_axes(
        axes,
        xlabel="Trial number",
        ylabel=f"Best-so-far INT8 {context.objective_label}",
        title="Optimization history",
        grid_axis="y",
    )
    axes.set_xlim(1, int(history_df["trial_id"].max()))
    add_reference_legend(axes, location="upper right")

    png_path = save_figure(figure, context.directories["plot_png"] / "optimization_history.png")
    csv_path = save_dataframe(history_df, context.directories["plot_csv"] / "optimization_history.csv")
    return png_path, csv_path


def run_optimization_history_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Load one NAS study and generate only the optimization-history outputs.

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
    return write_optimization_history(
        load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    )


if __name__ == "__main__":
    run_optimization_history_analysis(study_run_dir=STUDY_RUN_DIR)
