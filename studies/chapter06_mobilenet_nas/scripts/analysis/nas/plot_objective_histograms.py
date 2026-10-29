"""Generate the valid-trial NAS objective histograms and reconstruction CSV.

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

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_objective_histograms.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLOT_BIN_COUNT,
    PALETTE,
    create_two_panel_figure,
    save_dataframe,
    save_figure,
    style_axes,
)
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import NasStudyContext, load_nas_study_context


STUDY_RUN_DIR = Path("./runs")


def write_objective_histograms(context: NasStudyContext) -> tuple[Path, Path]:
    """Render and save the valid-trial histograms for both NAS objectives.

    Args:
        context (NasStudyContext): Prepared NAS study context containing the
            filtered valid-trial dataframe and the KiB-converted RAM column.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the two-panel
        histogram figure and the reconstruction table.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    trials_df = context.filtered_trials_df.sort_values("trial_id", kind="mergesort")
    figure, axes_array = create_two_panel_figure()
    loss_axes, ram_axes = axes_array

    loss_weights = [1.0 / len(trials_df)] * len(trials_df)
    ram_weights = [1.0 / len(trials_df)] * len(trials_df)
    loss_axes.hist(
        trials_df["best_qt_val_metric"],
        bins=DEFAULT_PLOT_BIN_COUNT,
        weights=loss_weights,
        color=PALETTE["primary"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7,
    )
    ram_axes.hist(
        trials_df["trained_ram_peak_kib"],
        bins=DEFAULT_PLOT_BIN_COUNT,
        weights=ram_weights,
        color=PALETTE["secondary"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7,
    )
    style_axes(
        loss_axes,
        xlabel=f"INT8 {context.objective_label}",
        ylabel="Normalized frequency",
        title="INT8 validation objective across valid trials",
        grid_axis="y",
    )
    style_axes(
        ram_axes,
        xlabel="Trained peak RAM (KiB)",
        ylabel="Normalized frequency",
        title="RAM objective across valid trials",
        grid_axis="y",
    )

    csv_rows = []
    for objective_name, series in (
        ("best_qt_val_metric", trials_df["best_qt_val_metric"]),
        ("trained_ram_peak_kib", trials_df["trained_ram_peak_kib"]),
    ):
        for trial_id, objective_value in zip(trials_df["trial_id"], series, strict=True):
            csv_rows.append(
                {
                    "trial_id": int(trial_id),
                    "objective_name": objective_name,
                    "objective_value": float(objective_value),
                    "histogram_source": "valid_trials",
                    "bin_count": int(DEFAULT_PLOT_BIN_COUNT),
                }
            )
    csv_df = pd.DataFrame(csv_rows)

    png_path = save_figure(figure, context.directories["plot_png"] / "objective_histogram_valid_trials.png")
    csv_path = save_dataframe(csv_df, context.directories["plot_csv"] / "objective_histogram_valid_trials.csv")
    return png_path, csv_path


def run_objective_histogram_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Load one NAS study and generate only the valid-trial histogram outputs.

    Args:
        study_run_dir (Path): NAS study output directory containing the
            ``optuna_study`` artifacts required by the shared loader.
        output_root_dir (Path): Root output directory for the generated PNG and
            CSV.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the figure.

    Raises:
        FileNotFoundError: If the study inputs are missing.
        ValueError: If the study data cannot satisfy the filtering contract.
        OSError: If the outputs cannot be written.
    """
    return write_objective_histograms(
        load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    )


if __name__ == "__main__":
    run_objective_histogram_analysis(study_run_dir=STUDY_RUN_DIR)
