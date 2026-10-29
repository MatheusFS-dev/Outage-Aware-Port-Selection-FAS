"""Generate the NAS Pareto-front plot and its reconstruction CSV.

Expected input:
    Analyze one NAS study output folder with the same required inputs as the
    full NAS analysis flow:
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

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_pareto_front.py'
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


def write_pareto_front(context: NasStudyContext) -> tuple[Path, Path]:
    """Render and save the Pareto-front plot for one prepared NAS study.

    Args:
        context (NasStudyContext): Prepared NAS study context returned by
            ``load_nas_study_context``. The function uses only the filtered
            valid trials, top-k rank markers, and the selected final trial from
            this context.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the Pareto
        figure and the exact plotted series.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    figure, axes = create_figure()
    trials_df = context.filtered_trials_df.sort_values("trial_id", kind="mergesort")

    axes.scatter(
        trials_df["trained_ram_peak_kib"],
        trials_df["best_qt_val_metric"],
        s=42,
        color=PALETTE["primary"],
        edgecolor="white",
        linewidth=0.6,
        label="Valid trials",
        zorder=3,
    )

    top_k_df = trials_df[trials_df["is_top_k"]].sort_values("rank", kind="mergesort")
    axes.scatter(
        top_k_df["trained_ram_peak_kib"],
        top_k_df["best_qt_val_metric"],
        s=74,
        facecolors="none",
        edgecolors=PALETTE["highlight"],
        linewidth=1.3,
        marker="o",
        label="Top-k trials",
        zorder=4,
    )

    selected_df = trials_df[trials_df["trial_id"] == context.selected_trial_id]
    axes.scatter(
        selected_df["trained_ram_peak_kib"],
        selected_df["best_qt_val_metric"],
        s=78,
        color=PALETTE["selection"],
        marker="D",
        edgecolor="white",
        linewidth=0.7,
        label="Selected trial",
        zorder=5,
    )

    style_axes(
        axes,
        xlabel="Peak RAM (KiB)",
        ylabel=f"INT8 {context.objective_label}",
        title="Pareto tradeoff between RAM and INT8 validation objective",
    )
    add_reference_legend(axes, location="upper right")

    csv_rows = []
    for trial_row in trials_df.itertuples(index=False):
        csv_rows.append(
            {
                "series": "valid_trials",
                "trial_id": int(trial_row.trial_id),
                "x": float(trial_row.trained_ram_peak_kib),
                "y": float(trial_row.best_qt_val_metric),
                "is_pareto": bool(trial_row.is_pareto),
                "is_top_k": bool(trial_row.is_top_k),
                "is_selected": bool(trial_row.trial_id == context.selected_trial_id),
                "ram_limit_kib": float("nan"),
            }
        )
    pareto_csv_df = pd.DataFrame(csv_rows)

    png_path = save_figure(figure, context.directories["plot_png"] / "pareto_front.png")
    csv_path = save_dataframe(pareto_csv_df, context.directories["plot_csv"] / "pareto_front.csv")
    return png_path, csv_path


def run_pareto_front_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Load one NAS study and generate only the Pareto-front outputs.

    Args:
        study_run_dir (Path): NAS study output directory containing the
            ``optuna_study`` artifacts required by the shared loader.
        output_root_dir (Path): Root output directory for the generated PNG and
            CSV. The standard NAS output tree is created beneath this root.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the requested
        single-plot regeneration.

    Raises:
        FileNotFoundError: If the study inputs are missing.
        ValueError: If the study data cannot satisfy the filtering contract.
        OSError: If the outputs cannot be written.
    """
    return write_pareto_front(load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir))


if __name__ == "__main__":
    run_pareto_front_analysis(study_run_dir=STUDY_RUN_DIR)
