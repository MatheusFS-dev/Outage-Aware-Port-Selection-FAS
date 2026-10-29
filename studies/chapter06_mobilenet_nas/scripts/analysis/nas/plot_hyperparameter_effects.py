"""Generate NAS hyperparameter-effect plots and reconstruction CSV files.

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

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/plot_hyperparameter_effects.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.common import (
    DEFAULT_OUTPUT_ROOT,
    PALETTE,
    add_boxplot_reference_legend,
    add_reference_legend,
    create_figure,
    draw_publication_boxplot,
    save_dataframe,
    save_figure,
    style_axes,
)
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import NasStudyContext, load_nas_study_context


STUDY_RUN_DIR = Path("./runs")
HYPERPARAMETER_PLOT_CONFIG = {
    "alpha": {
        "plot_slug": "alpha_effect",
        "xlabel": "Width multiplier α",
        "title": "Effect of width multiplier",
        "kind": "boxplot",
        "width": 6.4,
        "height": 4.6,
    },
    "truncation_block_name": {
        "plot_slug": "truncation_block_name_effect",
        "xlabel": "Truncation block",
        "title": "Effect of backbone truncation depth",
        "kind": "boxplot",
        "width": 7.0,
        "height": 4.6,
    },
    "head_depth": {
        "plot_slug": "head_depth_effect",
        "xlabel": "Head depth",
        "title": "Effect of prediction-head depth",
        "kind": "boxplot",
        "width": 6.0,
        "height": 4.6,
    },
    "head_width": {
        "plot_slug": "head_width_effect",
        "xlabel": "Head width",
        "title": "Effect of prediction-head width",
        "kind": "boxplot",
        "width": 6.6,
        "height": 4.6,
    },
    "learning_rate": {
        "plot_slug": "learning_rate_effect",
        "xlabel": "Learning rate",
        "title": "Effect of learning rate",
        "kind": "errorbar",
        "width": 6.6,
        "height": 4.4,
    },
}


def write_hyperparameter_effects(context: NasStudyContext) -> list[tuple[Path, Path]]:
    """Render and save every requested hyperparameter-effect plot for a study.

    Args:
        context (NasStudyContext): Prepared NAS study context. The function
            uses only hyperparameters that the shared loader identified as
            genuinely searched, so fixed parameters are skipped rather than
            plotted misleadingly.

    Returns:
        list[tuple[Path, Path]]: List of written ``(png_path, csv_path)`` pairs
        in the same order as the generated plots.

    Raises:
        OSError: If any PNG or CSV output cannot be written.
    """
    written_outputs: list[tuple[Path, Path]] = []
    for hyperparameter_name in context.hyperparameter_names:
        config = HYPERPARAMETER_PLOT_CONFIG[hyperparameter_name]
        if config["kind"] == "boxplot":
            written_outputs.append(write_boxplot_effect(context, hyperparameter_name))
        else:
            written_outputs.append(write_learning_rate_effect(context))
    return written_outputs


def write_boxplot_effect(context: NasStudyContext, hyperparameter_name: str) -> tuple[Path, Path]:
    """Render one categorical hyperparameter boxplot and reconstruction CSV.

    Args:
        context (NasStudyContext): Prepared NAS study context containing the
            valid-trial dataframe.
        hyperparameter_name (str): One categorical or discrete hyperparameter
            name from ``HYPERPARAMETER_PLOT_CONFIG``. The function uses the
            configured labels and dimensions for the corresponding preview.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the plot.

    Raises:
        KeyError: If the hyperparameter is not configured.
        OSError: If the PNG or CSV output cannot be written.
    """
    config = HYPERPARAMETER_PLOT_CONFIG[hyperparameter_name]
    trials_df = context.filtered_trials_df.dropna(subset=[hyperparameter_name]).copy()

    # Preserve numeric ordering for discrete numeric parameters while keeping
    # string ordering stable for categorical parameters such as block names.
    if pd.api.types.is_numeric_dtype(trials_df[hyperparameter_name]):
        ordered_values = sorted(trials_df[hyperparameter_name].unique().tolist())
    else:
        ordered_values = sorted(trials_df[hyperparameter_name].astype(str).unique().tolist())

    groups = []
    labels = []
    csv_rows = []
    for ordered_value in ordered_values:
        if pd.api.types.is_numeric_dtype(trials_df[hyperparameter_name]):
            group_df = trials_df[trials_df[hyperparameter_name] == ordered_value].copy()
            label = f"{ordered_value:g}"
        else:
            group_df = trials_df[trials_df[hyperparameter_name].astype(str) == str(ordered_value)].copy()
            label = str(ordered_value)

        groups.append(group_df["best_qt_val_metric"].to_numpy(dtype=float))
        labels.append(label)
        for trial_row in group_df.itertuples(index=False):
            csv_rows.append(
                {
                    "series": "raw",
                    "trial_id": int(trial_row.trial_id),
                    "hyperparameter_name": hyperparameter_name,
                    "hyperparameter_value": label,
                    "best_qt_val_metric": float(trial_row.best_qt_val_metric),
                }
            )

    figure, axes = create_figure(width=float(config["width"]), height=float(config["height"]))
    draw_publication_boxplot(axes, groups, labels)
    style_axes(
        axes,
        xlabel=str(config["xlabel"]),
        ylabel=f"INT8 {context.objective_label}",
        title=str(config["title"]),
        grid_axis="y",
    )
    add_boxplot_reference_legend(axes)

    png_path = save_figure(figure, context.directories["plot_png"] / f"{config['plot_slug']}.png")
    csv_path = save_dataframe(pd.DataFrame(csv_rows), context.directories["plot_csv"] / f"{config['plot_slug']}.csv")
    return png_path, csv_path


def write_learning_rate_effect(context: NasStudyContext) -> tuple[Path, Path]:
    """Render the preview-matched learning-rate effect plot and CSV output.

    Args:
        context (NasStudyContext): Prepared NAS study context containing the
            valid-trial dataframe with ``learning_rate`` values.

    Returns:
        tuple[Path, Path]: Written ``(png_path, csv_path)`` for the
        learning-rate effect plot and its reconstruction data.

    Raises:
        OSError: If the PNG or CSV output cannot be written.
    """
    config = HYPERPARAMETER_PLOT_CONFIG["learning_rate"]
    trials_df = context.filtered_trials_df.dropna(subset=["learning_rate"]).copy()
    summary_df = (
        trials_df.groupby("learning_rate", dropna=False)["best_qt_val_metric"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_int8_mse", "std": "std_int8_mse"})
        .sort_values("learning_rate", kind="mergesort")
    )
    summary_df["std_int8_mse"] = summary_df["std_int8_mse"].fillna(0.0)

    figure, axes = create_figure(width=float(config["width"]), height=float(config["height"]))
    axes.errorbar(
        summary_df["learning_rate"],
        summary_df["mean_int8_mse"],
        yerr=summary_df["std_int8_mse"],
        color=PALETTE["primary"],
        marker="o",
        markersize=5.2,
        linewidth=1.5,
        capsize=3.0,
        label="Mean +/- 1 SD",
    )
    axes.set_xscale("log")
    style_axes(
        axes,
        xlabel=str(config["xlabel"]),
        ylabel=f"Mean INT8 {context.objective_label}",
        title=str(config["title"]),
        grid_axis="y",
    )
    add_reference_legend(axes, location="upper center")

    csv_rows = []
    for trial_row in trials_df.itertuples(index=False):
        csv_rows.append(
            {
                "series": "raw",
                "trial_id": int(trial_row.trial_id),
                "hyperparameter_name": "learning_rate",
                "hyperparameter_value": float(trial_row.learning_rate),
                "best_qt_val_metric": float(trial_row.best_qt_val_metric),
                "mean_int8_mse": np.nan,
                "std_int8_mse": np.nan,
            }
        )
    for summary_row in summary_df.itertuples(index=False):
        csv_rows.append(
            {
                "series": "summary",
                "trial_id": np.nan,
                "hyperparameter_name": "learning_rate",
                "hyperparameter_value": float(summary_row.learning_rate),
                "best_qt_val_metric": np.nan,
                "mean_int8_mse": float(summary_row.mean_int8_mse),
                "std_int8_mse": float(summary_row.std_int8_mse),
            }
        )

    png_path = save_figure(figure, context.directories["plot_png"] / f"{config['plot_slug']}.png")
    csv_path = save_dataframe(pd.DataFrame(csv_rows), context.directories["plot_csv"] / f"{config['plot_slug']}.csv")
    return png_path, csv_path


def run_hyperparameter_effect_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> list[tuple[Path, Path]]:
    """Load one NAS study and generate only the hyperparameter-effect outputs.

    Args:
        study_run_dir (Path): NAS study output directory containing the
            ``optuna_study`` artifacts required by the shared loader.
        output_root_dir (Path): Root output directory for the generated PNG and
            CSV files.

    Returns:
        list[tuple[Path, Path]]: Written ``(png_path, csv_path)`` pairs for all
        generated hyperparameter effect plots.

    Raises:
        FileNotFoundError: If the study inputs are missing.
        ValueError: If the study data cannot satisfy the filtering contract.
        OSError: If any output cannot be written.
    """
    return write_hyperparameter_effects(
        load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    )


if __name__ == "__main__":
    run_hyperparameter_effect_analysis(study_run_dir=STUDY_RUN_DIR)
