"""Run the full NAS analysis workflow for MobileNet NAS study folders.

Expected input:
    Analyze one NAS study output folder or a folder of study outputs. Each
    study folder must contain:
    - ``optuna_study/config.json``
    - ``optuna_study/args/top_*_trial.txt``
    - one supported Optuna study storage artifact inside ``optuna_study/``
      such as ``study.pkl``, ``*.sqlite*``, ``*.db``, or a journal file
    - ranked-trial user attributes stored in the Optuna study

Folder examples:
    - ``STUDY_RUN_DIR`` points to one completed NAS study folder
    - ``STUDY_RUN_DIR`` points to a folder whose direct children are completed
      NAS study folders
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import logging
import sys
from pathlib import Path

CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/analysis/nas/nas_analysis.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from studies.chapter06_mobilenet_nas.scripts.analysis.nas.common import DEFAULT_OUTPUT_ROOT, save_dataframe
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.load_study_data import NasStudyContext, load_nas_study_context
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_hyperparameter_effects import write_hyperparameter_effects
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_model_size_vs_int8_loss import write_model_size_vs_int8_loss
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_objective_histograms import write_objective_histograms
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_optimization_history import write_optimization_history
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_pareto_front import write_pareto_front
from studies.chapter06_mobilenet_nas.scripts.analysis.nas.plot_quantization_degradation import write_quantization_degradation


STUDY_RUN_DIR = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/nas_pqt_mobv3_all_regimes'
)


def configure_logger(context: NasStudyContext) -> logging.Logger:
    """Create the NAS analysis file logger for one orchestrated run.

    Args:
        context (NasStudyContext): Prepared NAS study context whose output
            directories define the log-file location.

    Returns:
        logging.Logger: Configured logger that writes only to the study's
        ``analysis/outputs/nas_analysis/logs/nas_analysis.log`` file.

    Raises:
        OSError: If the log file cannot be opened for writing.
    """
    logger = logging.getLogger("nas_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(context.directories["logs"] / "nas_analysis.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def write_analysis_tables(context: NasStudyContext) -> list[Path]:
    """Write the mandatory NAS analysis tables for one prepared study.

    Args:
        context (NasStudyContext): Prepared NAS study context containing the
            filtered trial dataframe plus the mandatory top-k, feasibility, and
            Pareto summary tables.

    Returns:
        list[Path]: Written CSV paths for the full filtered trial export and
        the required table outputs.

    Raises:
        OSError: If any CSV output cannot be written.
    """
    written_paths = [
        save_dataframe(context.filtered_trials_df, context.directories["root"] / "nas_analysis_results.csv"),
        save_dataframe(context.top_ranked_trials_df, context.directories["tables"] / "top_k_trials.csv"),
        save_dataframe(context.feasibility_table_df, context.directories["tables"] / "feasibility_summary.csv"),
        save_dataframe(context.pareto_summary_df, context.directories["tables"] / "pareto_summary.csv"),
    ]
    return written_paths


def run_nas_analysis(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, list[Path]]:
    """Run the complete NAS analysis workflow for one study output folder.

    Args:
        study_run_dir (Path): Top-level NAS study output directory containing
            ``optuna_study/config.json``, the full study storage artifact, and
            ranked top-trial summaries.
        output_root_dir (Path): Root output directory for all final analysis
            artifacts. The standard table, plot, and log subdirectories are
            created beneath this root.

    Returns:
        dict[str, list[Path]]: Output-path summary keyed by ``tables``,
        ``plot_png``, ``plot_csv``, and ``logs``.

    Raises:
        FileNotFoundError: If required study inputs are missing.
        ValueError: If the study data cannot satisfy the filtering contract.
        OSError: If any CSV, PNG, or log output cannot be written.
    """
    context = load_nas_study_context(study_run_dir=study_run_dir, output_root_dir=output_root_dir)
    logger = configure_logger(context)

    logger.info("Analyzing NAS study folder: %s", context.study_run_dir)
    logger.info("Detected study scope: %s", context.study_scope)
    logger.info("Loaded %d total trials.", context.filtering_counts["loaded_trial_count"])
    for count_name, count_value in context.filtering_counts.items():
        if count_name == "loaded_trial_count":
            continue
        logger.info("%s=%d", count_name, count_value)

    table_paths = write_analysis_tables(context)
    for table_path in table_paths:
        logger.info("Wrote table CSV: %s", table_path)

    plot_outputs: list[tuple[Path, Path]] = [
        write_pareto_front(context),
        write_model_size_vs_int8_loss(context),
        write_optimization_history(context),
        write_quantization_degradation(context),
        write_objective_histograms(context),
    ]
    plot_outputs.extend(write_hyperparameter_effects(context))

    png_paths = [png_path for png_path, _ in plot_outputs]
    csv_paths = [csv_path for _, csv_path in plot_outputs]
    for png_path in png_paths:
        logger.info("Wrote plot PNG: %s", png_path)
    for csv_path in csv_paths:
        logger.info("Wrote plot CSV: %s", csv_path)

    log_paths = [context.directories["logs"] / "nas_analysis.log"]
    return {
        "tables": table_paths,
        "plot_png": png_paths,
        "plot_csv": csv_paths,
        "logs": log_paths,
    }


def discover_nas_study_run_dirs(study_run_dir: Path) -> list[Path]:
    """Resolve one study folder or the direct study children of a collection.

    Args:
        study_run_dir (Path): One NAS study folder or a collection directory
            containing NAS study folders as direct children.

    Returns:
        list[Path]: Resolved study folders in deterministic name order.

    Raises:
        FileNotFoundError: If the input does not exist or contains no studies.
    """
    resolved_study_run_dir = study_run_dir.resolve()
    if not resolved_study_run_dir.is_dir():
        raise FileNotFoundError(f"NAS study path is not a directory: {resolved_study_run_dir}.")
    if (resolved_study_run_dir / "optuna_study" / "config.json").is_file():
        return [resolved_study_run_dir]

    study_run_dirs = sorted(
        child_dir
        for child_dir in resolved_study_run_dir.iterdir()
        if child_dir.is_dir() and (child_dir / "optuna_study" / "config.json").is_file()
    )
    if not study_run_dirs:
        raise FileNotFoundError(
            f"No NAS studies found at {resolved_study_run_dir} or in its direct children."
        )
    return study_run_dirs


def run_nas_analyses(
    study_run_dir: Path,
    output_root_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, dict[str, list[Path]]]:
    """Run NAS analysis for one study or every study in one collection.

    Args:
        study_run_dir (Path): One NAS study folder or a collection directory
            containing NAS study folders as direct children.
        output_root_dir (Path): Root output directory. Collection runs write
            each study beneath a child directory named after the study.

    Returns:
        dict[str, dict[str, list[Path]]]: Output summaries keyed by study name.

    Raises:
        FileNotFoundError: If no NAS study folder can be discovered.
        ValueError: If any discovered study cannot satisfy the analysis contract.
        OSError: If any analysis artifact cannot be written.
    """
    study_run_dirs = discover_nas_study_run_dirs(study_run_dir)
    is_collection = len(study_run_dirs) > 1 or study_run_dirs[0] != study_run_dir.resolve()
    outputs_by_study = {}
    for resolved_study_dir in study_run_dirs:
        study_output_dir = output_root_dir / resolved_study_dir.name if is_collection else output_root_dir
        outputs_by_study[resolved_study_dir.name] = run_nas_analysis(
            study_run_dir=resolved_study_dir,
            output_root_dir=study_output_dir,
        )
    return outputs_by_study


if __name__ == "__main__":
    run_nas_analyses(study_run_dir=STUDY_RUN_DIR)
