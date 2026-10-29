"""Generate dissertation radar PDFs with the canonical Figure 6.21 LaTeX style."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import csv
from pathlib import Path
import shutil
import subprocess


REPOSITORY_ROOT = _COMPANION_ROOT
DISsertation_DIRECTORY = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/figures/sources'
SCRIPT_DIRECTORY = DISsertation_DIRECTORY
PDF_DIRECTORY = SCRIPT_DIRECTORY.parent
COMMON_SOURCE = SCRIPT_DIRECTORY / "colorized_radar_common.tex"
POLICY_SCRIPT_DIRECTORY = _COMPANION_ROOT / "studies/chapter08_policy_model_comparison/figures/sources"
POLICY_PDF_DIRECTORY = POLICY_SCRIPT_DIRECTORY.parent
POLICY_COMMON_SOURCE = POLICY_SCRIPT_DIRECTORY / "colorized_policy_radar_common.tex"
ALL_REGIME_SUMMARY = (
    _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/results/retrain_all_regimes/dissertation_analysis/regime_seed_summary.csv'
)
PER_REGIME_SUMMARY = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/retrain_per_regime/dissertation_analysis/regime_seed_summary.csv'
)
SELECTED_MODELS = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/nas_ptq_mobv3_per_regime/dissertation_analysis/selected_models.csv'
)
REGIMES = (
    "kappa0_mu1_m0",
    "kappa0_mu1_m2",
    "kappa0_mu1_m50",
    "kappa0_mu2_m50",
    "kappa0_mu5_m50",
    "kappa5_mu1_m0",
    "kappa5_mu1_m2",
    "kappa5_mu1_m50",
    "kappa5_mu2_m0",
    "kappa5_mu2_m2",
    "kappa5_mu2_m50",
    "kappa5_mu5_m0",
    "kappa5_mu5_m2",
    "kappa5_mu5_m50",
)
POLICIES = ("UNI", "MAR", "GC")
BUDGETS = (3, 4, 5, 6, 7)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries.

    Args:
        path: Source CSV path.

    Returns:
        Rows in source order.
    """
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Order and validate one radar's rows by the canonical regime sequence.

    Args:
        rows: Fourteen canonical radar records containing ``regime``.

    Returns:
        Records ordered according to ``REGIMES``.

    Raises:
        ValueError: If regime coverage is incomplete or duplicated.
    """
    indexed = {row["regime"]: row for row in rows}
    if len(rows) != len(REGIMES) or set(indexed) != set(REGIMES):
        raise ValueError("Radar input must contain each canonical regime exactly once")
    return [indexed[regime] for regime in REGIMES]


def write_radar_data(figure_name: str, rows: list[dict[str, str]]) -> Path:
    """Write canonical radar data with the first point repeated at 360 degrees.

    Args:
        figure_name: Dissertation figure basename.
        rows: Fourteen ordered records with ideal, observed, FP32, and INT8.

    Returns:
        Written plot-data path.
    """
    path = SCRIPT_DIRECTORY / f"{figure_name}_radar.csv"
    fieldnames = ("regime", "ideal", "observed", "fp32", "int8")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + [rows[0]])
    return path


def wrapper_tex(
    figure_name: str,
    metric: str,
) -> str:
    """Build a thin wrapper around the shared Figure 6.21 renderer.

    Args:
        figure_name: Dissertation figure basename.
        metric: Either ``op`` or ``non_outage``.
    Returns:
        Standalone wrapper source.

    Raises:
        ValueError: If ``metric`` is unsupported.
    """
    if metric == "op":
        axis_definitions = (
            r"\def\radarnonoutage{0}" "\n"
            r"\def\radartransform#1{ln(#1)/ln(10)+2}"
        )
    elif metric == "non_outage":
        axis_definitions = (
            r"\def\radarnonoutage{1}" "\n"
            r"\def\radartransform#1{#1}"
        )
    else:
        raise ValueError("metric must be 'op' or 'non_outage'")
    return (
        f"% {figure_name}: Figure 6.21 canonical radar style.\n"
        rf"\def\radardatafile{{{figure_name}_radar.csv}}" "\n"
        f"{axis_definitions}\n"
        r"\input{colorized_radar_common.tex}" "\n"
    )


def render_radar(
    figure_name: str,
    rows: list[dict[str, str]],
    metric: str,
) -> Path:
    """Write, compile, and install one canonical LaTeX radar figure.

    Args:
        figure_name: Dissertation figure basename.
        rows: Fourteen canonical radar records.
        metric: Either ``op`` or ``non_outage``.
    Returns:
        Installed dissertation PDF path.

    Raises:
        RuntimeError: If pdfLaTeX fails.
        ValueError: If rows or metric are invalid.
    """
    ordered = ordered_rows(rows)
    SCRIPT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_radar_data(figure_name, ordered)
    wrapper_path = SCRIPT_DIRECTORY / f"{figure_name}.tex"
    wrapper_path.write_text(
        wrapper_tex(figure_name, metric),
        encoding="utf-8",
        newline="\n",
    )
    command = (
        "pdflatex",
        "-interaction=batchmode",
        "-halt-on-error",
        "-output-directory=.",
        f"{figure_name}.tex",
    )
    completed = subprocess.run(
        command,
        cwd=DISsertation_DIRECTORY,
        check=False,
        capture_output=True,
        text=True,
    )
    built_pdf = SCRIPT_DIRECTORY / f"{figure_name}.pdf"
    if completed.returncode != 0 or not built_pdf.is_file():
        raise RuntimeError(
            f"pdfLaTeX failed for {figure_name}: {completed.stdout}{completed.stderr}"
        )
    installed_pdf = PDF_DIRECTORY / f"{figure_name}.pdf"
    shutil.copyfile(built_pdf, installed_pdf)
    for extension in (".aux", ".log", ".pdf"):
        build_artifact = SCRIPT_DIRECTORY / f"{figure_name}{extension}"
        build_artifact.unlink(missing_ok=True)
    return installed_pdf


def write_policy_radar_data(
    figure_name: str,
    rows: list[dict[str, str]],
) -> Path:
    """Write matched ideal, policy-baseline, and policy-model radar data.

    Args:
        figure_name: Dissertation figure basename.
        rows: Fourteen ordered regime records for one observation budget.

    Returns:
        Written plot-data path.
    """
    path = POLICY_SCRIPT_DIRECTORY / f"{figure_name}_radar.csv"
    fieldnames = (
        "regime",
        "ideal",
        "uni_observed",
        "mar_observed",
        "gc_observed",
        "uni",
        "mar",
        "gc",
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + [rows[0]])
    return path


def render_policy_radar(
    figure_name: str,
    rows: list[dict[str, str]],
) -> Path:
    """Write, compile, and install one combined reading-policy OP radar.

    Args:
        figure_name: Dissertation figure basename.
        rows: Fourteen matched policy records.

    Returns:
        Installed dissertation PDF path.

    Raises:
        RuntimeError: If pdfLaTeX fails.
        ValueError: If regime coverage is incomplete.
    """
    ordered = ordered_rows(rows)
    POLICY_SCRIPT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    POLICY_PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_policy_radar_data(figure_name, ordered)
    wrapper_path = POLICY_SCRIPT_DIRECTORY / f"{figure_name}.tex"
    wrapper_path.write_text(
        f"% {figure_name}: combined reading-policy radar.\n"
        rf"\def\radardatafile{{{figure_name}_radar.csv}}" "\n"
        r"\input{colorized_policy_radar_common.tex}" "\n",
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        (
            "pdflatex",
            "-interaction=batchmode",
            "-halt-on-error",
            "-output-directory=.",
            f"{figure_name}.tex",
        ),
        cwd=POLICY_SCRIPT_DIRECTORY,
        check=False,
        capture_output=True,
        text=True,
    )
    built_pdf = POLICY_SCRIPT_DIRECTORY / f"{figure_name}.pdf"
    if completed.returncode != 0 or not built_pdf.is_file():
        raise RuntimeError(
            f"pdfLaTeX failed for {figure_name}: {completed.stdout}{completed.stderr}"
        )
    installed_pdf = POLICY_PDF_DIRECTORY / f"{figure_name}.pdf"
    shutil.copyfile(built_pdf, installed_pdf)
    for extension in (".aux", ".log", ".pdf"):
        (POLICY_SCRIPT_DIRECTORY / f"{figure_name}{extension}").unlink(missing_ok=True)
    return installed_pdf


def generate_chapter8_policy_radars() -> list[Path]:
    """Generate one matched combined reading-policy radar per port budget.

    Returns:
        Installed PDF paths for Figures C.48--C.52.

    Raises:
        ValueError: If policy coverage or ideal-reference matching fails.
    """
    source_rows = read_rows(ALL_REGIME_SUMMARY)
    outputs = []
    for offset, budget in enumerate(BUDGETS):
        indexed = {
            (row["policy"], row["regime"]): row
            for row in source_rows
            if int(row["budget"]) == budget
            and row["artifact_type"] == "int8_tflite"
        }
        expected = {(policy, regime) for policy in POLICIES for regime in REGIMES}
        if set(indexed) != expected:
            raise ValueError(f"Incomplete Chapter 8 radar coverage for N={budget}")
        radar_rows = []
        for regime in REGIMES:
            policy_rows = {policy: indexed[(policy, regime)] for policy in POLICIES}
            ideals = {row["ideal_op_mean"] for row in policy_rows.values()}
            if len(ideals) != 1:
                raise ValueError(
                    f"Policy ideal references do not match for {regime}, N={budget}"
                )
            radar_rows.append(
                {
                    "regime": regime,
                    "ideal": ideals.pop(),
                    "uni_observed": policy_rows["UNI"]["observed_op_mean"],
                    "mar_observed": policy_rows["MAR"]["observed_op_mean"],
                    "gc_observed": policy_rows["GC"]["observed_op_mean"],
                    "uni": policy_rows["UNI"]["model_only_op_mean"],
                    "mar": policy_rows["MAR"]["model_only_op_mean"],
                    "gc": policy_rows["GC"]["model_only_op_mean"],
                }
            )
        outputs.append(render_policy_radar(f"figC.{48 + offset}", radar_rows))
    return outputs


def canonical_row(
    regime: str,
    ideal: str,
    observed: str,
    fp32: str,
    int8: str,
) -> dict[str, str]:
    """Create one normalized radar record without changing numerical values.

    Args:
        regime: Internal fading-regime identifier.
        ideal: Ideal-reference value.
        observed: Observed-only value.
        fp32: FP32 model value, or an unused placeholder.
        int8: INT8 model value.

    Returns:
        Canonical record consumed by the LaTeX renderer.
    """
    return {
        "regime": regime,
        "ideal": ideal,
        "observed": observed,
        "fp32": fp32,
        "int8": int8,
    }


def complement_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert OP records to complementary non-outage records.

    Args:
        rows: Canonical OP records.

    Returns:
        Copies whose four numeric series equal one minus OP.
    """
    output = []
    for row in rows:
        converted = {"regime": row["regime"]}
        for column in ("ideal", "observed", "fp32", "int8"):
            converted[column] = f"{1.0 - float(row[column]):.12g}"
        output.append(converted)
    return output


def generate_all_radar_figures() -> list[Path]:
    """Regenerate every radar formerly rendered as monochrome Matplotlib.

    Returns:
        Installed PDF paths for Figure 6.34, C.48--C.77, and C.92--C.101.
    """
    outputs = generate_chapter8_policy_radars()

    per_rows = read_rows(PER_REGIME_SUMMARY)
    for metric, start in (("op", 92), ("non_outage", 97)):
        for offset, budget in enumerate(BUDGETS):
            fp32 = {
                row["regime"]: row
                for row in per_rows
                if int(row["budget"]) == budget
                and row["artifact_type"] == "fp32_keras"
            }
            int8 = {
                row["regime"]: row
                for row in per_rows
                if int(row["budget"]) == budget
                and row["artifact_type"] == "int8_tflite"
            }
            radar_rows = [
                canonical_row(
                    regime,
                    fp32[regime]["ideal_op_mean"],
                    fp32[regime]["observed_op_mean"],
                    fp32[regime]["model_only_op_mean"],
                    int8[regime]["model_only_op_mean"],
                )
                for regime in REGIMES
            ]
            if metric == "non_outage":
                radar_rows = complement_rows(radar_rows)
            outputs.append(
                render_radar(
                    f"figC.{start + offset}",
                    radar_rows,
                    metric,
                )
            )

    selected_rows = [
        row for row in read_rows(SELECTED_MODELS) if int(row["observed_ports"]) == 3
    ]
    legacy_rows = [
        canonical_row(
            row["regime"],
            row["ideal_test_op"],
            row["observed_test_op"],
            row["fp32_test_op"],
            row["int8_test_op"],
        )
        for row in selected_rows
    ]
    outputs.append(
        render_radar(
            "fig6.34",
            legacy_rows,
            "op",
        )
    )
    return outputs


if __name__ == "__main__":
    for output in generate_all_radar_figures():
        print(output)
