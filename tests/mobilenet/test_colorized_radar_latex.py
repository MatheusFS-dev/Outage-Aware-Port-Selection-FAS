"""Regression checks for the shared Figure 6.21 radar renderer."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path


REPOSITORY_ROOT = _COMPANION_ROOT
SCRIPT_DIRECTORY = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/figures/sources'
COMMON_SOURCE = SCRIPT_DIRECTORY / "colorized_radar_common.tex"
POLICY_DIRECTORY = REPOSITORY_ROOT / "studies/chapter08_policy_model_comparison/figures/sources"


def test_shared_radar_source_copies_figure_621_visual_constants() -> None:
    """Require the canonical Figure 6.21 geometry, typography, and palette."""
    source = COMMON_SOURCE.read_text(encoding="utf-8")
    required_fragments = (
        r"width=18.42903cm",
        r"height=18.42903cm",
        r"\definecolor{observedcolor}{HTML}{009E73}",
        r"\definecolor{modelcolor}{HTML}{666666}",
        r"\definecolor{intcolor}{HTML}{D55E00}",
        r"font=\fontsize{11.5}{13.8}\selectfont",
        r"font=\fontsize{13}{15.6}\selectfont",
        r"major grid style={draw=gray!32,dashed,line width=0.3pt}",
        r"axis line style={draw=gray!55,line width=0.45pt}",
        r"at={(0.5,1.14)}",
        r"line width=1.8pt",
        r"fill opacity=0.10",
        "Rayleigh, severe",
        "Nakagami",
        "Rician, moderate",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_every_previously_monochrome_radar_uses_shared_latex_renderer() -> None:
    """Require every affected dissertation radar wrapper to use one renderer."""
    figure_names = ["fig6.34"]
    figure_names.extend(f"figC.{number}" for number in range(48, 78))
    figure_names.extend(f"figC.{number}" for number in range(92, 102))
    expected_input = r"\input{colorized_radar_common.tex}"
    for figure_name in figure_names:
        is_policy = figure_name.startswith("figC.") and 48 <= int(figure_name.split(".")[1]) <= 77
        wrapper = (POLICY_DIRECTORY if is_policy else SCRIPT_DIRECTORY) / f"{figure_name}.tex"
        is_combined_policy = figure_name.startswith("figC.") and 48 <= int(figure_name.split(".")[1]) <= 52
        renderer = r"\input{colorized_policy_radar_common.tex}" if is_combined_policy else expected_input
        assert wrapper.is_file(), figure_name
        assert renderer in wrapper.read_text(encoding="utf-8"), figure_name


def test_radar_wrapper_exposes_each_radial_tick_as_a_separate_label() -> None:
    """Prevent nested braces from collapsing all PGFPlots tick labels."""
    common = COMMON_SOURCE.read_text(encoding="utf-8")
    wrapper = (SCRIPT_DIRECTORY / "figC.92.tex").read_text(encoding="utf-8")
    assert (
        r"yticklabels={$0$,$0.2$,$0.4$,$0.6$,$0.8$,$1.0$}"
        in common
    )
    assert r"\def\radarnonoutage{0}" in wrapper


def test_shared_radar_keeps_ideal_with_figure_621_styled_series() -> None:
    """Keep Ideal while requiring the three Figure 6.21 styled series."""
    source = COMMON_SOURCE.read_text(encoding="utf-8")
    assert r"legend columns=4" in source
    assert r"\long\def\FluidraFigurePlotIdeal" in source
    assert r"\addlegendentry{Ideal}" in source
    assert r"\addlegendentry{Observed only}" in source
    assert r"\addlegendentry{FP32 model only}" in source
    assert r"\addlegendentry{INT8 model only}" in source


def test_policy_radars_keep_distinct_fp32_and_int8_data() -> None:
    """Prevent policy radars from duplicating INT8 values into FP32."""
    rows = (POLICY_DIRECTORY / "figC.48_radar.csv").read_text(encoding="utf-8")
    first_data_row = rows.splitlines()[1].split(",")
    assert first_data_row[3] != first_data_row[4]
