"""Analyze Chapter 8 all-regime retraining results and create dissertation figures."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter


POLICIES = ("UNI", "MAR", "GC")
BUDGETS = (3, 4, 5, 6, 7)
ARTIFACTS = ("fp32_keras", "int8_tflite")
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
REPRESENTATIVE_REGIMES = (
    (
        "kappa5_mu5_m50",
        r"$\kappa=5$, $\mu=5$, $m\rightarrow\infty$",
        "kappa5_mu5_minf",
    ),
    ("kappa5_mu2_m2", r"$\kappa=5$, $\mu=2$, $m=2$", "kappa5_mu2_m2"),
    (
        "kappa5_mu2_m0",
        r"$\kappa=5$, $\mu=2$, $m\rightarrow0$",
        "kappa5_mu2_m0",
    ),
)
REGIME_LABELS = (
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!\to\!0$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!=\!2$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!1,m\!\to\!\infty$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!2,m\!\to\!\infty$",
    r"$\kappa\!\to\!0$" "\n" r"$\mu\!=\!5,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!1,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!2,m\!\to\!\infty$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!\to\!0$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!=\!2$",
    r"$\kappa\!=\!5$" "\n" r"$\mu\!=\!5,m\!\to\!\infty$",
)
RUN_PATTERN = re.compile(r"seed_(\d+)/train_nas_top1_(uni|mar|gc)_(\d+)_ports")
OP_FIGURE_SIZE_INCHES = (13.2 / 2.54, 9.6 / 2.54)
STANDALONE_BORDER_INCHES = 2.0 / 72.0
RADAR_COLORS = {
    "Ideal": "black",
    "Observed only": "#009E73",
    "FP32": "#666666",
    "INT8": "#D55E00",
}


def _set_plot_style() -> None:
    """Apply the dissertation's restrained Computer Modern plot style.

    Returns:
        None: Matplotlib's process-wide style is updated.

    Raises:
        KeyError: If Matplotlib rejects a configured style key.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _set_op_axis_scale(axis, plotted_bounds: list[np.ndarray]) -> None:
    """Apply WCNPS-style logarithmic limits and visible decade labels.

    Args:
        axis (matplotlib.axes.Axes): Cartesian axis receiving the OP scale.
            The function changes its y-scale, limits, tick locators, and tick
            formatters in place.
        plotted_bounds (list[np.ndarray]): OP series and uncertainty bounds
            that determine the enclosing powers of ten. Every value must be a
            finite probability in the interval ``(0, 1]``.

    Returns:
        None: The supplied axis is configured in place.

    Raises:
        ValueError: If no values are supplied or any plotted bound is
            non-finite, non-positive, or greater than one.

    Examples:
        >>> figure, axis = plt.subplots()
        >>> _set_op_axis_scale(axis, [np.array([0.12, 0.30])])
        >>> axis.get_ylim()
        (0.1, 1.0)
    """
    if not plotted_bounds:
        raise ValueError("plotted_bounds must contain at least one OP series.")
    values = np.concatenate([np.asarray(bound, dtype=float) for bound in plotted_bounds])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("OP plot bounds must be finite probabilities in (0, 1].")

    # Enclosing full decades reproduce the WCNPS figures and guarantee at
    # least two labeled y ticks even when all measurements are close to one.
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    log_span = np.log(maximum) - np.log(minimum)
    padding = max(0.08 * log_span, 0.001)
    lower_limit = np.exp(np.log(minimum) - padding)
    upper_limit = min(1.0, np.exp(np.log(maximum) + padding))

    axis.set_yscale("log")
    axis.set_ylim(lower_limit, upper_limit)
    if maximum / minimum < 10.0:
        ticks = np.geomspace(lower_limit, upper_limit, 5)
        precision = max(
            2,
            int(np.ceil(-np.log10(upper_limit - lower_limit))) + 1,
        )
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.{precision}f}")
        )
    else:
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())


def load_results(results_root: Path) -> pd.DataFrame:
    """Load and validate every Chapter 8 held-out result summary.

    Args:
        results_root (Path): Root containing ``seed_*`` retraining directories,
            run contracts, and threshold-specific held-out summaries. Exactly
            five seeds, three policies, and five budgets are required.

    Returns:
        pd.DataFrame: Concatenated held-out rows augmented with ``seed``,
        ``policy``, and ``budget`` columns.

    Raises:
        FileNotFoundError: If a required result, contract, or pattern file is
            absent.
        ValueError: If case counts, regimes, artifacts, samples, port patterns,
            or row keys violate the Chapter 8 experimental contract.

    Examples:
        >>> frame = load_results(Path("results/retrain_all_regimes"))
        >>> frame[["seed", "policy", "budget"]].drop_duplicates().shape[0]
        75
    """
    pattern_path = results_root / "port_patterns.json"
    if not pattern_path.is_file():
        raise FileNotFoundError(f"Missing fixed port patterns: {pattern_path}.")
    fixed_patterns = json.loads(pattern_path.read_text(encoding="utf-8"))["patterns"]
    frames = []
    summary_paths = sorted(
        results_root.glob(
            "seed_*/train_nas_top1_*_ports/op_eval_t1.1885399/"
            "op_results_by_artifact_split_and_regime.csv"
        )
    )
    if len(summary_paths) != 75:
        raise ValueError(f"Expected 75 held-out summaries, found {len(summary_paths)}.")

    for summary_path in summary_paths:
        match = RUN_PATTERN.search(summary_path.as_posix())
        if match is None:
            raise ValueError(f"Unsupported retraining path: {summary_path}.")
        seed = int(match.group(1))
        policy_lower = match.group(2)
        policy = policy_lower.upper()
        budget = int(match.group(3))
        run_dir = summary_path.parents[1]
        contract_path = run_dir / "model_training" / "args" / "run_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"Missing retraining contract: {contract_path}.")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_indices = fixed_patterns[policy_lower][str(budget)]["sorted_pattern_0based"]
        if contract["observed_indices"] != expected_indices:
            raise ValueError(f"Observed indices do not match fixed patterns in {run_dir}.")
        evaluation_config_path = summary_path.parent / "op_eval_config.json"
        if not evaluation_config_path.is_file():
            raise FileNotFoundError(f"Missing held-out evaluation config: {evaluation_config_path}.")
        evaluation_config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
        if policy != "UNI" and evaluation_config.get("observed_indices") != expected_indices:
            raise ValueError(f"Held-out evaluation indices do not match {policy} in {run_dir}.")

        frame = pd.read_csv(summary_path)
        if len(frame) != len(ARTIFACTS) * len(REGIMES):
            raise ValueError(f"Unexpected held-out row count in {summary_path}.")
        if set(frame["artifact_type"]) != set(ARTIFACTS):
            raise ValueError(f"Artifact set mismatch in {summary_path}.")
        if set(frame["regime"]) != set(REGIMES):
            raise ValueError(f"Regime set mismatch in {summary_path}.")
        if set(frame["num_samples"]) != {200000}:
            raise ValueError(f"Held-out sample count mismatch in {summary_path}.")
        frame["seed"] = seed
        frame["policy"] = policy
        frame["budget"] = budget
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    key = ["seed", "policy", "budget", "artifact_type", "regime"]
    if results.duplicated(key).any():
        raise ValueError("Held-out results contain duplicate case keys.")
    return results


def build_summaries(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate seed variability and build budget-specific policy rankings.

    Args:
        results (pd.DataFrame): Validated held-out rows returned by
            ``load_results``. OP values must use the common 60/20/20 test split
            and threshold 1.1885399.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Regime-wise seed
        aggregates, macro averages over regimes with seed uncertainty, and the
        observed/FP32/INT8 policy-ranking table.

    Raises:
        ValueError: If any expected policy, budget, artifact, regime, or seed
            group is missing.

    Examples:
        >>> regime, macro, ranking = build_summaries(results)
        >>> set(ranking["budget"])
        {3, 4, 5, 6, 7}
    """
    metric_columns = ["ideal_op", "observed_op", "model_plus_observed_op", "model_only_op"]
    regime_summary = (
        results.groupby(["policy", "budget", "artifact_type", "regime"], sort=True)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    regime_summary.columns = [
        "_".join(value).rstrip("_") if isinstance(value, tuple) else value
        for value in regime_summary.columns
    ]

    seed_macro = (
        results.groupby(["policy", "budget", "artifact_type", "seed"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    macro_summary = (
        seed_macro.groupby(["policy", "budget", "artifact_type"], sort=True)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    macro_summary.columns = [
        "_".join(value).rstrip("_") if isinstance(value, tuple) else value
        for value in macro_summary.columns
    ]

    expected_regime_rows = len(POLICIES) * len(BUDGETS) * len(ARTIFACTS) * len(REGIMES)
    expected_macro_rows = len(POLICIES) * len(BUDGETS) * len(ARTIFACTS)
    if len(regime_summary) != expected_regime_rows or len(macro_summary) != expected_macro_rows:
        raise ValueError("Aggregated results are missing an expected experimental group.")

    ranking_rows = []
    for budget in BUDGETS:
        budget_macro = macro_summary[macro_summary["budget"] == budget]
        observed = budget_macro[budget_macro["artifact_type"] == "fp32_keras"].set_index("policy")
        fp32 = observed
        int8 = budget_macro[budget_macro["artifact_type"] == "int8_tflite"].set_index("policy")
        observed_rank = observed["observed_op_mean"].sort_values().index.tolist()
        fp32_rank = fp32["model_only_op_mean"].sort_values().index.tolist()
        int8_rank = int8["model_only_op_mean"].sort_values().index.tolist()

        regime_budget = regime_summary[regime_summary["budget"] == budget]
        observed_regime = regime_budget[regime_budget["artifact_type"] == "fp32_keras"]
        int8_regime = regime_budget[regime_budget["artifact_type"] == "int8_tflite"]
        observed_winners = observed_regime.pivot(
            index="regime",
            columns="policy",
            values="observed_op_mean",
        ).idxmin(axis=1)
        int8_winners = int8_regime.pivot(
            index="regime",
            columns="policy",
            values="model_only_op_mean",
        ).idxmin(axis=1)
        ranking_rows.append(
            {
                "budget": budget,
                "observed_ranking": " > ".join(observed_rank),
                "fp32_ranking": " > ".join(fp32_rank),
                "int8_ranking": " > ".join(int8_rank),
                "changed_regime_winners": int((observed_winners != int8_winners).sum()),
            }
        )
    return regime_summary, macro_summary, pd.DataFrame(ranking_rows)


def _plot_representative_metric(
    regime_summary: pd.DataFrame,
    regime: str,
    regime_text: str,
    policy: str,
    metric: str,
    output_path: Path,
) -> None:
    """Plot one policy in one representative regime.

    Args:
        regime_summary (pd.DataFrame): Regime-wise five-seed aggregates from
            ``build_summaries``.
        regime (str): Internal fading-regime key present in ``REGIMES``.
        regime_text (str): Dissertation-formatted regime annotation.
        policy (str): Reading policy in ``POLICIES``. Each output contains only
            this policy while retaining its ideal, observed-only, FP32, and
            INT8 references.
        metric (str): Either ``"op"`` for outage probability or
            ``"non_outage"`` for its complement. OP uses logarithmic axes;
            non-outage uses linear axes. Both write one PDF and have no
            randomness or device placement effects.
        output_path (Path): Final PDF path. Existing files are replaced.

    Returns:
        None: A deterministic single-panel PDF is written.

    Raises:
        ValueError: If ``policy`` or ``metric`` is unsupported or the requested
            regime lacks a budget or artifact row for the policy.
        OSError: If Matplotlib cannot write ``output_path``.

    Examples:
        >>> _plot_representative_metric(
        ...     summary, "kappa5_mu2_m2", label, "UNI", "op", path
        ... )
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    if metric not in {"op", "non_outage"}:
        raise ValueError("metric must be 'op' or 'non_outage'.")
    subset = regime_summary[
        (regime_summary["regime"] == regime) & (regime_summary["policy"] == policy)
    ]
    expected_rows = len(BUDGETS) * len(ARTIFACTS)
    if len(subset) != expected_rows:
        raise ValueError(f"Representative regime {regime} is incomplete for {policy}.")

    _set_plot_style()
    figure, axis = plt.subplots(figsize=OP_FIGURE_SIZE_INCHES)
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.99)
    plotted_bounds = []
    fp32 = subset[subset["artifact_type"] == "fp32_keras"].sort_values("budget")
    int8 = subset[subset["artifact_type"] == "int8_tflite"].sort_values("budget")
    if len(fp32) != len(BUDGETS) or len(int8) != len(BUDGETS):
        raise ValueError(f"Representative plot data are incomplete for {policy}.")

    ideal = fp32["ideal_op_mean"].to_numpy()
    observed = fp32["observed_op_mean"].to_numpy()
    fp32_mean = fp32["model_only_op_mean"].to_numpy()
    fp32_std = fp32["model_only_op_std"].to_numpy()
    int8_mean = int8["model_only_op_mean"].to_numpy()
    int8_std = int8["model_only_op_std"].to_numpy()
    if metric == "non_outage":
        ideal = 1.0 - ideal
        observed = 1.0 - observed
        fp32_mean = 1.0 - fp32_mean
        int8_mean = 1.0 - int8_mean
    plotted_bounds.extend(
        (
            ideal,
            observed,
            fp32_mean - fp32_std,
            fp32_mean + fp32_std,
            int8_mean - int8_std,
            int8_mean + int8_std,
        )
    )

    axis.plot(BUDGETS, ideal, color="black", linestyle="-", label="Ideal")
    axis.plot(
        BUDGETS,
        observed,
        color="black",
        linestyle="-",
        marker="*",
        label="Observed only",
    )
    axis.errorbar(
        BUDGETS,
        fp32_mean,
        yerr=fp32_std,
        color="black",
        linestyle="-",
        marker="p",
        markerfacecolor="white",
        ecolor="#B24C63",
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.4,
        label="FP32",
    )
    axis.errorbar(
        BUDGETS,
        int8_mean,
        yerr=int8_std,
        color="black",
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        ecolor="#B24C63",
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.4,
        label="INT8",
    )
    axis.set_xlabel(r"$N$")
    axis.set_xticks(BUDGETS)
    axis.grid(True, which="major", color="0.84", linestyle="-", linewidth=0.6)
    axis.tick_params(which="major", direction="in", top=True, right=True, length=5.0)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=3.0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    axis.text(
        0.98,
        0.98,
        regime_text,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=12,
        bbox={
            "boxstyle": "square,pad=0.25",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )
    if metric == "op":
        _set_op_axis_scale(axis, plotted_bounds)
    else:
        lower = min(float(np.min(values)) for values in plotted_bounds)
        upper = max(float(np.max(values)) for values in plotted_bounds)
        padding = max((upper - lower) * 0.08, 0.001)
        axis.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    axis.set_ylabel(r"$P_{\mathrm{out}}$" if metric == "op" else "Non-outage rate")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.32,
        labelspacing=0.0,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=STANDALONE_BORDER_INCHES,
    )
    plt.close(figure)


def plot_representative_figures(regime_summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Create policy-separated easy, medium, and very-hard Chapter 8 OP figures.

    Args:
        regime_summary (pd.DataFrame): Regime-wise five-seed aggregates from
            ``build_summaries``.
        figure_dir (Path): Dissertation PDF figure directory. Nine numbered OP
            PDF files are replaced deterministically and no other files are
            changed.

    Returns:
        list[Path]: Paths to nine OP PDFs, ordered by representative regime and
            policy.

    Raises:
        ValueError: If any representative result group is incomplete.
        OSError: If an output PDF cannot be written.

    Examples:
        >>> paths = plot_representative_figures(summary, Path("dissertation/Figuras/pdf"))
        >>> len(paths)
        9
    """
    output_paths = []
    figure_number = 1
    for regime, regime_text, regime_slug in REPRESENTATIVE_REGIMES:
        for policy in POLICIES:
            output_path = figure_dir / (
                f"fig8.{figure_number}_op_{policy.lower()}_{regime_slug}.pdf"
            )
            _plot_representative_metric(
                regime_summary,
                regime,
                regime_text,
                policy,
                "op",
                output_path,
            )
            output_paths.append(output_path)
            figure_number += 1
    return output_paths


def _plot_radar_metric(
    regime_summary: pd.DataFrame,
    budget: int,
    policy: str,
    metric: str,
    output_path: Path,
) -> None:
    """Plot one policy's cross-regime references for one observation budget.

    Args:
        regime_summary (pd.DataFrame): Regime-wise five-seed aggregates from
            ``build_summaries``.
        budget (int): Observation budget in ``{3,4,5,6,7}``.
        policy (str): Reading policy in ``POLICIES``. The output contains this
            policy's observed-only and INT8 contours plus the common
            ideal reference; no other reading policy is plotted.
        metric (str): ``"op"`` produces a logarithmic radial OP plot, whereas
            ``"non_outage"`` plots the complementary rate on ``[0,1]``.
        output_path (Path): Final Appendix C PDF path. Existing files are
            replaced deterministically.

    Returns:
        None: One radar PDF is written.

    Raises:
        ValueError: If ``budget``, ``policy``, ``metric``, or result
            completeness is invalid.
        OSError: If Matplotlib cannot write the PDF.

    Examples:
        >>> _plot_radar_metric(summary, 3, "UNI", "op", Path("figC.48.pdf"))
    """
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}.")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    if metric not in {"op", "non_outage"}:
        raise ValueError("metric must be 'op' or 'non_outage'.")
    subset = regime_summary[
        (regime_summary["budget"] == budget)
        & (regime_summary["policy"] == policy)
        & (regime_summary["artifact_type"] == "int8_tflite")
    ]
    if len(subset) != len(REGIMES):
        raise ValueError(f"Radar data are incomplete for {policy} at N={budget}.")

    _set_plot_style()
    angles = np.linspace(0.0, 2.0 * np.pi, len(REGIMES), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    figure, axis = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
    policy_rows = subset.set_index("regime").loc[list(REGIMES)]
    series = (
        ("Ideal", "ideal_op_mean", "-", None),
        ("Observed only", "observed_op_mean", "-", "*"),
        ("INT8", "model_only_op_mean", ":", "s"),
    )
    for label, column, linestyle, marker in series:
        values = policy_rows[column].to_numpy()
        if metric == "non_outage":
            values = 1.0 - values
        closed_values = np.concatenate([values, values[:1]])
        line_color = RADAR_COLORS[label]
        axis.plot(
            closed_angles,
            closed_values,
            color=line_color,
            linestyle=linestyle,
            marker=None,
            linewidth=1.8 if label != "Ideal" else 1.0,
            label=label,
        )
        if label == "Observed only":
            axis.fill(
                closed_angles,
                closed_values,
                color=RADAR_COLORS["Observed only"],
                alpha=0.10,
            )

    axis.set_xticks(angles)
    axis.set_xticklabels(REGIME_LABELS)
    axis.tick_params(axis="x", pad=10)
    axis.grid(True, color="0.80", linestyle="--", linewidth=0.45)
    axis.spines["polar"].set_color("0.55")
    axis.spines["polar"].set_linewidth(0.45)
    if metric == "op":
        axis.set_yscale("log")
        axis.set_ylim(0.001, 1.0)
        axis.set_yticks((0.01, 0.1, 1.0))
        axis.set_yticklabels((r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"))
    else:
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
        axis.set_yticklabels(("0.2", "0.4", "0.6", "0.8", "1.0"))
    # Place radial values between regime spokes and pull them inside the outer
    # circle so the 1.0 label cannot collide with a regime annotation.
    axis.set_rlabel_position(12.8)
    axis.tick_params(axis="y", pad=-2.0)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_appendix_radars(regime_summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Create policy-separated Appendix C OP and non-outage radar sequences.

    Args:
        regime_summary (pd.DataFrame): Regime-wise five-seed aggregates from
            ``build_summaries``.
        figure_dir (Path): Dissertation PDF figure directory. Files C.48--C.77
            are replaced deterministically; no other assets are modified.

    Returns:
        list[Path]: Thirty radar PDF paths ordered by metric, observation
        budget, and policy.

    Raises:
        ValueError: If any INT8 policy--budget--regime group is missing.
        OSError: If a PDF cannot be written.

    Examples:
        >>> paths = plot_appendix_radars(summary, Path("dissertation/Figuras/pdf"))
        >>> len(paths)
        30
    """
    output_paths = []
    figure_number = 48
    for metric in ("op", "non_outage"):
        for budget in BUDGETS:
            for policy in POLICIES:
                output_path = figure_dir / f"figC.{figure_number}.pdf"
                _plot_radar_metric(regime_summary, budget, policy, metric, output_path)
                output_paths.append(output_path)
                figure_number += 1
    return output_paths


def run_analysis(results_root: Path, figure_dir: Path) -> dict[str, Path]:
    """Validate results, write tabular sources, and create dissertation figures.

    Args:
        results_root (Path): Completed Chapter 8 retraining result root.
        figure_dir (Path): Dissertation PDF figure directory. The function
            writes only approved Chapter 8 and Appendix C figure numbers.

    Returns:
        dict[str, Path]: Paths to raw, regime-aggregate, macro-aggregate, and
        ranking CSV sources under ``dissertation_analysis``.

    Raises:
        FileNotFoundError: If required retraining evidence is missing.
        ValueError: If any experimental contract or aggregation check fails.
        OSError: If CSV or PDF outputs cannot be written.

    Examples:
        >>> outputs = run_analysis(Path("results/retrain_all_regimes"), figure_dir)
        >>> "ranking" in outputs
        True
    """
    results = load_results(results_root)
    regime_summary, macro_summary, ranking = build_summaries(results)
    analysis_dir = results_root / "dissertation_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "raw": analysis_dir / "held_out_results.csv",
        "regime": analysis_dir / "regime_seed_summary.csv",
        "macro": analysis_dir / "macro_seed_summary.csv",
        "ranking": analysis_dir / "policy_ranking.csv",
    }
    results.to_csv(output_paths["raw"], index=False)
    regime_summary.to_csv(output_paths["regime"], index=False)
    macro_summary.to_csv(output_paths["macro"], index=False)
    ranking.to_csv(output_paths["ranking"], index=False)
    plot_representative_figures(regime_summary, figure_dir)
    plot_appendix_radars(regime_summary, figure_dir)
    return output_paths
