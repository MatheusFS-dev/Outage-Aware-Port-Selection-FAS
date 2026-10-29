"""Generate one dissertation outage-probability figure source per fading regime."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import csv
import math
from pathlib import Path
import re
import statistics


REPOSITORY_ROOT = _COMPANION_ROOT
FIGURE_SOURCE_DIRECTORY = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/figures/sources'
APPENDIX_FRAGMENT = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/tables/op_regime_figures.tex'
)
ALL_REGIME_RESULTS_ROOT = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_all_regimes'
)
PER_REGIME_RESULTS_ROOT = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/train_per_regime'
)
NAS_SOURCE = (
    _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/results/nas_pqt_mobv3_all_regimes/dissertation_analysis/heldout_by_regime.csv'
)

REGIMES = (
    ("kappa0_mu1_m0", "kappa0_mu1_m0", r"$\kappa\!\rightarrow\!0$, $\mu=1$, $m\!\rightarrow\!0$"),
    ("kappa0_mu1_m2", "kappa0_mu1_m2", r"$\kappa\!\rightarrow\!0$, $\mu=1$, $m=2$"),
    ("kappa0_mu1_m50", "kappa0_mu1_minf", r"$\kappa\!\rightarrow\!0$, $\mu=1$, $m\!\rightarrow\!\infty$"),
    ("kappa0_mu2_m50", "kappa0_mu2_minf", r"$\kappa\!\rightarrow\!0$, $\mu=2$, $m\!\rightarrow\!\infty$"),
    ("kappa0_mu5_m50", "kappa0_mu5_minf", r"$\kappa\!\rightarrow\!0$, $\mu=5$, $m\!\rightarrow\!\infty$"),
    ("kappa5_mu1_m0", "kappa5_mu1_m0", r"$\kappa=5$, $\mu=1$, $m\!\rightarrow\!0$"),
    ("kappa5_mu1_m2", "kappa5_mu1_m2", r"$\kappa=5$, $\mu=1$, $m=2$"),
    ("kappa5_mu1_m50", "kappa5_mu1_minf", r"$\kappa=5$, $\mu=1$, $m\!\rightarrow\!\infty$"),
    ("kappa5_mu2_m0", "kappa5_mu2_m0", r"$\kappa=5$, $\mu=2$, $m\!\rightarrow\!0$"),
    ("kappa5_mu2_m2", "kappa5_mu2_m2", r"$\kappa=5$, $\mu=2$, $m=2$"),
    ("kappa5_mu2_m50", "kappa5_mu2_minf", r"$\kappa=5$, $\mu=2$, $m\!\rightarrow\!\infty$"),
    ("kappa5_mu5_m0", "kappa5_mu5_m0", r"$\kappa=5$, $\mu=5$, $m\!\rightarrow\!0$"),
    ("kappa5_mu5_m2", "kappa5_mu5_m2", r"$\kappa=5$, $\mu=5$, $m=2$"),
    ("kappa5_mu5_m50", "kappa5_mu5_minf", r"$\kappa=5$, $\mu=5$, $m\!\rightarrow\!\infty$"),
)
PER_REGIME_RUN_PATTERN = re.compile(
    r"train_(mobv[123])_.*_regime_(.+)_([3-7])_ports"
)
ALL_REGIME_RUN_PATTERN = re.compile(
    r"train_(mobv[123])_alpha02_60_20_20_all_regimes_([3-7])_ports"
)
BACKBONE_DISPLAY_NAMES = {
    "mobv1": "MobileNetV1",
    "mobv2": "MobileNetV2",
    "mobv3": "MobileNetV3",
}


def read_all_regime_validation_rows():
    """Aggregate the complete all-regime validation campaign for plotting.

    Returns:
        list[dict[str, str]]: Exactly 210 rows covering three backbones, 14
        validation regimes, and five observation budgets. Learned outage
        probability is summarized over five paired training seeds.

    Raises:
        FileNotFoundError: If the all-regime result directory is unavailable.
        ValueError: If run names, split rows, samples, seeds, regimes, budgets,
        or fixed reference values are incomplete or inconsistent.
    """
    if not ALL_REGIME_RESULTS_ROOT.is_dir():
        raise FileNotFoundError(
            f"All-regime result directory does not exist: {ALL_REGIME_RESULTS_ROOT}"
        )

    grouped_runs = {}
    summary_paths = sorted(
        ALL_REGIME_RESULTS_ROOT.glob(
            "seed_*/train_*/op_eval_t1.1885399/op_results_by_split_and_regime.csv"
        )
    )
    if len(summary_paths) != 75:
        raise ValueError(
            f"Expected 75 all-regime evaluation summaries, found {len(summary_paths)}"
        )

    for summary_path in summary_paths:
        training_name = summary_path.parents[1].name
        run_match = ALL_REGIME_RUN_PATTERN.fullmatch(training_name)
        if run_match is None:
            raise ValueError(f"Unexpected all-regime run name: {training_name}")
        backbone, observed_ports_text = run_match.groups()
        observed_ports = int(observed_ports_text)
        seed_directory = summary_path.parents[2].name
        if not seed_directory.startswith("seed_"):
            raise ValueError(f"Unexpected seed directory: {seed_directory}")
        seed = int(seed_directory.removeprefix("seed_"))

        validation_rows = [
            row for row in read_csv_rows(summary_path) if row["split"] == "val"
        ]
        if len(validation_rows) != 14:
            raise ValueError(
                f"Expected 14 validation rows in {summary_path}, found "
                f"{len(validation_rows)}"
            )
        for source_row in validation_rows:
            regime = source_row["regime"]
            if int(source_row["n_observed_ports"]) != observed_ports:
                raise ValueError(
                    f"Observation budget differs from run name: {summary_path}"
                )
            if int(source_row["num_samples"]) != 200000:
                raise ValueError(
                    f"Expected 200000 validation samples: {summary_path}"
                )
            condition = (backbone, regime, observed_ports)
            condition_runs = grouped_runs.setdefault(condition, {})
            if seed in condition_runs:
                raise ValueError(f"Duplicate seed {seed} for condition {condition}")
            condition_runs[seed] = {
                "ideal_op": float(source_row["ideal_op"]),
                "observed_op": float(source_row["observed_op"]),
                "model_op": float(source_row["model_plus_observed_op"]),
            }

    expected_conditions = {
        (backbone, regime, observed_ports)
        for backbone in BACKBONE_DISPLAY_NAMES
        for regime, _, _ in REGIMES
        for observed_ports in range(3, 8)
    }
    if set(grouped_runs) != expected_conditions:
        missing = sorted(expected_conditions - set(grouped_runs))
        unexpected = sorted(set(grouped_runs) - expected_conditions)
        raise ValueError(
            f"All-regime condition coverage differs: missing={missing}, "
            f"unexpected={unexpected}"
        )

    output_rows = []
    for backbone in BACKBONE_DISPLAY_NAMES:
        for regime, _, _ in REGIMES:
            for observed_ports in range(3, 8):
                condition = (backbone, regime, observed_ports)
                condition_runs = grouped_runs[condition]
                if set(condition_runs) != set(range(5)):
                    raise ValueError(
                        f"Expected seeds 0 through 4 for condition {condition}"
                    )
                ideal_values = {
                    run["ideal_op"] for run in condition_runs.values()
                }
                observed_values = {
                    run["observed_op"] for run in condition_runs.values()
                }
                if len(ideal_values) != 1 or len(observed_values) != 1:
                    raise ValueError(
                        f"Reference OP varies across seeds for condition {condition}"
                    )
                model_values = [
                    condition_runs[seed]["model_op"] for seed in range(5)
                ]
                output_rows.append(
                    {
                        "regime": regime,
                        "observed_ports": str(observed_ports),
                        "backbone": BACKBONE_DISPLAY_NAMES[backbone],
                        "seed_count": "5",
                        "mean_test_op": f"{statistics.fmean(model_values):.15g}",
                        "sd_test_op": f"{statistics.stdev(model_values):.15g}",
                        "ideal_op": f"{ideal_values.pop():.15g}",
                        "observed_op": f"{observed_values.pop():.15g}",
                    }
                )
    return output_rows


def read_csv_rows(path):
    """Read a CSV file as dictionaries without altering the stored values.

    Args:
        path (pathlib.Path): Existing UTF-8 CSV file to read.

    Returns:
        list[dict[str, str]]: Rows keyed by the source header names.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the CSV has no header or contains no data rows.
    """
    with path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def read_per_regime_validation_rows():
    """Aggregate the complete per-regime validation campaign for plotting.

    Each returned row represents one backbone, fading regime, and observation
    budget. The guarded validation outage probability is averaged over the five
    training seeds, while its sample standard deviation represents training-seed
    variability. Ideal and observed-only values must be identical across seeds
    within each condition.

    Args:
        None.

    Returns:
        list[dict[str, str]]: Exactly 210 summary rows covering three
        backbones, 14 regimes, and five observation budgets. The field names
        match the existing backbone plotting pipeline.

    Raises:
        FileNotFoundError: If the per-regime result directory or a referenced
        validation CSV is unavailable.
        ValueError: If run naming, seed coverage, split metadata, sample count,
        regime identity, observation budget, or fixed-reference values are
        incomplete or inconsistent.

    Examples:
        ``read_per_regime_validation_rows()`` returns the complete 210-row
        validation summary used by the specialized-backbone figures.
    """
    if not PER_REGIME_RESULTS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Per-regime result directory does not exist: {PER_REGIME_RESULTS_ROOT}"
        )

    grouped_runs = {}
    result_paths = sorted(
        PER_REGIME_RESULTS_ROOT.glob(
            "seed_*/train_*/op_eval_t1.1885399/val/op_results_by_regime.csv"
        )
    )
    if len(result_paths) != 1050:
        raise ValueError(
            f"Expected 1050 per-regime validation results, found {len(result_paths)}"
        )

    for result_path in result_paths:
        training_name = result_path.parents[2].name
        run_match = PER_REGIME_RUN_PATTERN.fullmatch(training_name)
        if run_match is None:
            raise ValueError(f"Unexpected per-regime run name: {training_name}")
        backbone, regime, observed_ports_text = run_match.groups()
        observed_ports = int(observed_ports_text)
        seed_directory = result_path.parents[3].name
        if not seed_directory.startswith("seed_"):
            raise ValueError(f"Unexpected seed directory: {seed_directory}")
        seed = int(seed_directory.removeprefix("seed_"))

        source_rows = read_csv_rows(result_path)
        if len(source_rows) != 1:
            raise ValueError(
                f"Expected one validation row in {result_path}, found {len(source_rows)}"
            )
        source_row = source_rows[0]
        if source_row["split"] != "val":
            raise ValueError(f"Expected validation split in {result_path}")
        if source_row["regime"] != regime:
            raise ValueError(f"Regime metadata differs from run name: {result_path}")
        if int(source_row["n_observed_ports"]) != observed_ports:
            raise ValueError(f"Observation budget differs from run name: {result_path}")
        if int(source_row["num_samples"]) != 200000:
            raise ValueError(f"Expected 200000 validation samples: {result_path}")

        condition = (backbone, regime, observed_ports)
        condition_runs = grouped_runs.setdefault(condition, {})
        if seed in condition_runs:
            raise ValueError(f"Duplicate seed {seed} for condition {condition}")
        condition_runs[seed] = {
            "ideal_op": float(source_row["ideal_op"]),
            "observed_op": float(source_row["observed_op"]),
            "model_op": float(source_row["model_plus_observed_op"]),
        }

    expected_conditions = {
        (backbone, regime, observed_ports)
        for backbone in BACKBONE_DISPLAY_NAMES
        for regime, _, _ in REGIMES
        for observed_ports in range(3, 8)
    }
    if set(grouped_runs) != expected_conditions:
        missing = sorted(expected_conditions - set(grouped_runs))
        unexpected = sorted(set(grouped_runs) - expected_conditions)
        raise ValueError(
            f"Per-regime condition coverage differs: missing={missing}, "
            f"unexpected={unexpected}"
        )

    output_rows = []
    for backbone in BACKBONE_DISPLAY_NAMES:
        for regime, _, _ in REGIMES:
            for observed_ports in range(3, 8):
                condition = (backbone, regime, observed_ports)
                condition_runs = grouped_runs[condition]
                if set(condition_runs) != set(range(5)):
                    raise ValueError(
                        f"Expected seeds 0 through 4 for condition {condition}"
                    )
                ideal_values = {
                    run["ideal_op"] for run in condition_runs.values()
                }
                observed_values = {
                    run["observed_op"] for run in condition_runs.values()
                }
                if len(ideal_values) != 1 or len(observed_values) != 1:
                    raise ValueError(
                        f"Reference OP varies across seeds for condition {condition}"
                    )
                model_values = [
                    condition_runs[seed]["model_op"] for seed in range(5)
                ]
                output_rows.append(
                    {
                        "regime": regime,
                        "observed_ports": str(observed_ports),
                        "backbone": BACKBONE_DISPLAY_NAMES[backbone],
                        "seed_count": "5",
                        "mean_test_op": f"{statistics.fmean(model_values):.15g}",
                        "sd_test_op": f"{statistics.stdev(model_values):.15g}",
                        "ideal_op": f"{ideal_values.pop():.15g}",
                        "observed_op": f"{observed_values.pop():.15g}",
                    }
                )
    return output_rows


def write_csv_rows(path, fieldnames, rows):
    """Write deterministic per-regime plot data to a CSV file.

    Args:
        path (pathlib.Path): Destination CSV path.
        fieldnames (tuple[str, ...]): Ordered output column names.
        rows (list[dict[str, str]]): Five validated observation-budget rows.

    Returns:
        None: The destination is written in UTF-8 with LF-compatible CSV rows.

    Raises:
        OSError: If the destination cannot be created or written.
        ValueError: If a row does not match ``fieldnames``.
    """
    with path.open("w", encoding="utf-8", newline="") as destination_file:
        writer = csv.DictWriter(
            destination_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def select_backbone_rows(source_rows, regime):
    """Pivot one regime's long-form backbone results into five plot rows.

    Args:
        source_rows (list[dict[str, str]]): Backbone summary rows containing one
            row per backbone, regime, and observation budget.
        regime (str): Exact internal regime identifier to select.

    Returns:
        list[dict[str, str]]: Rows for budgets 3 through 7 with ideal and
        observed-only OP, plus means, sample standard deviations, and seed
        counts for all three learned backbones.

    Raises:
        ValueError: If any regime--budget--backbone combination is missing,
            duplicated, has a seed count other than five, or uses another budget.

    Examples:
        ``select_backbone_rows(rows, "kappa5_mu2_m2")`` returns five rows.
    """
    selected = {}
    for row in source_rows:
        if row["regime"] != regime:
            continue
        key = (int(row["observed_ports"]), row["backbone"])
        if key in selected:
            raise ValueError(f"Duplicate backbone result for {regime}: {key}")
        if int(row["seed_count"]) != 5:
            raise ValueError(f"Expected five seeds for {regime}: {key}")
        selected[key] = row

    output_rows = []
    for observed_ports in range(3, 8):
        output_row = {"observed_ports": str(observed_ports), "seed_count": "5"}
        for backbone, prefix in (
            ("MobileNetV1", "mobilenet_v1"),
            ("MobileNetV2", "mobilenet_v2"),
            ("MobileNetV3", "mobilenet_v3"),
        ):
            key = (observed_ports, backbone)
            if key not in selected:
                raise ValueError(f"Missing backbone result for {regime}: {key}")
            output_row[f"{prefix}_mean_test_op"] = selected[key]["mean_test_op"]
            output_row[f"{prefix}_sd_test_op"] = selected[key]["sd_test_op"]
        ideal_values = {selected[key]["ideal_op"] for key in selected if key[0] == observed_ports}
        observed_values = {
            selected[key]["observed_op"] for key in selected if key[0] == observed_ports
        }
        if len(ideal_values) != 1 or len(observed_values) != 1:
            raise ValueError(f"Reference OP differs by backbone for {regime}, N={observed_ports}")
        output_row["ideal_op"] = ideal_values.pop()
        output_row["observed_only_op"] = observed_values.pop()
        output_rows.append(output_row)

    if len(selected) != 15:
        raise ValueError(f"Unexpected backbone rows for {regime}: {len(selected)}")
    return output_rows


def select_nas_rows(source_rows, regime):
    """Select and validate one regime's five held-out NAS result rows.

    Args:
        source_rows (list[dict[str, str]]): Held-out selected-model results with
            one row per regime and observation budget.
        regime (str): Exact internal regime identifier to select.

    Returns:
        list[dict[str, str]]: Five rows with ideal, observed-only, FP32-assisted,
        and INT8-assisted outage probabilities.

    Raises:
        ValueError: If budgets are missing or duplicated, if sample counts differ
            from 200,000, or if the selected regime contains unexpected rows.

    Examples:
        ``select_nas_rows(rows, "kappa5_mu5_m50")`` returns budgets 3 through 7.
    """
    selected = {}
    for row in source_rows:
        if row["regime"] != regime:
            continue
        observed_ports = int(row["observed_ports"])
        if observed_ports in selected:
            raise ValueError(f"Duplicate NAS result for {regime}, N={observed_ports}")
        if int(row["num_samples"]) != 200000:
            raise ValueError(f"Expected 200000 held-out samples for {regime}")
        selected[observed_ports] = row

    fieldnames = (
        "observed_ports",
        "num_samples",
        "ideal_op",
        "observed_only_op",
        "fp32_model_plus_observed_op",
        "int8_model_plus_observed_op",
    )
    output_rows = []
    for observed_ports in range(3, 8):
        if observed_ports not in selected:
            raise ValueError(f"Missing NAS result for {regime}, N={observed_ports}")
        source_row = selected[observed_ports]
        output_rows.append({name: source_row[name] for name in fieldnames})
    if len(selected) != 5:
        raise ValueError(f"Unexpected NAS rows for {regime}: {len(selected)}")
    return output_rows


def logarithmic_axis_exponents(lower_values, upper_values):
    """Compute enclosing base-10 decades for a logarithmic OP axis.

    Args:
        lower_values (list[float]): Positive lower curve or uncertainty bounds.
        upper_values (list[float]): Positive upper curve or uncertainty bounds.
            The list must have the same length as ``lower_values``.

    Returns:
        tuple[int, int]: Inclusive lower and upper base-10 exponents. At least
        one complete decade is returned so logarithmic tick labels remain
        readable when all plotted values occupy a narrow interval.

    Raises:
        ValueError: If the lists are empty, have different lengths, contain
            non-finite values, contain non-positive values, or have a lower
            bound greater than its corresponding upper bound.

    Examples:
        ``logarithmic_axis_exponents([0.01, 0.03], [0.02, 0.2])`` returns
        ``(-2, 0)``.
    """
    if not lower_values or len(lower_values) != len(upper_values):
        raise ValueError("Logarithmic bounds require equally sized non-empty lists")
    for lower_value, upper_value in zip(lower_values, upper_values):
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError("Logarithmic bounds must be finite")
        if lower_value <= 0 or upper_value <= 0:
            raise ValueError("Logarithmic bounds must be positive")
        if lower_value > upper_value:
            raise ValueError("A logarithmic lower bound cannot exceed its upper bound")

    lower_exponent = math.floor(math.log10(min(lower_values)))
    upper_exponent = math.ceil(math.log10(max(upper_values)))
    if upper_exponent == lower_exponent:
        upper_exponent += 1
    return lower_exponent, upper_exponent


def tight_logarithmic_axis(lower_values, upper_values, tick_count=5):
    """Compute tight positive bounds and explicit ticks for a log OP axis.

    Args:
        lower_values (list[float]): Positive lower curve or uncertainty bounds.
        upper_values (list[float]): Positive upper curve or uncertainty bounds.
            The list must have the same length as ``lower_values``.
        tick_count (int): Number of explicit major ticks. It must be at least
            two. Ticks are evenly spaced in base-10 logarithmic space so the
            resulting axis remains logarithmic even for near-unit OP values.

    Returns:
        tuple[float, float, tuple[float, ...]]: Positive lower bound, upper
        bound, and ordered major-tick positions. Bounds include logarithmic
        padding outside every supplied value.

    Raises:
        ValueError: If the lists are empty or misaligned, if ``tick_count`` is
            less than two, or if a bound is non-finite, non-positive, or
            ordered incorrectly.

    Examples:
        ``tight_logarithmic_axis([0.97], [0.98], 3)`` returns three ticks over
        a padded logarithmic interval containing both values.
    """
    if not lower_values or len(lower_values) != len(upper_values):
        raise ValueError("Logarithmic bounds require equally sized non-empty lists")
    if tick_count < 2:
        raise ValueError("A tight logarithmic axis requires at least two ticks")
    for lower_value, upper_value in zip(lower_values, upper_values):
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError("Logarithmic bounds must be finite")
        if lower_value <= 0 or upper_value <= 0:
            raise ValueError("Logarithmic bounds must be positive")
        if lower_value > upper_value:
            raise ValueError("A logarithmic lower bound cannot exceed its upper bound")

    minimum_log = math.log10(min(lower_values))
    maximum_log = math.log10(max(upper_values))
    logarithmic_span = maximum_log - minimum_log
    if logarithmic_span <= 0.0:
        raise ValueError("A tight logarithmic axis requires a non-zero data range")

    # Four-percent padding keeps every curve and error bar clear of the frame
    # while devoting more than 92% of the vertical range to measured values.
    padding = 0.04 * logarithmic_span
    lower_log = minimum_log - padding
    upper_log = maximum_log + padding
    if upper_log > 0.0 and max(upper_values) <= 1.0:
        upper_log = 0.0
    display_span = upper_log - lower_log
    lower_bound = 10.0 ** lower_log
    upper_bound = 10.0 ** upper_log
    tick_step = display_span / (tick_count - 1)
    tick_values = tuple(
        lower_bound
        if tick_index == 0
        else upper_bound
        if tick_index == tick_count - 1
        else 10.0 ** (lower_log + tick_index * tick_step)
        for tick_index in range(tick_count)
    )
    return lower_bound, upper_bound, tick_values


def logarithmic_plot_coordinate(value, lower_bound, upper_bound):
    """Map an outage probability to a stable normalized log coordinate.

    The mapping sends ``lower_bound`` to 1 and ``upper_bound`` to 10. Plotting
    the result on a base-10 logarithmic axis preserves positions proportional
    to ``log10(value)`` without exposing PGFPlots to a numerically tiny native
    log interval.

    Args:
        value (float): Positive finite outage probability within the bounds.
        lower_bound (float): Positive finite lower outage-probability bound.
        upper_bound (float): Positive finite upper outage-probability bound. It
            must be greater than ``lower_bound``.

    Returns:
        float: Normalized positive coordinate in the closed interval [1, 10].

    Raises:
        ValueError: If a value or bound is non-finite, non-positive, unordered,
            or outside the supplied interval.

    Examples:
        ``logarithmic_plot_coordinate(0.1, 0.01, 1.0)`` returns a coordinate
        between 1 and 10 whose plotted position is logarithmic in 0.1.
    """
    if not all(math.isfinite(item) for item in (value, lower_bound, upper_bound)):
        raise ValueError("Logarithmic plot coordinates must be finite")
    if lower_bound <= 0 or lower_bound >= upper_bound:
        raise ValueError("Logarithmic plot bounds must be positive and ordered")
    if value < lower_bound or value > upper_bound:
        raise ValueError("A logarithmic plot value must lie within its bounds")
    lower_log = math.log10(lower_bound)
    logarithmic_span = math.log10(upper_bound) - lower_log
    return 10.0 ** ((math.log10(value) - lower_log) / logarithmic_span)


def add_logarithmic_plot_coordinates(
    rows,
    lower_bound,
    upper_bound,
    include_references=True,
):
    """Add stable log-axis coordinates and asymmetric errors to plot rows.

    Args:
        rows (list[dict[str, str]]): Backbone rows containing ideal,
            observed-only, and three learned-model OP statistics.
        lower_bound (float): Positive lower OP bound shared by the figure.
        upper_bound (float): Positive upper OP bound shared by the figure. It
            must exceed ``lower_bound`` and contain every curve and error bar.
        include_references (bool): Whether to transform the ideal and
            observed-only columns. Pass ``False`` when those curves are not
            plotted and therefore do not define the visible axis range.

    Returns:
        list[dict[str, str]]: Copies of the input rows augmented with normalized
        logarithmic coordinates. Learned-model uncertainty is represented by
        separate positive and negative coordinate errors so the raw symmetric
        OP interval remains exact after logarithmic transformation.

    Raises:
        KeyError: If a required reference, mean, or standard-deviation column
            is absent.
        ValueError: If a learned lower error endpoint is not positive or any
            value falls outside the supplied bounds.

    Examples:
        ``add_logarithmic_plot_coordinates(rows, 0.2, 0.4)`` prepares rows for
        a tightly zoomed base-10 logarithmic PGFPlots axis.
    """
    plotted_rows = []
    for row in rows:
        plotted_row = dict(row)
        for reference_column in ("ideal_op", "observed_only_op"):
            plotted_row[f"{reference_column}_plot"] = (
                logarithmic_plot_coordinate(
                    float(row[reference_column]),
                    lower_bound,
                    upper_bound,
                )
                if include_references
                else ""
            )
        for backbone_prefix in (
            "mobilenet_v1",
            "mobilenet_v2",
            "mobilenet_v3",
        ):
            mean = float(row[f"{backbone_prefix}_mean_test_op"])
            standard_deviation = float(row[f"{backbone_prefix}_sd_test_op"])
            lower_endpoint = mean - standard_deviation
            upper_endpoint = mean + standard_deviation
            if lower_endpoint <= 0:
                raise ValueError(
                    f"Non-positive OP error endpoint for {backbone_prefix}"
                )
            mean_coordinate = logarithmic_plot_coordinate(
                mean,
                lower_bound,
                upper_bound,
            )
            lower_coordinate = logarithmic_plot_coordinate(
                lower_endpoint,
                lower_bound,
                upper_bound,
            )
            upper_coordinate = logarithmic_plot_coordinate(
                upper_endpoint,
                lower_bound,
                upper_bound,
            )
            plotted_row[f"{backbone_prefix}_plot_op"] = mean_coordinate
            plotted_row[f"{backbone_prefix}_plot_error_plus"] = (
                upper_coordinate - mean_coordinate
            )
            plotted_row[f"{backbone_prefix}_plot_error_minus"] = (
                mean_coordinate - lower_coordinate
            )
        plotted_rows.append(plotted_row)
    return plotted_rows


def format_op_tick(value, minimum_spacing):
    """Format one explicit outage-probability tick without exponent notation.

    Args:
        value (float): Positive finite outage probability shown on a tight
            logarithmic axis.
        minimum_spacing (float): Positive minimum separation between adjacent
            raw OP ticks. It determines enough decimal places to keep every
            displayed tick label distinct.

    Returns:
        str: Fixed-point label with precision suited to the value magnitude.

    Raises:
        ValueError: If ``value`` or ``minimum_spacing`` is non-finite or
            non-positive.

    Examples:
        ``format_op_tick(0.97546, 0.00025)`` returns ``"0.97546"``.
    """
    if not math.isfinite(value) or not math.isfinite(minimum_spacing):
        raise ValueError("An OP tick value and spacing must be finite")
    if value <= 0 or minimum_spacing <= 0:
        raise ValueError("An OP tick value and spacing must be positive")
    decimal_places = max(
        4,
        min(7, math.ceil(-math.log10(minimum_spacing)) + 1),
    )
    if value >= 0.001:
        return f"{value:.{decimal_places}f}"
    return f"{value:.2e}"


def backbone_figure_tex(
    csv_basename,
    regime_label,
    lower_bound,
    upper_bound,
    tick_values,
    include_references,
):
    """Build a self-contained PGFPlots source for one backbone comparison.

    Args:
        csv_basename (str): Plot-data basename without directory components.
        regime_label (str): LaTeX-formatted ``kappa``, ``mu``, and ``m`` values
            shown in the figure title.
        lower_bound (float): Positive data-derived y-axis minimum.
        upper_bound (float): Positive data-derived y-axis maximum. It must be
            greater than ``lower_bound``.
        tick_values (tuple[float, ...]): At least two positive, increasing
            major-tick positions within the supplied bounds.
        include_references (bool): If ``True``, include the ideal and
            observed-only curves before the three backbone curves for the
            all-regime comparison. If ``False``, emit only the three learned
            backbone curves for the specialized comparison. This option does
            not change the axes, uncertainty bars, output files, or data.

    Returns:
        str: Standalone LaTeX source with three learned curves and explicit
            error bars, optionally preceded by two fixed reference curves.

    Raises:
        ValueError: If ``csv_basename`` contains a path separator.
        ValueError: If the logarithmic bounds or tick positions are invalid.

    Examples:
        ``backbone_figure_tex(name, label, low, high, ticks, True)`` creates
        an all-regime source with the two fixed reference curves.
    """
    if "/" in csv_basename or "\\" in csv_basename:
        raise ValueError("The CSV basename cannot contain a path separator")
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise ValueError("The logarithmic bounds must be finite")
    if lower_bound <= 0 or lower_bound >= upper_bound:
        raise ValueError("The logarithmic bounds must be positive and ordered")
    if len(tick_values) < 2:
        raise ValueError("At least two logarithmic tick positions are required")
    if any(
        not math.isfinite(value) or value <= 0
        for value in tick_values
    ):
        raise ValueError("Logarithmic tick positions must be finite and positive")
    if tuple(sorted(tick_values)) != tuple(tick_values):
        raise ValueError("Logarithmic tick positions must be ordered")
    if tick_values[0] < lower_bound or tick_values[-1] > upper_bound:
        raise ValueError("Logarithmic tick positions must lie within the bounds")
    transformed_ticks = tuple(
        logarithmic_plot_coordinate(value, lower_bound, upper_bound)
        for value in tick_values
    )
    y_ticks = ",".join(f"{value:.12g}" for value in transformed_ticks)
    minimum_tick_spacing = min(
        upper_value - lower_value
        for lower_value, upper_value in zip(tick_values, tick_values[1:])
    )
    y_tick_labels = ",".join(
        format_op_tick(value, minimum_tick_spacing) for value in tick_values
    )
    reference_curves = ""
    if include_references:
        reference_curves = rf"""\addplot[black,dotted,line width=1.1pt,mark=none]
    table[col sep=comma,x=observed_ports,y=ideal_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{Ideal}}
\addplot[black,dashdotted,line width=1.1pt,mark=triangle,mark size=2.6pt,
    mark options={{solid,fill=white}}]
    table[col sep=comma,x=observed_ports,y=observed_only_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{Observed only}}
"""
    return rf"""\documentclass[12pt,tikz,border=2pt]{{standalone}}

\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=15

\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usetikzlibrary{{plotmarks}}
\definecolor{{uncertainty}}{{HTML}}{{B24C63}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=13.2cm,
    height=9.6cm,
    xmin=2.75,
    xmax=7.25,
    xtick={{3,4,5,6,7}},
    ymode=log,
    log basis y=10,
    % Values are normalized before this native log axis so arbitrarily tight
    % raw OP intervals remain exact without overflowing TeX dimensions.
    ymin=1,
    ymax=10,
    ytick={{{y_ticks}}},
    yticklabels={{{y_tick_labels}}},
    xlabel={{$N$}},
    ylabel={{$P_{{\mathrm{{out}}}}$}},
    axis lines=box,
    tick align=inside,
    minor tick num=0,
    tick style={{black,line width=0.35pt}},
    tick label style={{font=\normalsize}},
    label style={{font=\normalsize}},
    grid=major,
    major grid style={{draw=gray!45,solid,line width=0.35pt}},
    legend style={{at={{(0.5,1.02)}},anchor=south,draw=black,fill=white,
        font=\normalsize,legend columns=3,column sep=5pt,
        legend cell align={{left}},inner sep=3pt,row sep=-1pt}}
]
\node[anchor=north east,draw=black,fill=white,inner sep=3pt,font=\normalsize]
    at (rel axis cs:0.98,0.98) {{{regime_label}}};
{reference_curves}\addplot[black,solid,line width=1.1pt,mark=pentagon,mark size=2.8pt,
    mark options={{solid,fill=white}},error bars/.cd,y dir=both,y explicit,
    error bar style={{draw=uncertainty,line width=0.9pt}},error mark options={{draw=uncertainty,mark size=2.0pt}}]
    table[col sep=comma,x=observed_ports,y=mobilenet_v1_plot_op,
        y error plus=mobilenet_v1_plot_error_plus,
        y error minus=mobilenet_v1_plot_error_minus]
    {{{csv_basename}.csv}};
\addlegendentry{{MobileNetV1}}
\addplot[black,solid,line width=1.1pt,mark=asterisk,mark size=3.0pt,
    error bars/.cd,y dir=both,y explicit,error bar style={{draw=uncertainty,line width=0.9pt}},
    error mark options={{draw=uncertainty,mark size=2.0pt}}]
    table[col sep=comma,x=observed_ports,y=mobilenet_v2_plot_op,
        y error plus=mobilenet_v2_plot_error_plus,
        y error minus=mobilenet_v2_plot_error_minus]
    {{{csv_basename}.csv}};
\addlegendentry{{MobileNetV2}}
\addplot[black,dashed,line width=1.1pt,mark=square,mark size=2.6pt,
    mark options={{solid,fill=white}},error bars/.cd,y dir=both,y explicit,
    error bar style={{draw=uncertainty,line width=0.9pt}},error mark options={{draw=uncertainty,mark size=2.0pt}}]
    table[col sep=comma,x=observed_ports,y=mobilenet_v3_plot_op,
        y error plus=mobilenet_v3_plot_error_plus,
        y error minus=mobilenet_v3_plot_error_minus]
    {{{csv_basename}.csv}};
\addlegendentry{{MobileNetV3}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""


def add_nas_logarithmic_plot_coordinates(rows, lower_bound, upper_bound):
    """Add stable tight-log coordinates for the four selected-NAS curves.

    Args:
        rows (list[dict[str, str]]): Selected-NAS rows containing four OP
            series.
        lower_bound (float): Positive lower OP bound shared by the figure.
        upper_bound (float): Positive upper OP bound shared by the figure.

    Returns:
        list[dict[str, str]]: Copies augmented with normalized plot columns.

    Raises:
        KeyError: If a required OP column is absent.
        ValueError: If a value or bound is invalid.
    """
    plotted_rows = []
    for row in rows:
        plotted_row = dict(row)
        for column in (
            "ideal_op",
            "observed_only_op",
            "fp32_model_plus_observed_op",
            "int8_model_plus_observed_op",
        ):
            plotted_row[f"{column}_plot"] = logarithmic_plot_coordinate(
                float(row[column]),
                lower_bound,
                upper_bound,
            )
        plotted_rows.append(plotted_row)
    return plotted_rows


def nas_figure_tex(
    csv_basename,
    regime_label,
    lower_bound,
    upper_bound,
    tick_values,
):
    """Build a self-contained PGFPlots source for one selected NAS model.

    Args:
        csv_basename (str): Plot-data basename without directory components.
        regime_label (str): LaTeX-formatted ``kappa``, ``mu``, and ``m`` values
            shown in the figure title.
        lower_bound (float): Positive data-derived y-axis minimum.
        upper_bound (float): Positive data-derived y-axis maximum.
        tick_values (tuple[float, ...]): Positive, increasing raw OP ticks.

    Returns:
        str: Standalone LaTeX source with four held-out outage curves.

    Raises:
        ValueError: If ``csv_basename`` contains a path separator.
        ValueError: If the logarithmic axis exponents are not strictly ordered.

    Examples:
        ``nas_figure_tex("op_nas_selected_kappa5_mu2_m2", label)`` creates the
        corresponding standalone source text.
    """
    if "/" in csv_basename or "\\" in csv_basename:
        raise ValueError("The CSV basename cannot contain a path separator")
    if lower_bound <= 0 or lower_bound >= upper_bound:
        raise ValueError("The logarithmic bounds must be positive and ordered")
    transformed_ticks = tuple(
        logarithmic_plot_coordinate(value, lower_bound, upper_bound)
        for value in tick_values
    )
    y_ticks = ",".join(f"{value:.12g}" for value in transformed_ticks)
    minimum_tick_spacing = min(
        upper_value - lower_value
        for lower_value, upper_value in zip(tick_values, tick_values[1:])
    )
    y_tick_labels = ",".join(
        format_op_tick(value, minimum_tick_spacing) for value in tick_values
    )
    return rf"""\documentclass[12pt,tikz,border=2pt]{{standalone}}

\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=15

\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usetikzlibrary{{plotmarks}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=13.2cm,
    height=9.6cm,
    xmin=2.75,
    xmax=7.25,
    xtick={{3,4,5,6,7}},
    ymode=log,
    log basis y=10,
    % Raw OP values are mapped exactly onto a stable one-decade coordinate
    % interval so the curves use the full panel even when all values are near 1.
    ymin=1,
    ymax=10,
    ytick={{{y_ticks}}},
    yticklabels={{{y_tick_labels}}},
    xlabel={{$N$}},
    ylabel={{$P_{{\mathrm{{out}}}}$}},
    axis lines=box,
    tick align=inside,
    minor tick num=0,
    tick style={{black,line width=0.35pt}},
    tick label style={{font=\normalsize}},
    label style={{font=\normalsize}},
    grid=major,
    major grid style={{draw=gray!45,solid,line width=0.35pt}},
    legend style={{at={{(0.5,1.02)}},anchor=south,draw=black,fill=white,
        font=\normalsize,legend columns=4,column sep=5pt,
        legend cell align={{left}},inner sep=3pt,row sep=-1pt}}
]
\node[anchor=north east,draw=black,fill=white,inner sep=3pt,font=\normalsize]
    at (rel axis cs:0.98,0.98) {{{regime_label}}};
\addplot[black,solid,line width=0.9pt]
    table[col sep=comma,x=observed_ports,y=ideal_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{Ideal}}
\addplot[black,solid,line width=0.9pt,mark=asterisk,mark size=2.4pt]
    table[col sep=comma,x=observed_ports,y=observed_only_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{Observed only}}
\addplot[black,solid,line width=0.9pt,mark=pentagon,mark size=2.2pt,
    mark options={{solid,fill=white}}]
    table[col sep=comma,x=observed_ports,y=fp32_model_plus_observed_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{FP32}}
\addplot[black,dashed,line width=0.9pt,mark=square,mark size=2.0pt,
    mark options={{solid,fill=white}}]
    table[col sep=comma,x=observed_ports,y=int8_model_plus_observed_op_plot]
    {{{csv_basename}.csv}};
\addlegendentry{{INT8}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""


def figure_environment(label, basename, short_caption, caption):
    """Create one appendix figure environment for a single fading regime.

    Args:
        label (str): Display-number-based LaTeX figure label, such as ``figC.1``.
        basename (str): Compiled PDF basename in ``dissertation/Figuras/pdf``.
        short_caption (str): Plain-text list-of-figures caption.
        caption (str): Full caption describing scope, regime, and uncertainty.

    Returns:
        str: A complete LaTeX figure environment.

    Raises:
        ValueError: If ``basename`` contains a path separator.

    Examples:
        ``figure_environment("figC.1", "op_nas_selected_kappa5_mu2_m2", ... )``
        creates one appendix figure block.
    """
    if "/" in basename or "\\" in basename:
        raise ValueError("The PDF basename cannot contain a path separator")
    return rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.80\linewidth]{{../{basename}.pdf}}
\caption[{short_caption}]{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def appendix_tex():
    """Build Appendix C sections containing all single-regime OP figures.

    Args:
        None.

    Returns:
        str: Three LaTeX sections containing 42 separate figure environments.

    Raises:
        ValueError: If the configured regime metadata are not unique.

    Examples:
        ``appendix_tex()`` returns the three appendix sections with figure labels
        ``figC.1`` through ``figC.42``.
    """
    suffixes = [suffix for _, suffix, _ in REGIMES]
    if len(suffixes) != len(set(suffixes)):
        raise ValueError("Regime filename suffixes must be unique")

    sections = []
    figure_number = 1
    configurations = (
        (
            "Preliminary Backbone Evaluation Results}\n\n\\subsection{All-Regime Results",
            "op_all_regime_backbone",
            "All-regime backbone OP",
            "All-regime backbone validation \\displayac{op}",
            "Learned curves report the five-seed mean and one sample standard "
            "deviation at each observation budget. Ideal and observed-only "
            "curves are fixed references.",
        ),
        (
            "Per-Regime Results",
            "op_per_regime_backbone",
            "Per-regime backbone validation OP",
            "Per-regime backbone validation \\displayac{op}",
            "Markers and error bars report the five-seed mean guarded \\displayac{op} "
            "and one sample standard deviation at each observation budget. "
            "Every seed value uses 200,000 validation samples.",
        ),
        (
            "Neural Architecture Search Results}\n\n\\subsection{All-Regime Results",
            "op_nas_selected",
            "Selected all-regime NAS model OP",
            "Selected all-regime \\displayac{nas} model held-out \\displayac{op}",
            "Each point uses 200,000 held-out samples. The ideal, observed-only, "
            "FP32, and INT8 curves share the same samples and "
            "campaign threshold.",
        ),
    )
    for section_title, basename_prefix, short_prefix, caption_prefix, detail in configurations:
        command = "subsection" if section_title == "Per-Regime Results" else "section"
        figures = [rf"\{command}{{{section_title}}}", ""]
        figures.append(
            "Each figure reports one fading regime separately at "
            r"$\gamma_{\mathrm{th}}=1.1885399$; no curve is averaged across regimes."
        )
        figures.append(detail)
        figures.append("")
        for _, suffix, regime_label in REGIMES:
            source_basename = f"{basename_prefix}_{suffix}"
            pdf_basename = f"figC.{figure_number}_{source_basename}"
            short_caption = f"{short_prefix}: {regime_label}."
            caption = f"{caption_prefix} for {regime_label}."
            figures.append(
                figure_environment(
                    f"figC.{figure_number}",
                    pdf_basename,
                    short_caption,
                    caption,
                )
            )
            figure_number += 1
        sections.append("\n".join(figures))
    return "\n\n".join(sections).rstrip() + "\n"


def generate_regime_op_assets():
    """Generate 42 matched CSV/TEX pairs and their appendix LaTeX fragment.

    Args:
        None.

    Returns:
        None: Files are written to the dissertation source directories.

    Raises:
        FileNotFoundError: If an authoritative result CSV is absent.
        ValueError: If source coverage or sample-count invariants fail.
        OSError: If an output file cannot be written.

    Examples:
        Running this module executes ``generate_regime_op_assets()`` once and
        deterministically refreshes the matched plot sources.
    """
    all_regime_rows = read_all_regime_validation_rows()
    per_regime_rows = read_csv_rows(PER_REGIME_BACKBONE_SOURCE)
    nas_rows = read_csv_rows(NAS_SOURCE)
    backbone_fieldnames = (
        "observed_ports",
        "seed_count",
        "ideal_op",
        "observed_only_op",
        "ideal_op_plot",
        "observed_only_op_plot",
        "mobilenet_v1_mean_test_op",
        "mobilenet_v1_sd_test_op",
        "mobilenet_v1_plot_op",
        "mobilenet_v1_plot_error_plus",
        "mobilenet_v1_plot_error_minus",
        "mobilenet_v2_mean_test_op",
        "mobilenet_v2_sd_test_op",
        "mobilenet_v2_plot_op",
        "mobilenet_v2_plot_error_plus",
        "mobilenet_v2_plot_error_minus",
        "mobilenet_v3_mean_test_op",
        "mobilenet_v3_sd_test_op",
        "mobilenet_v3_plot_op",
        "mobilenet_v3_plot_error_plus",
        "mobilenet_v3_plot_error_minus",
    )
    nas_fieldnames = (
        "observed_ports",
        "num_samples",
        "ideal_op",
        "observed_only_op",
        "fp32_model_plus_observed_op",
        "int8_model_plus_observed_op",
        "ideal_op_plot",
        "observed_only_op_plot",
        "fp32_model_plus_observed_op_plot",
        "int8_model_plus_observed_op_plot",
    )

    FIGURE_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for regime, suffix, regime_label in REGIMES:
        for prefix, source_rows in (
            ("op_all_regime_backbone", all_regime_rows),
            ("op_per_regime_backbone", per_regime_rows),
        ):
            basename = f"{prefix}_{suffix}"
            plot_rows = select_backbone_rows(source_rows, regime)
            lower_values = []
            upper_values = []
            for row in plot_rows:
                if prefix == "op_all_regime_backbone":
                    for reference_column in ("ideal_op", "observed_only_op"):
                        value = float(row[reference_column])
                        lower_values.append(value)
                        upper_values.append(value)
                for backbone_prefix in (
                    "mobilenet_v1",
                    "mobilenet_v2",
                    "mobilenet_v3",
                ):
                    mean = float(row[f"{backbone_prefix}_mean_test_op"])
                    standard_deviation = float(row[f"{backbone_prefix}_sd_test_op"])
                    lower_values.append(mean - standard_deviation)
                    upper_values.append(mean + standard_deviation)
            lower_bound, upper_bound, tick_values = tight_logarithmic_axis(
                lower_values,
                upper_values,
            )
            plot_rows = add_logarithmic_plot_coordinates(
                plot_rows,
                lower_bound,
                upper_bound,
                prefix == "op_all_regime_backbone",
            )
            write_csv_rows(
                FIGURE_SOURCE_DIRECTORY / f"{basename}.csv",
                backbone_fieldnames,
                plot_rows,
            )
            (FIGURE_SOURCE_DIRECTORY / f"{basename}.tex").write_text(
                backbone_figure_tex(
                    basename,
                    regime_label,
                    lower_bound,
                    upper_bound,
                    tick_values,
                    prefix == "op_all_regime_backbone",
                ),
                encoding="utf-8",
                newline="\n",
            )

        nas_basename = f"op_nas_selected_{suffix}"
        nas_plot_rows = select_nas_rows(nas_rows, regime)
        nas_values = [
            float(row[column])
            for row in nas_plot_rows
            for column in (
                "ideal_op",
                "observed_only_op",
                "fp32_model_plus_observed_op",
                "int8_model_plus_observed_op",
            )
        ]
        lower_bound, upper_bound, tick_values = tight_logarithmic_axis(
            nas_values,
            nas_values,
        )
        nas_plot_rows = add_nas_logarithmic_plot_coordinates(
            nas_plot_rows,
            lower_bound,
            upper_bound,
        )
        write_csv_rows(
            FIGURE_SOURCE_DIRECTORY / f"{nas_basename}.csv",
            nas_fieldnames,
            nas_plot_rows,
        )
        (FIGURE_SOURCE_DIRECTORY / f"{nas_basename}.tex").write_text(
            nas_figure_tex(
                nas_basename,
                regime_label,
                lower_bound,
                upper_bound,
                tick_values,
            ),
            encoding="utf-8",
            newline="\n",
        )

    APPENDIX_FRAGMENT.write_text(appendix_tex(), encoding="utf-8", newline="\n")


def generate_all_regime_backbone_assets():
    """Generate only the completed all-regime backbone plot sources.

    This scoped entry point deliberately does not read or rewrite the pending
    per-regime backbone and NAS result families.

    Args:
        None.

    Returns:
        None: Fourteen matched CSV and standalone LaTeX files are written to
        the dissertation figure-source directory.

    Raises:
        FileNotFoundError: If the authoritative all-regime summary is absent.
        ValueError: If regime, budget, backbone, seed, or numerical coverage is
            incomplete or inconsistent.
        OSError: If a plot-data or LaTeX source file cannot be written.

    Examples:
        ``generate_all_regime_backbone_assets()`` refreshes the completed
        all-regime family without accessing pending experiment tables.
    """
    source_rows = read_all_regime_validation_rows()
    fieldnames = (
        "observed_ports",
        "seed_count",
        "ideal_op",
        "observed_only_op",
        "ideal_op_plot",
        "observed_only_op_plot",
        "mobilenet_v1_mean_test_op",
        "mobilenet_v1_sd_test_op",
        "mobilenet_v1_plot_op",
        "mobilenet_v1_plot_error_plus",
        "mobilenet_v1_plot_error_minus",
        "mobilenet_v2_mean_test_op",
        "mobilenet_v2_sd_test_op",
        "mobilenet_v2_plot_op",
        "mobilenet_v2_plot_error_plus",
        "mobilenet_v2_plot_error_minus",
        "mobilenet_v3_mean_test_op",
        "mobilenet_v3_sd_test_op",
        "mobilenet_v3_plot_op",
        "mobilenet_v3_plot_error_plus",
        "mobilenet_v3_plot_error_minus",
    )

    FIGURE_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for regime, suffix, regime_label in REGIMES:
        basename = f"op_all_regime_backbone_{suffix}"
        plot_rows = select_backbone_rows(source_rows, regime)
        lower_values = []
        upper_values = []
        for row in plot_rows:
            for reference_column in ("ideal_op", "observed_only_op"):
                value = float(row[reference_column])
                lower_values.append(value)
                upper_values.append(value)
            for backbone_prefix in (
                "mobilenet_v1",
                "mobilenet_v2",
                "mobilenet_v3",
            ):
                mean = float(row[f"{backbone_prefix}_mean_test_op"])
                standard_deviation = float(row[f"{backbone_prefix}_sd_test_op"])
                lower_values.append(mean - standard_deviation)
                upper_values.append(mean + standard_deviation)
        lower_bound, upper_bound, tick_values = tight_logarithmic_axis(
            lower_values,
            upper_values,
        )
        plot_rows = add_logarithmic_plot_coordinates(
            plot_rows,
            lower_bound,
            upper_bound,
        )
        write_csv_rows(
            FIGURE_SOURCE_DIRECTORY / f"{basename}.csv",
            fieldnames,
            plot_rows,
        )
        (FIGURE_SOURCE_DIRECTORY / f"{basename}.tex").write_text(
            backbone_figure_tex(
                basename,
                regime_label,
                lower_bound,
                upper_bound,
                tick_values,
                True,
            ),
            encoding="utf-8",
            newline="\n",
        )


def generate_per_regime_backbone_assets():
    """Generate the completed per-regime validation backbone plot sources.

    The function reads all 1,050 raw validation outputs, aggregates the five
    paired training seeds within each backbone--regime--budget condition, and
    rewrites only the 14 specialized-backbone CSV and standalone LaTeX pairs.
    All-regime and selected-NAS figure families are left unchanged.

    Args:
        None.

    Returns:
        None: Fourteen matched CSV and standalone LaTeX files are written to
        the dissertation figure-source directory.

    Raises:
        FileNotFoundError: If the per-regime result directory or any required
        raw validation result is absent.
        ValueError: If run, regime, budget, backbone, seed, split, sample-count,
        or numerical coverage is incomplete or inconsistent.
        OSError: If a plot-data or LaTeX source file cannot be written.

    Examples:
        ``generate_per_regime_backbone_assets()`` refreshes the 14 validation
        figures without accessing per-regime NAS artifacts.
    """
    source_rows = read_per_regime_validation_rows()
    fieldnames = (
        "observed_ports",
        "seed_count",
        "ideal_op",
        "observed_only_op",
        "ideal_op_plot",
        "observed_only_op_plot",
        "mobilenet_v1_mean_test_op",
        "mobilenet_v1_sd_test_op",
        "mobilenet_v1_plot_op",
        "mobilenet_v1_plot_error_plus",
        "mobilenet_v1_plot_error_minus",
        "mobilenet_v2_mean_test_op",
        "mobilenet_v2_sd_test_op",
        "mobilenet_v2_plot_op",
        "mobilenet_v2_plot_error_plus",
        "mobilenet_v2_plot_error_minus",
        "mobilenet_v3_mean_test_op",
        "mobilenet_v3_sd_test_op",
        "mobilenet_v3_plot_op",
        "mobilenet_v3_plot_error_plus",
        "mobilenet_v3_plot_error_minus",
    )

    FIGURE_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    main_figure_numbers = {
        "kappa5_mu5_minf": "6.8",
        "kappa5_mu2_m2": "6.9",
        "kappa5_mu2_m0": "6.10",
    }
    for regime, suffix, regime_label in REGIMES:
        basename = f"op_per_regime_backbone_{suffix}"
        plot_rows = select_backbone_rows(source_rows, regime)
        lower_values = []
        upper_values = []
        for row in plot_rows:
            for backbone_prefix in (
                "mobilenet_v1",
                "mobilenet_v2",
                "mobilenet_v3",
            ):
                mean = float(row[f"{backbone_prefix}_mean_test_op"])
                standard_deviation = float(row[f"{backbone_prefix}_sd_test_op"])
                lower_values.append(mean - standard_deviation)
                upper_values.append(mean + standard_deviation)
            for reference_column in ("ideal_op", "observed_only_op"):
                reference_value = float(row[reference_column])
                lower_values.append(reference_value)
                upper_values.append(reference_value)
        lower_bound, upper_bound, tick_values = tight_logarithmic_axis(
            lower_values,
            upper_values,
        )
        plot_rows = add_logarithmic_plot_coordinates(
            plot_rows,
            lower_bound,
            upper_bound,
            True,
        )
        write_csv_rows(
            FIGURE_SOURCE_DIRECTORY / f"{basename}.csv",
            fieldnames,
            plot_rows,
        )
        (FIGURE_SOURCE_DIRECTORY / f"{basename}.tex").write_text(
            backbone_figure_tex(
                basename,
                regime_label,
                lower_bound,
                upper_bound,
                tick_values,
                True,
            ),
            encoding="utf-8",
            newline="\n",
        )

        if suffix in main_figure_numbers:
            main_basename = (
                f"fig{main_figure_numbers[suffix]}_op_per_regime_backbone_{suffix}"
            )
            main_rows = select_backbone_rows(source_rows, regime)
            main_lower_values = []
            main_upper_values = []
            for row in main_rows:
                for backbone_prefix in (
                    "mobilenet_v1",
                    "mobilenet_v2",
                    "mobilenet_v3",
                ):
                    mean = float(row[f"{backbone_prefix}_mean_test_op"])
                    standard_deviation = float(
                        row[f"{backbone_prefix}_sd_test_op"]
                    )
                    main_lower_values.append(mean - standard_deviation)
                    main_upper_values.append(mean + standard_deviation)
            main_lower_bound, main_upper_bound, main_tick_values = (
                tight_logarithmic_axis(main_lower_values, main_upper_values)
            )
            main_plot_rows = add_logarithmic_plot_coordinates(
                main_rows,
                main_lower_bound,
                main_upper_bound,
                False,
            )
            write_csv_rows(
                FIGURE_SOURCE_DIRECTORY / f"{main_basename}.csv",
                fieldnames,
                main_plot_rows,
            )
            (FIGURE_SOURCE_DIRECTORY / f"{main_basename}.tex").write_text(
                backbone_figure_tex(
                    main_basename,
                    regime_label,
                    main_lower_bound,
                    main_upper_bound,
                    main_tick_values,
                    False,
                ),
                encoding="utf-8",
                newline="\n",
            )


def generate_nas_assets():
    """Generate the 14 selected all-regime NAS plot sources with tight axes.

    Args:
        None.

    Returns:
        None: Fourteen matched CSV and standalone LaTeX files are written.

    Raises:
        FileNotFoundError: If the selected-NAS source is absent.
        ValueError: If regime or budget coverage is invalid.
        OSError: If an output file cannot be written.
    """
    fieldnames = (
        "observed_ports",
        "num_samples",
        "ideal_op",
        "observed_only_op",
        "fp32_model_plus_observed_op",
        "int8_model_plus_observed_op",
        "ideal_op_plot",
        "observed_only_op_plot",
        "fp32_model_plus_observed_op_plot",
        "int8_model_plus_observed_op_plot",
    )
    FIGURE_SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for regime, suffix, regime_label in REGIMES:
        basename = f"op_nas_selected_{suffix}"
        existing_rows = read_csv_rows(
            FIGURE_SOURCE_DIRECTORY / f"{basename}.csv"
        )
        plot_rows = []
        for row in existing_rows:
            plot_rows.append(
                {
                    "observed_ports": row["observed_ports"],
                    "num_samples": row["num_samples"],
                    "ideal_op": row["ideal_op"],
                    "observed_only_op": row["observed_only_op"],
                    "fp32_model_plus_observed_op": row.get(
                        "fp32_model_plus_observed_op",
                        row.get("fp32_model_op"),
                    ),
                    "int8_model_plus_observed_op": row.get(
                        "int8_model_plus_observed_op",
                        row.get("int8_model_op"),
                    ),
                }
            )
        if len(plot_rows) != 5:
            raise ValueError(f"Expected five NAS plot rows for {regime}")
        values = [
            float(row[column])
            for row in plot_rows
            for column in (
                "ideal_op",
                "observed_only_op",
                "fp32_model_plus_observed_op",
                "int8_model_plus_observed_op",
            )
        ]
        lower_bound, upper_bound, tick_values = tight_logarithmic_axis(
            values,
            values,
        )
        plot_rows = add_nas_logarithmic_plot_coordinates(
            plot_rows,
            lower_bound,
            upper_bound,
        )
        write_csv_rows(
            FIGURE_SOURCE_DIRECTORY / f"{basename}.csv",
            fieldnames,
            plot_rows,
        )
        (FIGURE_SOURCE_DIRECTORY / f"{basename}.tex").write_text(
            nas_figure_tex(
                basename,
                regime_label,
                lower_bound,
                upper_bound,
                tick_values,
            ),
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    generate_all_regime_backbone_assets()
