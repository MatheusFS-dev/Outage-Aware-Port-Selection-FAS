"""Utility helpers to evaluate outage probability (OP) per regime.

This module is intentionally lightweight and script-agnostic.
You provide the model, scaler, and regime data; the module returns
DataFrames and can save them to disk.
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .data import KMS_REGIME_LABELS


def predict_regime(
    model: tf.keras.Model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    true_values: np.ndarray,
    num_observed_ports: int,
    observed_indices: np.ndarray,
    total_ports: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model prediction for one regime and return selection details.

    Args:
        model (tf.keras.Model): Trained model producing full-port predictions.
        scaler_x (MinMaxScaler): Input scaler fitted during training.
        scaler_y (MinMaxScaler): Output scaler fitted during training.
        true_values (np.ndarray): Full-port ground-truth matrix with shape
            ``(n_samples, total_ports)``.
        num_observed_ports (int): Number of observed ports used as model input.
        observed_indices (np.ndarray): One-dimensional zero-based physical port
            indices used to construct the compact model input. Its length must
            equal ``num_observed_ports`` and all values must be unique and lie
            in ``[0, total_ports)``.
        total_ports (int): Total number of ports in each sample.
        batch_size (int): Batch size used for ``model.predict``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ``(observed_indices, predicted_values, predicted_ports,
            predicted_selected_values, true_selected_values)``.

    Raises:
        ValueError: If ``true_values`` or ``observed_indices`` is invalid.
    """
    if true_values.ndim != 2 or true_values.shape[1] != total_ports:
        raise ValueError("true_values must have shape (n_samples, total_ports).")
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    if observed_indices.ndim != 1 or len(observed_indices) != num_observed_ports:
        raise ValueError("observed_indices must be one-dimensional and match num_observed_ports.")
    if len(np.unique(observed_indices)) != len(observed_indices):
        raise ValueError("observed_indices must contain unique values.")
    if np.any(observed_indices < 0) or np.any(observed_indices >= total_ports):
        raise ValueError("observed_indices values must lie in [0, total_ports).")

    # Evaluation must preserve the policy-specific physical subset because
    # substituting uniformly spaced columns changes the model input and the
    # observed-only baseline.
    observed_values = true_values[:, observed_indices]
    scaled_inputs = scaler_x.transform(observed_values).astype(np.float32)
    scaled_predictions = model.predict(scaled_inputs, batch_size=batch_size, verbose=0)
    
    # Unscale predictions using scaler_y
    predicted_values = scaler_y.inverse_transform(scaled_predictions).astype(np.float32)

    predicted_ports = np.argmax(predicted_values, axis=1).astype(np.int64)
    row_idx = np.arange(true_values.shape[0])
    predicted_selected_values = predicted_values[row_idx, predicted_ports].astype(np.float32)
    true_selected_values = true_values[row_idx, predicted_ports].astype(np.float32)

    return (
        observed_indices.astype(np.int64),
        predicted_values.astype(np.float32),
        predicted_ports,
        predicted_selected_values,
        true_selected_values,
    )


def _compute_selected_values(
    observed_indices: np.ndarray,
    predicted_values: np.ndarray,
    true_values: np.ndarray,
    total_ports: int,
    combine_observed_and_predicted: bool,
) -> np.ndarray:
    """Return per-sample values selected by an OP evaluation strategy.

    Args:
        observed_indices (np.ndarray): One-dimensional array of observed port
            indices. Values must refer to valid columns in ``true_values``.
        predicted_values (np.ndarray): Model output matrix with shape
            ``(n_samples, total_ports)``. The largest predicted value in each
            row is treated as the model-selected port.
        true_values (np.ndarray): Ground-truth port-value matrix with shape
            ``(n_samples, total_ports)``. Selected values are always measured
            from this matrix so OP reflects realized channel quality.
        total_ports (int): Number of ports expected in each sample. This must
            match the second dimension of ``predicted_values`` and
            ``true_values``.
        combine_observed_and_predicted (bool): Selection-mode switch. If
            ``True``, the selected value is the better realized value between
            the model-selected port and the best observed port for each sample,
            which uses additional observed-port information. If ``False``, only
            the model-selected port is evaluated, which isolates model-only OP
            and avoids the extra observed-port comparison work.

    Returns:
        np.ndarray: One-dimensional array with one selected realized value per
            sample.

    Raises:
        ValueError: If matrix dimensions do not match ``total_ports``.
        IndexError: If ``observed_indices`` contains an invalid port index.

    Examples:
        >>> observed = np.array([0])
        >>> predicted = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        >>> true = np.array([[1.0, 0.2], [0.1, 0.7]], dtype=np.float32)
        >>> _compute_selected_values(observed, predicted, true, 2, False).tolist()
        [0.20000000298023224, 0.10000000149011612]
    """
    if predicted_values.ndim != 2 or predicted_values.shape[1] != total_ports:
        raise ValueError("predicted_values must have shape (n_samples, total_ports).")
    if true_values.ndim != 2 or true_values.shape[1] != total_ports:
        raise ValueError("true_values must have shape (n_samples, total_ports).")
    if predicted_values.shape[0] != true_values.shape[0]:
        raise ValueError("predicted_values and true_values must contain the same number of samples.")

    n_samples = true_values.shape[0]
    row_idx = np.arange(n_samples)
    best_predicted_indices = np.argmax(predicted_values, axis=1)
    model_selected_values = true_values[row_idx, best_predicted_indices]

    if not combine_observed_and_predicted:
        return model_selected_values.astype(np.float32)

    # The combined OP definition compares the model-selected realized value
    # against the best physically observed value, then keeps the better option.
    observed_selected_values = np.max(true_values[:, observed_indices], axis=1)
    return np.maximum(observed_selected_values, model_selected_values).astype(np.float32)


def _compute_outage_metrics(
    selected_values: np.ndarray,
    threshold: float,
    snr_linear: float,
    prefix: str,
) -> dict[str, Any]:
    """Compute OP, count, and standard-deviation metrics for selected values.

    Args:
        selected_values (np.ndarray): One-dimensional array of realized values
            selected for OP evaluation. Values are compared against
            ``threshold / snr_linear``.
        threshold (float): Outage threshold numerator in linear scale.
        snr_linear (float): SNR denominator in linear scale. A value of zero is
            invalid because the outage cutoff is undefined.
        prefix (str): Column-name prefix for returned metrics. For example,
            ``"model_only"`` produces ``"model_only_op"``,
            ``"model_only_n_outage_samples"``, and
            ``"model_only_op_std"``.

    Returns:
        dict[str, Any]: Metrics containing outage probability, outage sample
            count, and population standard deviation of the per-sample outage
            indicator.

    Raises:
        ValueError: If ``selected_values`` is empty, not one-dimensional, or
            ``snr_linear`` is zero.

    Examples:
        >>> metrics = _compute_outage_metrics(np.array([0.1, 0.9]), 0.5, 1.0, "model")
        >>> metrics["model_op"]
        0.5
        >>> metrics["model_n_outage_samples"]
        1
    """
    if selected_values.ndim != 1:
        raise ValueError("selected_values must be one-dimensional.")
    if selected_values.shape[0] == 0:
        raise ValueError("selected_values is empty.")
    if snr_linear == 0.0:
        raise ValueError("snr_linear must be non-zero.")

    cutoff = threshold / snr_linear

    # Existing OP code treats outage as the complement of ``value > cutoff``.
    # Keeping ``<=`` here preserves that boundary behavior while exposing the
    # count behind the probability.
    outage_flags = selected_values <= cutoff
    outage_indicator = outage_flags.astype(np.float64)

    return {
        f"{prefix}_op": float(np.mean(outage_indicator)),
        f"{prefix}_n_outage_samples": int(np.sum(outage_flags)),
        f"{prefix}_op_std": float(np.std(outage_indicator)),
    }


def evaluate_op_per_regime(
    model: tf.keras.Model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    data: np.ndarray,
    num_observed_ports: int,
    total_ports: int = 100,
    threshold: float = 0.8,
    snr_linear: float = 1.0,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute OP metrics and prediction details for one dataset.

    Metrics produced:
    - ``ideal_op`` plus count and standard deviation.
    - ``observed_op`` plus count and standard deviation.
    - ``model_plus_observed_op`` plus count, standard deviation, and ideal gap percentage.
    - ``model_only_op`` plus count, standard deviation, and ideal gap percentage.

    Args:
        model (tf.keras.Model): Trained model producing full-port predictions.
        scaler_x (MinMaxScaler): Input scaler fitted during training.
        scaler_y (MinMaxScaler): Output scaler fitted during training.
        data (np.ndarray): Full-port matrix with shape
            ``(n_samples, total_ports)``.
        num_observed_ports (int): Number of uniformly spaced observed ports
            used by this compatibility wrapper and the model.
        total_ports (int): Total ports in each sample.
        threshold (float): Outage threshold numerator.
        snr_linear (float): SNR scaling denominator.
        batch_size (int): Prediction batch size.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            ``(results_df, predictions_df)`` where:
            - ``results_df`` has one summary row with OP metrics.
            - ``predictions_df`` has one row per sample with selected port and
              selected values.

    Raises:
        ValueError: If ``dataset`` is empty, has invalid shape, or
            ``snr_linear`` is zero.
    """
    results_by_threshold, predictions_df = evaluate_op_per_regime_for_thresholds(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        data=data,
        num_observed_ports=num_observed_ports,
        observed_indices=np.linspace(
            0,
            total_ports - 1,
            num_observed_ports,
            dtype=np.int64,
        ),
        total_ports=total_ports,
        thresholds=[threshold],
        snr_linear=snr_linear,
        batch_size=batch_size,
    )
    return results_by_threshold[float(threshold)], predictions_df


def evaluate_op_per_regime_for_thresholds(
    model: tf.keras.Model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    data: np.ndarray,
    num_observed_ports: int,
    observed_indices: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
    total_ports: int = 100,
    snr_linear: float = 1.0,
    batch_size: int = 64,
) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    """Compute OP metrics for multiple thresholds using one prediction pass.

    Args:
        model (tf.keras.Model): Trained model producing full-port predictions.
            The model is called exactly once for the provided ``data``.
        scaler_x (MinMaxScaler): Input scaler fitted during training. It is
            used once before prediction to transform observed-port values.
        scaler_y (MinMaxScaler): Output scaler fitted during training. It is
            used once after prediction to invert model outputs to port values.
        data (np.ndarray): Full-port matrix with shape
            ``(n_samples, total_ports)``. It must contain at least one sample.
        num_observed_ports (int): Number of observed ports used by the model.
            It must match the fitted input scaler and the model input width.
        observed_indices (np.ndarray): One-dimensional zero-based physical port
            indices used as both the model input and observed-only reference.
            Its length must equal ``num_observed_ports``.
        thresholds (list[float] | tuple[float, ...]): Outage threshold values
            in linear scale. Each value produces a separate results DataFrame.
            More thresholds add only metric-computation work; they do not add
            model inference work.
        total_ports (int): Total ports in each sample. Defaults to ``100`` and
            must match ``data.shape[1]`` and model output width.
        snr_linear (float): SNR scaling denominator. A zero value is invalid
            because the outage cutoff ``threshold / snr_linear`` is undefined.
        batch_size (int): Prediction batch size passed to ``model.predict``.
            Larger values can improve GPU throughput and memory use increases
            with the number of samples processed per batch.

    Returns:
        tuple[dict[float, pd.DataFrame], pd.DataFrame]: ``(results_by_threshold,
            predictions_df)``. ``results_by_threshold`` maps each threshold to
            the same one-row results schema produced by
            ``evaluate_op_per_regime``. ``predictions_df`` is shared by all
            thresholds because selected ports do not depend on threshold.

    Raises:
        ValueError: If ``data`` is empty or has invalid shape, ``thresholds`` is
            empty, or ``snr_linear`` is zero.

    Examples:
        >>> class DummyModel:
        ...     def predict(self, x, batch_size=64, verbose=0):
        ...         return np.ones((x.shape[0], 2), dtype=np.float32)
        >>> scaler = MinMaxScaler().fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        >>> data = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        >>> results, predictions = evaluate_op_per_regime_for_thresholds(
        ...     DummyModel(), scaler, scaler, data, 2, np.array([0, 1]),
        ...     [0.5], total_ports=2
        ... )
        >>> list(results)
        [0.5]
        >>> predictions.shape[0]
        2
    """
    if data.ndim != 2 or data.shape[1] != total_ports:
        raise ValueError("dataset must have shape (n_samples, total_ports).")
    if data.shape[0] == 0:
        raise ValueError("dataset is empty.")
    if not thresholds:
        raise ValueError("thresholds must contain at least one threshold.")

    observed_indices, predicted_values, predicted_ports, predicted_selected_values, true_selected_values = (
        predict_regime(
            model=model,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            true_values=data,
            num_observed_ports=num_observed_ports,
            observed_indices=observed_indices,
            total_ports=total_ports,
            batch_size=batch_size,
        )
    )

    ideal_selected_values = np.max(data, axis=1).astype(np.float32)
    observed_selected_values = np.max(data[:, observed_indices], axis=1).astype(np.float32)
    model_plus_observed_selected_values = _compute_selected_values(
        observed_indices=observed_indices,
        predicted_values=predicted_values,
        true_values=data,
        total_ports=total_ports,
        combine_observed_and_predicted=True,
    )
    model_only_selected_values = _compute_selected_values(
        observed_indices=observed_indices,
        predicted_values=predicted_values,
        true_values=data,
        total_ports=total_ports,
        combine_observed_and_predicted=False,
    )

    results_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        ideal_metrics = _compute_outage_metrics(
            selected_values=ideal_selected_values,
            threshold=threshold_value,
            snr_linear=snr_linear,
            prefix="ideal",
        )
        observed_metrics = _compute_outage_metrics(
            selected_values=observed_selected_values,
            threshold=threshold_value,
            snr_linear=snr_linear,
            prefix="observed",
        )
        model_plus_observed_metrics = _compute_outage_metrics(
            selected_values=model_plus_observed_selected_values,
            threshold=threshold_value,
            snr_linear=snr_linear,
            prefix="model_plus_observed",
        )
        model_only_metrics = _compute_outage_metrics(
            selected_values=model_only_selected_values,
            threshold=threshold_value,
            snr_linear=snr_linear,
            prefix="model_only",
        )

        results_rows: list[dict[str, Any]] = [
            {
                "num_samples": int(data.shape[0]),
                "n_observed_ports": int(num_observed_ports),
                **ideal_metrics,
                **observed_metrics,
                **model_plus_observed_metrics,
                "model_plus_observed_op_gap_from_ideal_pct": float(
                    100.0 * (model_plus_observed_metrics["model_plus_observed_op"] - ideal_metrics["ideal_op"])
                ),
                **model_only_metrics,
                "model_only_op_gap_from_ideal_pct": float(
                    100.0 * (model_only_metrics["model_only_op"] - ideal_metrics["ideal_op"])
                ),
            }
        ]
        results_by_threshold[threshold_value] = pd.DataFrame(results_rows)

    prediction_rows: list[dict[str, Any]] = []
    for i in range(data.shape[0]):
        prediction_rows.append(
            {
                "sample_index": int(i),
                "selected_port": int(predicted_ports[i]),
                "predicted_selected_value": float(predicted_selected_values[i]),
                "true_selected_value": float(true_selected_values[i]),
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)
    return results_by_threshold, predictions_df


def save_op_outputs(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    max_samples: int | None = 10,
) -> dict[str, Path]:
    """Save OP metrics and per-sample predictions to CSV files.

    Args:
        results_df (pd.DataFrame): Per-regime OP metrics table.
        predictions_df (pd.DataFrame): Per-sample prediction details table.
        output_dir (str | Path): Output directory for CSV files.
        max_samples (int | None): Maximum number of samples to save in predictions
            table. If ``None``, save all samples. Default is 10.

    Returns:
        dict[str, Path]: Paths of written files with keys
            ``results_table`` and ``predictions_table``.

    Raises:
        OSError: If file writing fails.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "op_results_by_regime.csv"
    predictions_path = output_path / "model_predictions_by_sample.csv"

    results_df.to_csv(results_path, index=False)
    
    # Limit predictions to max_samples if specified
    predictions_to_save = predictions_df if max_samples is None else predictions_df.head(max_samples)
    predictions_to_save.to_csv(predictions_path, index=False)

    return {
        "results_table": results_path,
        "predictions_table": predictions_path,
    }
