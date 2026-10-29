"""Measure empirical linear-SNR spatial correlation without neural training."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

ROOT = _COMPANION_ROOT
REGIMES = {
    'kappa0_mu1_m50': 'SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m50.0.mat',
    'kappa5_mu2_m2': 'SNR_events_W1.0_U1_N100_kappa5.0e+00_mu2.0_m2.0.mat',
}
BATCH_SIZE = 8192


def correlation_matrix(data, batch_size=BATCH_SIZE):
    """Compute Pearson correlation using stable centered batch moments.

    Args:
        data: Sample-by-port array of finite linear SNR values.
        batch_size: Positive number of rows processed at a time.

    Returns:
        Float64 port-by-port Pearson correlation matrix.

    Raises:
        ValueError: If shape, values, batch size or port variances are invalid.
    """
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('Expected at least two samples and two ports')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    count = 0
    mean = np.zeros(data.shape[1], dtype=np.float64)
    moment = np.zeros((data.shape[1], data.shape[1]), dtype=np.float64)
    for start in range(0, len(data), batch_size):
        block = np.array(data[start:start + batch_size], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError('SNR contains nonfinite values')
        block_mean = block.mean(axis=0)
        block -= block_mean
        size = len(block)
        delta = block_mean - mean
        total = count + size
        moment += block.T @ block + np.outer(delta, delta) * (count * size / total)
        mean += delta * (size / total)
        count = total
    variance = np.diag(moment)
    if np.any(variance <= 0):
        raise ValueError('Pearson correlation is undefined for constant ports')
    scale = np.sqrt(variance)
    result = np.clip(moment / np.outer(scale, scale), -1, 1)
    np.fill_diagonal(result, 1)
    return result


def separation_statistics(correlation):
    """Summarize unique port pairs at each integer separation.

    Args:
        correlation: Square port correlation matrix.

    Returns:
        List of dictionaries with separation, pair count, mean, population
        standard deviation, minimum and maximum across each upper diagonal.
        The spread describes positions, not sampling uncertainty.
    """
    rows = []
    for lag in range(len(correlation)):
        diagonal = np.diag(correlation, lag)
        rows.append({'separation_ports': lag, 'pairs': len(diagonal),
                     'mean': float(diagonal.mean()), 'std_across_pairs': float(diagonal.std()),
                     'min': float(diagonal.min()), 'max': float(diagonal.max())})
    return rows


def main():
    """Analyze both original MATLAB regimes and save numeric run artifacts.

    Raises:
        FileNotFoundError: If an original MATLAB source is unavailable.
        ValueError: If a source is malformed or has invalid SNR values.
    """
    sources = {label: ROOT / 'data/kappa-mu_shadowed' / name
               for label, name in REGIMES.items()}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'runs' / 'study1_spatial_pretest' / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = {'created_utc': stamp, 'python': platform.python_version(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'batch_size': BATCH_SIZE,
              'quantity': 'linear SNR', 'estimator': 'Pearson correlation across sample rows',
              'regimes': {}, 'status': 'running'}
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Output: {output}', flush=True)
    for label, source in sources.items():
        print(f'Loading {label}', flush=True)
        data = loadmat(source, variable_names=['SNR_events'])['SNR_events']
        if data.shape != (1000000, 100):
            raise ValueError(f'{source}: expected (1000000, 100), got {data.shape}')
        correlation = correlation_matrix(data)
        rows = separation_statistics(correlation)
        folder = output / label
        folder.mkdir()
        np.save(folder / 'correlation.npy', correlation)
        np.savetxt(folder / 'correlation.csv', correlation, delimiter=',')
        with (folder / 'by_separation.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        distances = np.abs(np.arange(100)[:, None] - np.arange(100))
        approximation = np.array([row['mean'] for row in rows])[distances]
        off_diagonal = ~np.eye(100, dtype=bool)
        residual = (correlation - approximation)[off_diagonal]
        config['regimes'][label] = {
            'source': str(source), 'source_bytes': source.stat().st_size,
            'source_mtime_ns': source.stat().st_mtime_ns,
            'shape': list(data.shape), 'dtype': str(data.dtype),
            'off_diagonal_lag_model_rmse': float(np.sqrt(np.mean(residual ** 2))),
            'off_diagonal_lag_model_max_abs_error': float(np.max(np.abs(residual))),
        }
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        del data
        print(f'Saved {label}', flush=True)
    config['status'] = 'complete'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Complete. These statistics assess spatial structure, not CNN superiority.', flush=True)


if __name__ == '__main__':
    main()
