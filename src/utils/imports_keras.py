"""
Central imports module for the project.

Usage:
    from utils.imports_keras import *
"""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

RED = "\033[31m"
RESET = "\033[0m"

# Global toggle for TensorFlow/absl import-time log suppression.
ENABLE_TF_LOG_SUPPRESSION = True

from typing import *

# ————————————————————————————— STANDARD LIBRARY ————————————————————————————— #
import os, gc, shutil, sys, traceback, warnings
from pathlib import Path
import pickle
from contextlib import contextmanager, nullcontext


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Suppress native stderr writes while noisy libraries initialize.

    This context manager is intended for TensorFlow and absl import-time logs
    that write directly to process file descriptor 2 before Python logging and
    warnings filters can handle them. It assumes stdout must remain untouched so
    Keras model summaries, epoch progress, and project-level status prints keep
    their normal behavior.

    Args:
        None.

    Yields:
        None: Control returns to the caller while native stderr writes are
        redirected to the operating system null device.

    Returns:
        Iterator[None]: Generator iterator consumed by `contextmanager`; callers
        should use this function only as a `with` context.

    Raises:
        OSError: Raised if stderr cannot be duplicated, redirected, restored, or
        closed by the operating system.

    Examples:
        >>> with _suppress_native_stderr():
        ...     import tensorflow as tf
    """
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            # TensorFlow C++ logs bypass `sys.stderr`, so redirect the file
            # descriptor itself instead of using `contextlib.redirect_stderr`.
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        # Restore stderr before closing the duplicate so later tracebacks and
        # runtime errors remain visible to the caller.
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


# —————————————————————————————— DATA PROCESSING ————————————————————————————— #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ———————————————————————— Scikit-learn preprocessing ———————————————————————— #
from sklearn.model_selection import train_test_split

# ———————————————————————————————— TENSORFLOW ———————————————————————————————— #
# TensorFlow uses native logging for some INFO/WARNING messages. The environment
# variable handles logs emitted after import, while the stderr context handles
# earlier absl messages emitted before Python-level logging exists.
if ENABLE_TF_LOG_SUPPRESSION:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

_tf_import_context = _suppress_native_stderr() if ENABLE_TF_LOG_SUPPRESSION else nullcontext()

with _tf_import_context:
    import tensorflow as tf

    # ——————————————————————————————————— Keras —————————————————————————————————— #
    from tensorflow.keras import (
        Model,
        callbacks,
        initializers,
        activations,
        layers,
        losses,
        metrics,
        mixed_precision,
        optimizers,
        regularizers,
    )
    from tensorflow.keras.backend import clear_session
    from tensorflow.keras.utils import set_random_seed

if ENABLE_TF_LOG_SUPPRESSION:
    tf.get_logger().setLevel("ERROR")

# —————————————————————————————————— Optuna —————————————————————————————————— #
import optuna, optunahub
from optuna.integration import KerasPruningCallback
from optuna.terminator import report_cross_validation_scores

# ——————————————————————————————— Scikit-Learn ——————————————————————————————— #
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)

# —————————————————————————————————— Araras —————————————————————————————————— #
# Model builders
from araras.ml.model.builders.dnn import *

# Callbacks
from araras.ml.optuna.callbacks import (
    NanLossPrunerOptuna,
    get_callbacks_study,
    StopIfKeepBeingPruned,
    ImprovementStagnation,
    StopWhenNoValueImprovement,
)
from araras.ml.model.callbacks import get_callbacks_model

# Araras utilities
from araras.ml.model.hyperparams import KParams
from araras.ml.model.tools import punish_model, print_tensor_mem, validate_steps_per_execution
from araras.ml.optuna.utils import (
    get_remaining_trials,
    cleanup_non_top_trials,
    rename_top_k_files,
    get_top_trials,
    save_top_k_trials,
    init_study_dirs,
    log_trial_error,
    run_study,
)
from araras.ml.optuna.analyzer import analyze_study
from araras.ml.optuna.model_tools import (
    set_user_attr_model_stats,
    prune_model_by_config,
    plot_model_param_distribution,
)
from araras.ml.model.stats import write_model_stats_to_file

# Plotting configuration
from araras.visualization.configs import config_plt

# Utility functions
from araras.utils.io import get_caller_stem, select_path
from araras.utils.system import get_gpu_info, log_resources
from araras.utils.misc import clear

# ————————————————————————————— PROJECT SPECIFIC ————————————————————————————— #
import scipy


# ———————————————————————————————————————————————————————————————————————————— #
# clear()
get_gpu_info()
