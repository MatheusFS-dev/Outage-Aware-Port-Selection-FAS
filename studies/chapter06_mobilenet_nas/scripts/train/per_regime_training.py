"""Shared entrypoint for per-regime MobileNet safe-selector campaigns."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import gc
import os
import sys
from pathlib import Path

from multi_seed_runner import (
    TRAIN_REGIME_LABELS,
    build_per_regime_case_specs,
    is_finished_training_run,
    read_required_train_seed,
    read_required_wrapper_run_root,
)


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/per_regime_training.py'
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = _COMPANION_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from studies.chapter06_mobilenet_nas.scripts.train.safe_selector import (
    run_safe_selector_training_case,
)
from utils.data import split_kms_60_20_20_cache


EPOCHS = 100
BATCH_SIZE = 64
DATA_SEED = 0
TOTAL_PORTS = 100
OBSERVED_PORTS_LIST = (3, 4, 5, 6, 7)
STEPS_PER_EXECUTION = "auto"
JIT_COMPILE = True
PRECISION_POLICY = "float32"
ENABLE_OP_DETERMINISM = False
LEARNING_RATE = 3e-4
SELECTION_SAMPLES_PER_REGIME = 5_000


def run_per_regime_campaign(variant_name: str) -> None:
    """Train one MobileNet variant separately on every configured regime."""
    train_seed = read_required_train_seed()
    wrapper_run_root = read_required_wrapper_run_root()

    for regime_label in TRAIN_REGIME_LABELS:
        (
            y_train,
            y_val,
            y_test,
            train_parts,
            val_parts,
            test_parts,
        ) = split_kms_60_20_20_cache(
            common_path=None,
            seed=DATA_SEED,
            regime_labels=(regime_label,),
            samples_per_regime=None,
            verbose=True,
        )
        del y_test
        del train_parts
        del test_parts

        try:
            case_specs = build_per_regime_case_specs(
                run_root=wrapper_run_root,
                train_seed=train_seed,
                variant_name=variant_name,
                regime_labels=(regime_label,),
                observed_ports_list=OBSERVED_PORTS_LIST,
            )
            for case_spec in case_specs:
                if is_finished_training_run(case_spec.run_dir):
                    print(
                        "[SKIP] Finished run already exists: "
                        f"{case_spec.run_dir}"
                    )
                    continue

                print(
                    "[RUN] "
                    f"variant={variant_name}, "
                    f"regime={regime_label}, "
                    f"observed_ports={case_spec.num_observed_ports}, "
                    f"train_seed={train_seed}"
                )
                run_safe_selector_training_case(
                    variant_name=variant_name,
                    run_dir=case_spec.run_dir,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    data_seed=DATA_SEED,
                    train_seed=train_seed,
                    num_observed_ports=case_spec.num_observed_ports,
                    total_ports=TOTAL_PORTS,
                    train_regime_labels=(regime_label,),
                    y_train=y_train,
                    y_val=y_val,
                    val_parts=val_parts,
                    steps_per_execution=STEPS_PER_EXECUTION,
                    show_summary=True,
                    jit_compile=JIT_COMPILE,
                    precision_policy=PRECISION_POLICY,
                    enable_op_determinism=ENABLE_OP_DETERMINISM,
                    learning_rate=LEARNING_RATE,
                    selection_samples_per_regime=(
                        SELECTION_SAMPLES_PER_REGIME
                    ),
                )
        finally:
            del y_train
            del y_val
            del val_parts
            gc.collect()
