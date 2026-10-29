"""Train the validated MobileNetV2 safe selector on all 60/20/20 regimes."""

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
    build_all_regimes_case_specs,
    is_finished_training_run,
    read_required_train_seed,
    read_required_wrapper_run_root,
)


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/scripts/train/train_all_regimes_mobv2_alpha02.py'
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


TRAIN_SEED = read_required_train_seed()
WRAPPER_RUN_ROOT = read_required_wrapper_run_root()

VARIANT_NAME = "mobv2_alpha02"
EPOCHS = 100
BATCH_SIZE = 2048
DATA_SEED = 0
TOTAL_PORTS = 100
OBSERVED_PORTS_LIST = (3, 4, 5, 6, 7)
TRAIN_REGIME_LABELS = (
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
STEPS_PER_EXECUTION = "auto"
JIT_COMPILE = True
PRECISION_POLICY = "float32"
ENABLE_OP_DETERMINISM = False
LEARNING_RATE = 3e-4
SELECTION_SAMPLES_PER_REGIME = 5_000


def main() -> None:
    """Load the full split once and run the original five-case campaign."""
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
        regime_labels=TRAIN_REGIME_LABELS,
        samples_per_regime=None,
        verbose=True,
    )
    del y_test
    del train_parts
    del test_parts

    try:
        for case_spec in build_all_regimes_case_specs(
            run_root=WRAPPER_RUN_ROOT,
            train_seed=TRAIN_SEED,
            variant_name=VARIANT_NAME,
            observed_ports_list=OBSERVED_PORTS_LIST,
        ):
            if is_finished_training_run(case_spec.run_dir):
                print(f"[SKIP] Finished run already exists: {case_spec.run_dir}")
                continue

            print(
                "[RUN] "
                f"variant={VARIANT_NAME}, regimes=all, "
                f"observed_ports={case_spec.num_observed_ports}, "
                f"train_seed={TRAIN_SEED}"
            )
            run_safe_selector_training_case(
                variant_name=VARIANT_NAME,
                run_dir=case_spec.run_dir,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                data_seed=DATA_SEED,
                train_seed=TRAIN_SEED,
                num_observed_ports=case_spec.num_observed_ports,
                total_ports=TOTAL_PORTS,
                train_regime_labels=TRAIN_REGIME_LABELS,
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


if __name__ == "__main__":
    main()
