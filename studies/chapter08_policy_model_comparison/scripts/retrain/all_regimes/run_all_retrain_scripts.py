"""Run every NAS-selected port-reading retraining script sequentially."""

# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))

import hashlib
import json
import subprocess
import sys
from pathlib import Path


CURRENT_FILE = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/retrain/all_regimes/run_all_retrain_scripts.py'
RETRAIN_DIR = _COMPANION_ROOT / 'studies/chapter08_policy_model_comparison/scripts/retrain/all_regimes'
POLICIES = ("uni", "mar", "gc")
BUDGETS = (3, 4, 5, 6, 7)
TRAIN_SEEDS = (0, 1, 2, 3, 4)
TRAINING_SCRIPTS = tuple(
    RETRAIN_DIR / f"train_{policy}_{budget}_ports.py"
    for policy in POLICIES
    for budget in BUDGETS
)
RUN_ROOT = RETRAIN_DIR / "runs"


def case_run_dir(policy: str, budget: int, train_seed: int) -> Path:
    """Return the stable output directory for one scheduled case.

    Args:
        policy (str): Reading-policy identifier embedded in the directory name.
        budget (int): Number of observed ports embedded in the directory name.
        train_seed (int): Training seed used as the parent directory suffix.

    Returns:
        Path: Expected case directory under the ignored retraining run root.

    Raises:
        None.
    """
    return (
        RUN_ROOT
        / f"seed_{train_seed}"
        / f"train_nas_top1_{policy}_{budget}_ports"
    )


def is_complete_case(run_dir: Path) -> bool:
    """Return whether a case has its marker and all required final artifacts.

    Args:
        run_dir (Path): Case directory to inspect. Partial directories and
            zero-byte artifacts remain incomplete so their script is rerun.

    Returns:
        bool: ``True`` only when the marker and complete artifact set exist.

    Raises:
        None.
    """
    required = (
        run_dir / "completion.json",
        run_dir / "model_training" / "args" / "run_contract.json",
        run_dir / "model_training" / "args" / "mobilenet_config.json",
        run_dir / "model_training" / "args" / "model_stats.txt",
        run_dir / "model_training" / "model" / "model.keras",
        run_dir / "model_training" / "model" / "compact_decision_model.keras",
        run_dir / "model_training" / "model" / "explorer_model.keras",
        run_dir / "model_training" / "tflite" / "model_int8.tflite",
        run_dir / "model_training" / "scaler" / "scaler_X.pkl",
        run_dir / "model_training" / "scaler" / "scaler_y.pkl",
        run_dir / "model_training" / "history" / "history.csv",
        run_dir / "model_training" / "history" / "selection_history.csv",
        run_dir / "model_training" / "checkpoint" / "best.weights.h5",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        marker = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
        contract_text = (
            run_dir / "model_training" / "args" / "run_contract.json"
        ).read_text(encoding="utf-8")
        contract = json.loads(contract_text)
    except (OSError, json.JSONDecodeError):
        return False
    encoded_contract = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        marker.get("status") == "complete"
        and marker.get("contract_sha256")
        == hashlib.sha256(encoded_contract).hexdigest()
    )


def run_all_retraining_scripts() -> None:
    """Execute the 15 policy-budget scripts and report 75-case progress.

    Scripts run sequentially to avoid concurrent GPU use. A non-zero child
    status stops the campaign. Rerunning this launcher skips scripts whose five
    cases are complete, while incomplete child cases resume from their own
    checkpoints.

    Args:
        None.

    Returns:
        None: The function returns after all cases are complete.

    Raises:
        FileNotFoundError: If any required child script is missing.
        subprocess.CalledProcessError: If a child training script fails.
        RuntimeError: If a child exits successfully without completing all five
            required training cases.

    Examples:
        Execute from the repository root with the project interpreter:

        >>> callable(run_all_retraining_scripts)
        True
    """
    missing_scripts = [path for path in TRAINING_SCRIPTS if not path.is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing retraining scripts: {missing_scripts}")

    all_cases = [
        case_run_dir(policy, budget, train_seed)
        for policy in POLICIES
        for budget in BUDGETS
        for train_seed in TRAIN_SEEDS
    ]
    total_cases = len(all_cases)
    finished_cases = sum(is_complete_case(path) for path in all_cases)
    for script_path in TRAINING_SCRIPTS:
        name_parts = script_path.stem.split("_")
        policy = name_parts[1]
        budget = int(name_parts[2])
        script_cases = [
            case_run_dir(policy, budget, train_seed)
            for train_seed in TRAIN_SEEDS
        ]
        complete_count = sum(is_complete_case(path) for path in script_cases)
        print(
            f"[PROGRESS] script={script_path.name}, "
            f"complete={complete_count}/{len(TRAIN_SEEDS)}, "
            f"remaining_total={total_cases - finished_cases}/{total_cases}",
            flush=True,
        )
        if complete_count == len(TRAIN_SEEDS):
            print(f"[SKIP] All cases complete for {script_path.name}.", flush=True)
            continue

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=RETRAIN_DIR,
            check=True,
        )
        remaining_cases = [
            path for path in script_cases if not is_complete_case(path)
        ]
        if remaining_cases:
            raise RuntimeError(
                f"{script_path.name} exited without completing: {remaining_cases}"
            )
        finished_cases += len(TRAIN_SEEDS) - complete_count

    print(f"[DONE] Accounted for {total_cases} retraining cases.", flush=True)


if __name__ == "__main__":
    run_all_retraining_scripts()
