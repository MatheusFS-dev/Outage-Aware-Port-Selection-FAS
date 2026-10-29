
# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))
import shutil
import subprocess
import sys
from pathlib import Path

from tflm_model_transform import prepare_tflm_deployment_model


TEMPLATE_DIR = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/firmware'
SOURCE_ROOT = TEMPLATE_DIR.parents[1]
MODEL_ROOT = TEMPLATE_DIR.parent / "models/per_regime/nas"
UF2_ROOT = TEMPLATE_DIR / "uf2_files" / "nas_pqt_mobv3_per_regime"
MODEL_PATH = TEMPLATE_DIR / "model.tflite"
MODEL_SOURCE_PATH = TEMPLATE_DIR / "model_data.cc"
MODEL_HEADER_PATH = TEMPLATE_DIR / "model_data.h"
BUILD_UF2_PATH = TEMPLATE_DIR / "build" / "pico_tflm_benchmark.uf2"
REPRESENTATIVE_PREFIXES = {
    "kappa5_mu5_m50": "easy_",
    "kappa5_mu2_m2": "medium_",
    "kappa5_mu2_m0": "hard_",
}


def list_selected_models() -> list[Path]:
    """List the complete selected per-regime TFLite model matrix.

    Returns:
        list[Path]: The 70 selected strict-INT8 TFLite paths ordered by port
        count, regime, and trial filename.

    Raises:
        RuntimeError: If any port count does not contain exactly 14 models or
            if the complete matrix does not contain exactly 70 models.

    Examples:
        Calling ``list_selected_models()`` returns all 3-port models before
        the first 4-port model.
    """
    selected_models: list[Path] = []
    regime_directories = sorted(path for path in MODEL_ROOT.iterdir() if path.is_dir())
    if len(regime_directories) != 14:
        raise RuntimeError(
            f"Expected 14 per-regime model directories, found {len(regime_directories)}."
        )

    for ports in range(3, 8):
        port_models: list[Path] = []
        for regime_directory in regime_directories:
            matches = sorted(
                (regime_directory / f"{ports}_ports").glob("top_1_trial_*.tflite")
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one selected model for {regime_directory.name} "
                    f"with {ports} ports, found {len(matches)}."
                )
            port_models.extend(matches)
        if len(port_models) != 14:
            raise RuntimeError(
                f"Expected 14 selected models for {ports} ports, found {len(port_models)}."
            )
        selected_models.extend(port_models)

    if len(selected_models) != 70:
        raise RuntimeError(f"Expected 70 selected models, found {len(selected_models)}.")
    return selected_models


def build_uf2_output_path(selected_model_path: Path) -> Path:
    """Build the required port-folder and representative-prefixed UF2 path.

    Args:
        selected_model_path (Path): Selected model stored as
            ``<regime>/<ports>_ports/top_1_trial_<n>.tflite`` under the fixed
            per-regime weights root.

    Returns:
        Path: Destination UF2 path beneath the matching ``3_ports`` through
        ``7_ports`` folder. Easy, medium, and hard representative regimes
        receive their required filename prefixes.

    Raises:
        ValueError: If the regime label or port directory does not follow the
            selected-model path convention.

    Examples:
        A 3-port ``kappa5_mu5_m50`` model maps to an ``easy_`` filename under
        the ``3_ports`` UF2 folder.
    """
    regime = selected_model_path.parent.parent.name
    ports_directory = selected_model_path.parent.name
    if not regime.startswith("kappa"):
        raise ValueError(f"Unexpected regime directory: {regime}")
    if ports_directory not in {f"{ports}_ports" for ports in range(3, 8)}:
        raise ValueError(f"Unexpected port directory: {ports_directory}")

    ports = int(ports_directory.removesuffix("_ports"))
    prefix = REPRESENTATIVE_PREFIXES.get(regime, "")
    filename = (
        f"{prefix}pico_tflm_benchmark_nas_pqt_mobv3_regime_"
        f"{regime}_{ports}_ports_224.uf2"
    )
    return UF2_ROOT / ports_directory / filename


def run_build_command(command: list[str]) -> None:
    """Run one firmware-generation command from the template directory.

    Args:
        command (list[str]): Non-empty executable and argument list. The
            command runs without a shell, captures normal output, inherits the
            current environment, and may update generated template/build files.

    Returns:
        None: Returns after a zero exit status.

    Raises:
        ValueError: If ``command`` is empty.
        subprocess.CalledProcessError: If the command exits unsuccessfully;
            captured stdout and stderr remain attached to the exception.
        OSError: If the executable cannot be started.
    """
    if not command:
        raise ValueError("command must not be empty.")
    subprocess.run(
        command,
        cwd=TEMPLATE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def generate_per_regime_uf2() -> None:
    """Transform, build, and save all 70 selected per-regime Pico firmwares.

    Every selected model receives the fixed batch-one BroadcastTo, float-LOG,
    and float-GREATER TFLM deployment transformations before embedding. The
    template's tracked model and generated C++ files are restored byte-for-byte
    afterward, and the build directory is rebuilt against that restored model.

    Returns:
        None: Writes 70 UF2 files under ``nas_pqt_mobv3_per_regime`` and prints
        deterministic progress to stdout.

    Raises:
        RuntimeError: If the selected model matrix or transformed model graph
            differs from the required structure, or if a built UF2 is absent.
        OSError: If model, generated source, build, or UF2 files cannot be read
            or written.
        subprocess.CalledProcessError: If model-data generation or firmware
            compilation fails.
    """
    selected_models = list_selected_models()
    original_files = {
        MODEL_PATH: MODEL_PATH.read_bytes(),
        MODEL_SOURCE_PATH: MODEL_SOURCE_PATH.read_bytes(),
        MODEL_HEADER_PATH: MODEL_HEADER_PATH.read_bytes(),
    }

    try:
        for index, selected_model_path in enumerate(selected_models, start=1):
            output_path = build_uf2_output_path(selected_model_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            transformed_model = prepare_tflm_deployment_model(
                selected_model_path.read_bytes()
            )
            MODEL_PATH.write_bytes(transformed_model)
            run_build_command([sys.executable, "make_model_data.py"])
            run_build_command(["cmake", "--build", "build", "--parallel", "4"])
            if not BUILD_UF2_PATH.is_file():
                raise RuntimeError(f"Missing built firmware: {BUILD_UF2_PATH}")
            shutil.copyfile(BUILD_UF2_PATH, output_path)
            print(f"[{index}/70] generated {output_path.name}", flush=True)
    finally:
        # Restoring the exact source bytes keeps batch generation from changing
        # the template's checked-in reference model or its generated C++ data.
        for path, original_bytes in original_files.items():
            path.write_bytes(original_bytes)
        run_build_command(["cmake", "--build", "build", "--parallel", "4"])


if __name__ == "__main__":
    generate_per_regime_uf2()
