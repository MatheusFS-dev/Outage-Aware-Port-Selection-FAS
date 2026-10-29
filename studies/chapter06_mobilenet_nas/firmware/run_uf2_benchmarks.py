
# Standalone companion paths, independent of the original checkout.
import sys as _companion_sys
from pathlib import Path as _CompanionPath
_COMPANION_ROOT = next(p for p in _CompanionPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _companion_path in (_COMPANION_ROOT, _COMPANION_ROOT / "src"):
    if str(_companion_path) not in _companion_sys.path:
        _companion_sys.path.insert(0, str(_companion_path))
import csv
import os
import re
import select
import subprocess
import termios
import time
import tty
from pathlib import Path


SOURCE_ROOT = _COMPANION_ROOT / 'src'
UF2_ROOT = _COMPANION_ROOT / 'studies/chapter06_mobilenet_nas/firmware/uf2_files/nas_pqt_mobv3_per_regime'
RESULT_ROOT = _COMPANION_ROOT / "studies/chapter06_mobilenet_nas/results/pico_tflm_benchmark_nas_pqt_mobv3_per_regime"
SERIAL_DIRECTORY = Path("/dev/serial/by-id")
SERIAL_PATTERN = "usb-Raspberry_Pi_Pico_*-if00"
BOOT_LABEL = "RPI-RP2"
DEVICE_TIMEOUT_SECONDS = 20.0
BENCHMARK_TIMEOUT_SECONDS = 300.0
FATAL_MARKERS = (
    "schema_version_mismatch",
    "operator_resolver_setup_failed",
    "allocate_tensors_failed",
    "warmup_invoke_failed",
    "benchmark_invoke_failed",
    "power_measurement_invoke_failed",
    "unsupported_input_type=",
)
FILENAME_PATTERN = re.compile(
    r"^(?:(easy|medium|hard)_)?"
    r"pico_tflm_benchmark_nas_pqt_mobv3_regime_"
    r"(kappa\d+_mu\d+_m\d+)_([3-7])_ports_224\.uf2$"
)


def list_firmware_files() -> list[Path]:
    """List the complete per-regime UF2 benchmark set in deterministic order.

    Returns:
        list[Path]: The 70 UF2 paths ordered first by port-count folder and
        then lexicographically by filename.

    Raises:
        RuntimeError: If the firmware set does not contain exactly 70 files,
        five port folders, or 14 firmware files in each port folder.

    Examples:
        Calling ``list_firmware_files()`` returns files from ``3_ports``
        before files from ``4_ports``.
    """
    firmware_files: list[Path] = []

    # Validate the expected 14-regime matrix before touching the connected
    # board so a partial input set cannot produce a misleading test campaign.
    for ports in range(3, 8):
        port_files = sorted((UF2_ROOT / f"{ports}_ports").glob("*.uf2"))
        if len(port_files) != 14:
            raise RuntimeError(
                f"Expected 14 UF2 files for {ports} ports, found {len(port_files)}."
            )
        firmware_files.extend(port_files)

    if len(firmware_files) != 70:
        raise RuntimeError(f"Expected 70 UF2 files, found {len(firmware_files)}.")
    return firmware_files


def parse_firmware_name(firmware_path: Path) -> tuple[str, str, int]:
    """Extract representative class, regime, and port count from a UF2 name.

    Args:
        firmware_path (Path): UF2 firmware path following the generated
            per-regime filename convention. Representative filenames may
            begin with ``easy_``, ``medium_``, or ``hard_``; other regimes
            have no representative prefix.

    Returns:
        tuple[str, str, int]: Representative class as ``easy``, ``medium``,
        ``hard``, or an empty string; KMS regime label; and observed port
        count in the inclusive range 3--7.

    Raises:
        ValueError: If the filename does not follow the required convention.

    Examples:
        ``easy_...regime_kappa5_mu5_m50_3_ports_224.uf2`` produces
        ``("easy", "kappa5_mu5_m50", 3)``.
    """
    match = FILENAME_PATTERN.fullmatch(firmware_path.name)
    if match is None:
        raise ValueError(f"Unexpected UF2 filename: {firmware_path.name}")
    representative, regime, ports = match.groups()
    return representative or "", regime, int(ports)


def find_serial_path() -> Path:
    """Find the stable USB serial path for the single connected Pico.

    Returns:
        Path: The sole Raspberry Pi Pico USB-serial symbolic link under
        ``/dev/serial/by-id``.

    Raises:
        RuntimeError: If zero or multiple Pico USB-serial devices are present.
    """
    serial_paths = sorted(SERIAL_DIRECTORY.glob(SERIAL_PATTERN))
    if len(serial_paths) != 1:
        raise RuntimeError(
            f"Expected one Pico serial device, found {len(serial_paths)}."
        )
    return serial_paths[0]


def wait_for_path(path: Path, should_exist: bool, timeout_seconds: float) -> None:
    """Wait until a device path reaches the required existence state.

    Args:
        path (Path): Device or symbolic-link path to monitor.
        should_exist (bool): If ``True``, wait for the path to appear. If
            ``False``, wait for it to disappear.
        timeout_seconds (float): Maximum wait duration in seconds; must be
            positive. This function only polls filesystem state and has no
            logging, file-write, randomness, or device-placement side effects.

    Returns:
        None: Returns when the requested path state is observed.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
        TimeoutError: If the path does not reach the requested state before
            the timeout expires.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    deadline = time.monotonic() + timeout_seconds
    while path.exists() != should_exist:
        if time.monotonic() >= deadline:
            state = "appear" if should_exist else "disappear"
            raise TimeoutError(f"Timed out waiting for {path} to {state}.")
        time.sleep(0.1)


def find_boot_device() -> Path:
    """Find the single block device carrying the Pico BOOTSEL volume label.

    Returns:
        Path: Absolute block-device path labeled ``RPI-RP2``.

    Raises:
        RuntimeError: If ``lsblk`` fails or if zero or multiple matching block
            devices are present.
    """
    result = subprocess.run(
        ["lsblk", "-nrpo", "NAME,LABEL"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = [
        Path(parts[0])
        for line in result.stdout.splitlines()
        if len(parts := line.split()) == 2 and parts[1] == BOOT_LABEL
    ]
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one {BOOT_LABEL} block device, found {len(devices)}."
        )
    return devices[0]


def find_or_mount_boot_volume(device_path: Path) -> Path:
    """Return the mount point for the Pico BOOTSEL volume, mounting if needed.

    Args:
        device_path (Path): Exact BOOTSEL block-device path returned by
            ``find_boot_device``. Mounting uses the active user's UDisks
            policy and writes no repository files.

    Returns:
        Path: Existing mount directory for the BOOTSEL volume.

    Raises:
        RuntimeError: If the device has multiple mount points, UDisks cannot
            mount it, or no mount point exists after the mount request.
    """
    result = subprocess.run(
        ["lsblk", "-nrpo", "MOUNTPOINTS", str(device_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    mount_points = [Path(line) for line in result.stdout.splitlines() if line]
    if len(mount_points) > 1:
        raise RuntimeError(f"Multiple mount points found for {device_path}.")

    if not mount_points:
        subprocess.run(
            ["udisksctl", "mount", "-b", str(device_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["lsblk", "-nrpo", "MOUNTPOINTS", str(device_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        mount_points = [Path(line) for line in result.stdout.splitlines() if line]

    if len(mount_points) != 1 or not mount_points[0].is_dir():
        raise RuntimeError(f"Could not resolve the mount point for {device_path}.")
    return mount_points[0]


def flash_firmware(firmware_path: Path, serial_path: Path) -> None:
    """Reset the connected Pico to BOOTSEL and flash one UF2 firmware.

    Args:
        firmware_path (Path): Existing UF2 file to copy to the connected Pico.
            The operation replaces the firmware currently stored on the board
            and causes the board to reboot. It does not modify the source UF2.
        serial_path (Path): Stable USB serial symbolic link for the target Pico.
            The path disappears during BOOTSEL and must return after flashing.

    Returns:
        None: Returns after the stable USB serial path reappears.

    Raises:
        FileNotFoundError: If ``firmware_path`` or the configured serial device
            does not exist before flashing.
        RuntimeError: If BOOTSEL device discovery or mounting is ambiguous.
        subprocess.CalledProcessError: If the reset or UF2 copy command fails.
        TimeoutError: If USB device transitions do not finish within the
            configured device timeout.
    """
    if not firmware_path.is_file():
        raise FileNotFoundError(f"Missing UF2 firmware: {firmware_path}")
    if not serial_path.exists():
        raise FileNotFoundError(f"Missing Pico serial device: {serial_path}")

    # The Pico SDK USB-stdio interface treats 1200 baud as an explicit request
    # to reboot into BOOTSEL, avoiding a physical BOOTSEL-button cycle.
    subprocess.run(["stty", "-F", str(serial_path), "1200"], check=True)
    wait_for_path(serial_path, should_exist=False, timeout_seconds=DEVICE_TIMEOUT_SECONDS)

    boot_label_path = Path("/dev/disk/by-label") / BOOT_LABEL
    wait_for_path(boot_label_path, should_exist=True, timeout_seconds=DEVICE_TIMEOUT_SECONDS)
    boot_device = find_boot_device()
    mount_path = find_or_mount_boot_volume(boot_device)

    subprocess.run(["cp", str(firmware_path), str(mount_path)], check=True)
    wait_for_path(boot_label_path, should_exist=False, timeout_seconds=DEVICE_TIMEOUT_SECONDS)
    wait_for_path(serial_path, should_exist=True, timeout_seconds=DEVICE_TIMEOUT_SECONDS)


def capture_serial_output(serial_path: Path, timeout_seconds: float) -> tuple[list[str], str, str]:
    """Capture one benchmark record from the configured Pico serial device.

    Args:
        serial_path (Path): Stable USB serial symbolic link for the target Pico.
            The path must exist throughout capture.
        timeout_seconds (float): Maximum capture duration in seconds; must be
            positive. Capture ends at ``benchmark_status=complete``, at a
            known fatal firmware marker, or at the timeout. Serial bytes are
            read in raw mode without writing commands to the Pico.

    Returns:
        tuple[list[str], str, str]: Captured non-empty lines, terminal status
        as ``complete``, ``failed``, or ``timeout``, and an explicit failure
        reason. The reason is empty only for a completed benchmark.

    Raises:
        FileNotFoundError: If the configured serial path is absent.
        ValueError: If ``timeout_seconds`` is not positive.
        OSError: If the serial device cannot be opened, configured, or read.
        UnicodeDecodeError: If the firmware emits non-UTF-8 output.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if not serial_path.exists():
        raise FileNotFoundError(f"Missing Pico serial device: {serial_path}")

    lines: list[str] = []
    pending = bytearray()
    deadline = time.monotonic() + timeout_seconds
    file_descriptor = os.open(serial_path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    original_attributes = termios.tcgetattr(file_descriptor)

    try:
        tty.setraw(file_descriptor)
        while time.monotonic() < deadline:
            readable, _, _ = select.select([file_descriptor], [], [], 0.5)
            if not readable:
                continue

            chunk = os.read(file_descriptor, 4096)
            if not chunk:
                continue
            pending.extend(chunk)

            while b"\n" in pending:
                encoded_line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                line = encoded_line.rstrip(b"\r").decode("utf-8")
                if not line:
                    continue
                lines.append(line)

                if line == "benchmark_status=complete":
                    return lines, "complete", ""
                if any(marker in line for marker in FATAL_MARKERS):
                    return lines, "failed", line
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSANOW, original_attributes)
        os.close(file_descriptor)

    return lines, "timeout", f"No terminal benchmark status within {timeout_seconds:.0f} seconds."


def parse_key_values(lines: list[str]) -> dict[str, str]:
    """Parse machine-readable benchmark fields from captured serial lines.

    Args:
        lines (list[str]): Captured serial lines. Lines without exactly one
            leading field name followed by ``=`` remain only in the raw log.
            Repeated fields use their final value in the consolidated CSV.

    Returns:
        dict[str, str]: Mapping of lowercase field names to their final values.

    Raises:
        ValueError: If ``lines`` contains a non-string item.

    Examples:
        ``["runs=100", "latency_mean_us=25"]`` produces a mapping with
        ``runs`` and ``latency_mean_us`` entries.
    """
    fields: dict[str, str] = {}
    for line in lines:
        if not isinstance(line, str):
            raise ValueError("Every captured line must be a string.")
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            fields[key] = value
    return fields


def validate_completed_result(fields: dict[str, str]) -> str:
    """Validate sensor and energy fields for a completed benchmark.

    Args:
        fields (dict[str, str]): Final parsed benchmark fields. Validation
            requires successful INA219 initialization, at least one accepted
            power sample, and a numeric energy result. It does not alter logs
            or board state.

    Returns:
        str: Empty string when the completed record is valid, otherwise an
        explicit validation failure reason.

    Raises:
        ValueError: If ``power_sample_runs`` exists but is not an integer.
    """
    if fields.get("ina219_status") != "ok":
        return f"ina219_status={fields.get('ina219_status', 'missing')}"
    if "power_sample_runs" not in fields:
        return "power_sample_runs=missing"
    if int(fields["power_sample_runs"]) <= 0:
        return f"power_sample_runs={fields['power_sample_runs']}"
    energy_value = fields.get("energy_mean_nJ", "")
    if not re.fullmatch(r"\d+", energy_value):
        return f"energy_mean_nJ={energy_value or 'missing'}"
    return ""


def write_raw_log(log_path: Path, lines: list[str]) -> None:
    """Write the exact normalized serial record for one firmware.

    Args:
        log_path (Path): Destination Markdown path. Parent directories are
            created when absent, and an existing file at this exact path is
            overwritten by the current hardware run.
        lines (list[str]): Captured non-empty serial lines in device order.

    Returns:
        None: The log is persisted as UTF-8 text with one line per record.

    Raises:
        OSError: If the directory or log file cannot be created or written.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, str]]) -> Path:
    """Write a consolidated CSV containing metadata and parsed measurements.

    Args:
        rows (list[dict[str, str]]): One row per attempted firmware. Each row
            contains firmware metadata, terminal status, failure reason, log
            path, and any captured machine-readable benchmark fields.

    Returns:
        Path: Path to the written ``benchmark_results.csv`` file.

    Raises:
        ValueError: If ``rows`` is empty.
        OSError: If the result directory or CSV file cannot be written.
    """
    if not rows:
        raise ValueError("Cannot write an empty benchmark summary.")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_ROOT / "benchmark_results.csv"
    fixed_fields = [
        "firmware",
        "ports",
        "regime",
        "representative",
        "status",
        "failure_reason",
        "log_path",
    ]
    measurement_fields = sorted(
        {key for row in rows for key in row if key not in fixed_fields}
    )
    fieldnames = fixed_fields + measurement_fields

    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def run_benchmarks() -> None:
    """Flash and measure the complete 70-firmware per-regime Pico campaign.

    The campaign is deterministic and has no selectable modes: every UF2 is
    tested once in port-folder and filename order. Each attempt replaces the
    Pico firmware, writes one raw serial log, and updates the consolidated CSV.
    Explicit firmware failures and timeouts are recorded before continuing;
    USB flashing or device-discovery failures stop the campaign because later
    measurements would not have a trustworthy target device.

    Returns:
        None: Progress is printed to stdout and results are written beneath
        ``pico_tflm_benchmark_nas_pqt_mobv3_per_regime``.

    Raises:
        FileNotFoundError: If required firmware or device paths are absent.
        RuntimeError: If the firmware matrix or connected BOOTSEL device is
            ambiguous.
        OSError: If flashing, serial capture, or result writing fails.
        subprocess.CalledProcessError: If a required host command fails.
    """
    firmware_files = list_firmware_files()
    serial_path = find_serial_path()
    rows: list[dict[str, str]] = []

    for index, firmware_path in enumerate(firmware_files, start=1):
        representative, regime, ports = parse_firmware_name(firmware_path)
        relative_log_path = Path(f"{ports}_ports") / f"{firmware_path.stem}.md"
        log_path = RESULT_ROOT / relative_log_path
        print(f"[{index}/70] flashing {firmware_path.name}", flush=True)

        flash_firmware(firmware_path, serial_path)
        lines, status, failure_reason = capture_serial_output(
            serial_path,
            BENCHMARK_TIMEOUT_SECONDS,
        )
        fields = parse_key_values(lines)

        if status == "complete":
            validation_failure = validate_completed_result(fields)
            if validation_failure:
                status = "failed"
                failure_reason = validation_failure

        write_raw_log(log_path, lines)
        row = {
            "firmware": firmware_path.name,
            "ports": str(ports),
            "regime": regime,
            "representative": representative,
            "status": status,
            "failure_reason": failure_reason,
            "log_path": str(relative_log_path),
            **fields,
        }
        rows.append(row)
        write_summary(rows)
        print(f"[{index}/70] {status}: {failure_reason or 'benchmark_status=complete'}", flush=True)

    summary_path = write_summary(rows)
    complete_count = sum(row["status"] == "complete" for row in rows)
    failed_count = len(rows) - complete_count
    print(
        f"Campaign finished: complete={complete_count} failed={failed_count} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    run_benchmarks()
