# Raspberry Pi Pico W TensorFlow Lite Micro Benchmark with INA219 Energy Measurement

This project runs a TensorFlow Lite Micro model on a Raspberry Pi Pico W and measures:

- Inference latency over repeated runs.
- Core-0 inference occupancy during the latency measurement window.
- RP2040 reported and hardware-measured system clock frequency, plus benchmark core usage.
- Tensor arena memory usage.
- Model flash size.
- Input and output tensor sizes.
- Approximate inference energy using an INA219 current and voltage sensor.

This is a **dummy inference benchmark**. It does not use real input data, scalers, NumPy, TensorFlow Python, pickle files, or dataset reconstruction. The goal is to verify that a `.tflite` model can run on the Pico W and to collect basic embedded execution metrics.

## 1. Measurement model

Latency and power are measured in separate phases. The latency phase runs
`interpreter.Invoke()` repeatedly without INA219 I2C transactions between
invocations. The power phase runs continuous inference for a fixed load window,
then reads one INA219 sample. This avoids treating two asynchronous sensor
register reads as a synchronized per-inference waveform.

Energy is estimated from the aggregate means:

```text
energy_mean_nJ = mean_continuous_inference_power_uW * mean_latency_us / 1000
```

This is a board-level aggregate approximation. It is useful for quick
comparative experiments, but it is not the same as integrating a synchronized
high-rate current waveform during each inference.

For publication-grade energy measurements, validate the result with a power analyzer, oscilloscope plus shunt resistor, or a high-rate external logger.

## 2. Hardware required

Minimum hardware:

- Raspberry Pi Pico W.
- INA219 current and voltage sensor module.
- External power supply for the Pico W measurement path.
- USB cable for programming and serial output.
- Breadboard and jumper wires.

Recommended instrumentation:

- USB data-only cable or modified cable with USB VBUS disconnected.
- Oscilloscope or external power analyzer for validation.
- Known-value shunt resistor module. Most INA219 breakout boards use 0.1 ohm.

## 3. Critical wiring warning

Do not power the Pico W through USB while trying to measure its current through the INA219 on VSYS, unless USB VBUS is isolated.

If USB VBUS is still connected, part or all of the board current can bypass the INA219. That makes the energy estimate invalid.

For meaningful board-level energy measurement, power the Pico W through the INA219 path and use USB only for data.

## 4. Default wiring

The default code assumes:

```cpp
constexpr uint8_t kIna219Address = 0x40;
constexpr uint kIna219SdaPin = 4;
constexpr uint kIna219SclPin = 5;
constexpr uint32_t kIna219I2cBaudrateHz = 100000;
constexpr int32_t kIna219ShuntMicroOhms = 100000;
```

Default I2C wiring:

```text
INA219 VCC -> Pico W 3V3
INA219 GND -> Pico W GND
INA219 SDA -> Pico W GP4
INA219 SCL -> Pico W GP5
```

Default power measurement wiring:

```text
External supply positive -> INA219 VIN+
INA219 VIN- -> Pico W VSYS
External supply ground -> Pico W GND
```

This measures board-level current flowing into VSYS. It does not isolate only the RP2040 core energy.

## 5. Repository files

```text
CMakeLists.txt
README.md
benchmark_metrics.cpp
benchmark_metrics.h
ina219.cpp
ina219.h
main.cpp
make_model_data.py
model.tflite
generate_per_regime_uf2.py
run_uf2_benchmarks.py
tflm_model_transform.py
```

File roles:

- `main.cpp`, TensorFlow Lite Micro setup, dummy input generation, repeated inference, latency statistics, memory statistics, and INA219 energy estimate.
- `benchmark_metrics.cpp` and `benchmark_metrics.h`, unit conversions and percentile indexing used by the benchmark.
- `ina219.cpp`, minimal INA219 I2C driver using bus voltage and shunt voltage registers.
- `ina219.h`, INA219 sample structure and driver declarations.
- `make_model_data.py`, converts `model.tflite` into C/C++ source files.
- `model.tflite`, fixed-batch deployment model embedded by `make_model_data.py`.
- `CMakeLists.txt`, Pico SDK build configuration.
- `tflm_model_transform.py`, rewrites model operations that are not compatible with the Pico TensorFlow Lite Micro runtime.
- `generate_per_regime_uf2.py`, transforms and builds every per-regime model into a separate UF2 file.
- `run_uf2_benchmarks.py`, flashes every generated UF2, captures its serial output, and writes the campaign results.
- `uf2_files/`, generated UF2 files grouped by port count.

Generated files:

```text
model_data.cc
model_data.h
build/
uf2_files/
```

Do not edit `model_data.cc` manually. Regenerate it whenever the `.tflite` model changes.

## 6. Install system dependencies on Ubuntu

Install the basic build tools:

```bash
sudo apt update
sudo apt install -y \
    git \
    cmake \
    build-essential \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    libstdc++-arm-none-eabi-newlib \
    python3 \
    python3-pip \
    unzip \
    minicom
```

Optional tools:

```bash
sudo apt install -y picocom screen
```

Only one serial monitor is needed. This README uses `minicom`.

## 7. Create the Pico development directory

```bash
mkdir -p "$HOME/pico"
cd "$HOME/pico"
```

## 8. Install the Raspberry Pi Pico SDK

Clone the SDK:

```bash
cd "$HOME/pico"
git clone https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init
```

Set the SDK path for the current terminal:

```bash
export PICO_SDK_PATH="$HOME/pico/pico-sdk"
```

Make it persistent for Bash:

```bash
echo 'export PICO_SDK_PATH="$HOME/pico/pico-sdk"' >> "$HOME/.bashrc"
```

Make it persistent for Zsh:

```bash
echo 'export PICO_SDK_PATH="$HOME/pico/pico-sdk"' >> "$HOME/.zshrc"
```

Restart the terminal or source the relevant shell file:

```bash
source "$HOME/.bashrc"
```

or:

```bash
source "$HOME/.zshrc"
```

## 9. Install the Pico TensorFlow Lite Micro port

Clone the Pico TFLM port:

```bash
cd "$HOME/pico"
git clone https://github.com/raspberrypi/pico-tflmicro.git
```

Set the path for the current terminal:

```bash
export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"
```

Make it persistent for Bash:

```bash
echo 'export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"' >> "$HOME/.bashrc"
```

Make it persistent for Zsh:

```bash
echo 'export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"' >> "$HOME/.zshrc"
```

Restart the terminal or source the relevant shell file.

## 10. Copy this project

Place this project under:

```text
$HOME/pico/pico_tflm_benchmark_ina219_template
```

Example:

```bash
cd "$HOME/pico"
unzip /path/to/pico_tflm_benchmark_ina219_template.zip
cd pico_tflm_benchmark_ina219_template
```

Confirm the expected files exist:

```bash
ls
```

Expected output includes:

```text
CMakeLists.txt
README.md
ina219.cpp
ina219.h
main.cpp
make_model_data.py
```

## 11. Add your TFLite model

Copy your model into the project folder and name it `model.tflite`:

```bash
cp /path/to/your_model.tflite ./model.tflite
```

Example using your MobileNetV1 NAS model path:

```bash
cp ../models/per_regime/nas/kappa0_mu1_m0/3_ports/top_1_trial_*.tflite ./model.tflite
```

Check the file exists:

```bash
ls -lh model.tflite
```

The included model is the seven-port NAS rank-1 trial 38 artifact specialized
to batch size 1. Its two `BROADCAST_TO` shape inputs are fixed to
`[1, 32, 32]` because Pico TFLM validates those shapes before executing the
model's dynamic shape subgraph. Its quantized `LOG` is decomposed into
`DEQUANTIZE`, float `LOG`, and `QUANTIZE` because Pico TFLM's `LOG` invocation
supports only float tensors. Model weights, quantization parameters, and input
and output tensors are unchanged. The deployment graph is 158552 bytes, 240
bytes larger than the canonical 158312-byte artifact.

## 12. Convert the model into C/C++ source

Run:

```bash
python3 make_model_data.py
```

Expected generated files:

```text
model_data.cc
model_data.h
```

Expected terminal output:

```text
Wrote model_data.h
Wrote model_data.cc
Model bytes: ...
```

This embeds the model into firmware flash. On microcontrollers, the model is usually compiled into the firmware rather than copied as a runtime file.

## 13. Configure benchmark constants

Open `main.cpp`.

The most important constants are:

```cpp
constexpr int kWarmupRuns = 5;
constexpr int kBenchmarkRuns = 100;
constexpr int kPowerSampleRuns = 20;
constexpr int kTensorArenaSize = 160 * 1024;

constexpr bool kEnableIna219 = true;
constexpr uint8_t kIna219Address = 0x40;
constexpr uint kIna219SdaPin = 4;
constexpr uint kIna219SclPin = 5;
constexpr uint32_t kIna219I2cBaudrateHz = 100000;
constexpr int32_t kIna219ShuntMicroOhms = 100000;
constexpr uint32_t kPowerLoadWindowUs = 5000;
constexpr uint32_t kApplicationPeriodUs = 0;
```

### 13.1 Warmup runs

Warmup runs are not timed:

```cpp
constexpr int kWarmupRuns = 5;
```

Use warmup to remove first-run effects from the latency statistics.

### 13.2 Benchmark runs

Timed runs:

```cpp
constexpr int kBenchmarkRuns = 100;
```

Increase this for more stable statistics.

### 13.3 Tensor arena size

TensorFlow Lite Micro needs a manually allocated tensor arena:

```cpp
constexpr int kTensorArenaSize = 160 * 1024;
```

If allocation fails, increase it carefully. The Pico W has limited SRAM, so this value cannot grow indefinitely.

### 13.4 INA219 shunt resistance

Default:

```cpp
constexpr int32_t kIna219ShuntMicroOhms = 100000;
```

This means:

```text
100000 micro-ohms = 0.1 ohm
```

Change this if your INA219 module uses another shunt value.

### 13.5 Power sampling

The firmware runs continuous inference for `kPowerLoadWindowUs` before each
INA219 sample. The default 5 ms window is longer than one complete 12-bit shunt
plus bus conversion cycle. Increase it if inference or sensor readings are
unstable.

`kPowerSampleRuns` controls the number of power-sampling attempts. Invalid
samples are rejected and are not retried; `power_sample_runs` reports the
accepted count.

### 13.6 Application CPU occupancy

If your application performs one inference every fixed period, set:

```cpp
constexpr uint32_t kApplicationPeriodUs = 100000;
```

For example, `100000` means one inference every 100 ms. The firmware then reports:

```text
application_cpu_core0_occupancy_percent_x100
```

The value is multiplied by 100. For example:

```text
250
```

means:

```text
2.50 percent duty cycle
```

## 14. Build the firmware

From the project folder:

```bash
cd "$HOME/pico/pico_tflm_benchmark_ina219_template"
export PICO_SDK_PATH="$HOME/pico/pico-sdk"
export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"
mkdir -p build
cd build
cmake -DPICO_BOARD=pico_w ..
make -j"$(nproc)"
```

Expected output file:

```text
pico_tflm_benchmark.uf2
```

Check:

```bash
ls -lh pico_tflm_benchmark.uf2
```

## 15. Flash the Pico W

1. Disconnect the Pico W.
2. Hold the `BOOTSEL` button.
3. Plug the Pico W into USB while holding `BOOTSEL`.
4. Release `BOOTSEL`.
5. A drive named `RPI-RP2` should appear.
6. Copy the UF2 file:

```bash
cp pico_tflm_benchmark.uf2 /media/$USER/RPI-RP2/
```

The Pico W will reboot automatically.

If your system mounts the board elsewhere, find it with:

```bash
lsblk
```

or:

```bash
find /media "$HOME" -maxdepth 3 -type d -name RPI-RP2 2>/dev/null
```

## 16. Open the serial monitor

After flashing, the firmware prints results over USB serial.

Find the serial device:

```bash
ls /dev/ttyACM*
```

Open it with:

```bash
minicom -D /dev/ttyACM0 -b 115200
```

Exit `minicom` with:

```text
Ctrl+A
X
```

If `/dev/ttyACM0` does not exist, reconnect the Pico W and try again.

## 17. Fix serial permission errors

If you get a permission error for `/dev/ttyACM0`, add your user to the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

Then log out and log back in.

Verify:

```bash
groups
```

The output should include:

```text
dialout
```

Or, you can try:

```bash
sudo chown -R $USER:$USER /media/$USER/RPI-RP2/
```

## 18. Expected serial output

Successful startup should print fields similar to:

```text
Pico W TFLM dummy benchmark with INA219
ina219_enabled=1
ina219_i2c_port=i2c0
ina219_address=0x40
ina219_sda_pin=4
ina219_scl_pin=5
ina219_i2c_baudrate_hz=100000
ina219_shunt_micro_ohms=100000
ina219_status=ok
ina219_initial_bus_mV=...
ina219_initial_shunt_uV=...
ina219_initial_current_uA=...
ina219_initial_power_uW=...
ina219_initial_conversion_ready=1
ina219_initial_overflow=0
model_flash_bytes=...
cpu_clock_reported_hz=...
cpu_clock_measured_khz=...
cpu_cores=2
cpu_benchmark_core=0
cpu_core1_used_by_benchmark=0
tensor_arena_configured_bytes=...
tensor_arena_used_bytes=...
benchmark_result_buffers_bytes=...
memory_scope=tflm_arena_and_benchmark_buffers_not_total_firmware_ram
input_bytes=...
output_bytes=...
input_shape=[...]
output_shape=[...]
runs=100
latency_mean_us=...
latency_std_us=...
latency_min_us=...
latency_p50_us=...
latency_p95_us=...
latency_max_us=...
cpu_core0_inference_occupancy_percent_x100=...
latency_measurement_elapsed_us=...
latency_total_invoke_us=...
cpu_occupancy_scope=invoke_time_over_latency_measurement_window
application_cpu_core0_occupancy_percent_x100=not_defined_set_kApplicationPeriodUs
power_continuous_inference_mean_uW=...
power_continuous_inference_std_uW=...
power_continuous_inference_min_uW=...
power_continuous_inference_p50_uW=...
power_continuous_inference_p95_uW=...
power_continuous_inference_max_uW=...
power_sample_runs=...
power_inference_count=...
power_load_window_us=5000
energy_mean_nJ=...
energy_method=mean_continuous_inference_power_times_mean_latency
energy_warning=aggregate_board_level_estimate_not_synchronized_waveform_integral
```

## 19. Interpreting results

### 19.1 Latency

The latency values are measured only around:

```cpp
interpreter.Invoke()
```

They exclude:

- INA219 I2C reads.
- Serial printing.
- Tensor allocation.
- Model loading.
- Dummy input filling.
- Warmup runs.

Report the mean, standard deviation, min, p50, p95, and max.

### 19.2 Memory

The relevant fields are:

```text
model_flash_bytes
tensor_arena_configured_bytes
tensor_arena_used_bytes
benchmark_result_buffers_bytes
input_bytes
output_bytes
```

Interpretation:

- `model_flash_bytes`, size of the embedded `.tflite` model.
- `tensor_arena_configured_bytes`, static arena reserved in SRAM.
- `tensor_arena_used_bytes`, arena actually used by TensorFlow Lite Micro after allocation.
- `benchmark_result_buffers_bytes`, static storage used by latency and power result arrays.
- `input_bytes`, model input tensor size.
- `output_bytes`, model output tensor size.

This does not include all firmware memory, stack usage, resolver code, or static
globals. After building, inspect static firmware sections with:

```bash
arm-none-eabi-size build/pico_tflm_benchmark.elf
```

The generated `build/pico_tflm_benchmark.elf.map` provides section and symbol
details. Neither static section sizes nor the runtime fields measure peak stack
usage; that requires a separate stack high-water-mark measurement.

### 19.3 Energy

The reported energy is an aggregate board-level inference estimate:

```text
energy_mean_nJ
```

It includes whatever the INA219 measurement path sees. Depending on wiring, that can include:

- RP2040.
- Flash access.
- Voltage regulator losses.
- Pico W board overhead.
- Wireless chip idle contribution, if powered.
- Any attached hardware powered through the measured rail.

It is not pure neural network arithmetic energy. It also does not provide a
per-run energy distribution because INA219 samples are not synchronized to
individual inference boundaries. The estimate uses total active board power;
it does not subtract idle power.

### 19.4 CPU usage

On the Pico W there is no Linux-like process CPU usage counter. The benchmark
instead measures how much of the latency loop wall time core 0 spends inside
`interpreter.Invoke()`:

```text
cpu_core0_inference_occupancy_percent_x100
```

The value is multiplied by 100. For example, `9850` means 98.50% of that
measurement window. This is measured rather than hard-coded. Core 1 is not used
by the benchmark.

For an application with a known inference period, the firmware also reports the
estimated core-0 occupancy:

```text
application_cpu_core0_occupancy = latency_us / application_period_us
```

## 20. Troubleshooting

### 20.1 CMake cannot find Pico SDK

Symptom:

```text
PICO_SDK_PATH is not set
```

Fix:

```bash
export PICO_SDK_PATH="$HOME/pico/pico-sdk"
```

Check:

```bash
ls "$PICO_SDK_PATH"
```

You should see files such as:

```text
CMakeLists.txt
external
src
tools
```

### 20.2 CMake cannot find pico-tflmicro

Symptom:

```text
add_subdirectory given source ... which is not an existing directory
```

Fix:

```bash
export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"
```

Check:

```bash
ls "$PICO_TFLMICRO_PATH"
```

### 20.3 `model_data.cc` missing

Symptom:

```text
Cannot find source file: model_data.cc
```

Fix:

```bash
python3 make_model_data.py
```

Make sure `model.tflite` exists before running it.

### 20.4 Tensor allocation fails

Symptom:

```text
allocate_tensors_failed increase_kTensorArenaSize_or_check_ops
```

Possible causes:

1. `kTensorArenaSize` is too small.
2. The model requires operators not supported by TensorFlow Lite Micro.
3. The model is too large for Pico W SRAM.
4. The firmware has too many static allocations.

First try increasing:

```cpp
constexpr int kTensorArenaSize = 180 * 1024;
```

If that still fails, reduce the model or inspect its operators.

### 20.5 Firmware builds but invoke fails

Symptom:

```text
benchmark_invoke_failed run=...
```

Possible causes:

1. Unsupported operator.
2. Tensor shape mismatch inside the model.
3. Model conversion issue.
4. TFLite Micro kernel limitation.

For a first diagnostic, test with a smaller known-good model.

### 20.6 INA219 not detected

Symptom:

```text
ina219_status=not_found_or_config_failed
```

Check:

1. SDA is connected to GP4.
2. SCL is connected to GP5.
3. INA219 VCC is connected to Pico W 3V3.
4. INA219 GND and Pico W GND are common.
5. The module address is really `0x40`.

Some modules allow address changes through solder jumpers. If needed, change:

```cpp
constexpr uint8_t kIna219Address = 0x40;
```

Possible alternatives are often `0x41`, `0x44`, or `0x45`, depending on the module.

### 20.7 Current readings are zero or too low

Most likely causes:

1. The Pico W is still being powered through USB.
2. The INA219 is not actually in series with the board supply.
3. VIN+ and VIN- are reversed.
4. The shunt value constant is wrong.
5. The measured load current is below the practical resolution of your setup.

### 20.8 Negative current

A negative current usually means the INA219 polarity is reversed.

Swap the power path direction:

```text
Supply positive -> VIN+
VIN- -> load
```

### 20.9 Energy readings vary strongly

Expected causes:

1. USB power path is still connected.
2. INA219 sampling is not synchronized with the real current waveform.
3. Inference is too short relative to INA219 conversion timing.
4. Serial output or board state changes affect power.
5. Wireless chip or regulator behavior adds background variation.

For better measurements, run many inferences in a block and measure the whole block externally. Then divide by the number of inferences.

## 21. Recommended reporting format

For a paper or lab report, report at least:

```text
Board: Raspberry Pi Pico W
MCU: RP2040
Model: <model name>
Quantization: INT8 PTQ or other
Input shape: <shape>
Output shape: <shape>
Model size: <model_flash_bytes> bytes
Tensor arena configured: <tensor_arena_configured_bytes> bytes
Tensor arena used: <tensor_arena_used_bytes> bytes
Benchmark result buffers: <benchmark_result_buffers_bytes> bytes
CPU clock reported: <cpu_clock_reported_hz> Hz
CPU clock measured: <cpu_clock_measured_khz> kHz
Benchmark core: core 0, core 1 unused
Warmup runs: <kWarmupRuns>
Benchmark runs: <kBenchmarkRuns>
Latency mean: <latency_mean_us> us
Latency std: <latency_std_us> us
Latency p50: <latency_p50_us> us
Latency p95: <latency_p95_us> us
CPU core-0 inference occupancy: <cpu_core0_inference_occupancy_percent_x100> / 100 percent
Energy method: INA219 continuous-inference mean power times mean latency
Energy mean: <energy_mean_nJ> nJ
Power path: external supply through INA219 into VSYS
USB VBUS isolated: yes or no
```

Also state clearly that the energy estimate is board-level unless your wiring and instrumentation isolate a narrower domain.

## 22. Clean rebuild

If the build directory becomes inconsistent:

```bash
cd "$HOME/pico/pico_tflm_benchmark_ina219_template"
rm -rf build
mkdir build
cd build
cmake -DPICO_BOARD=pico_w ..
make -j"$(nproc)"
```

## 23. Updating the model

Whenever the model changes:

```bash
cd "$HOME/pico/pico_tflm_benchmark_ina219_template"
cp /path/to/new_model.tflite ./model.tflite
python3 make_model_data.py
cd build
make -j"$(nproc)"
```

Then flash the new UF2 again.

## 24. Notes on model compatibility

TensorFlow Lite Micro is not desktop TensorFlow Lite. A `.tflite` model that runs on a PC with `tf.lite.Interpreter` may still fail on a microcontroller if:

- It uses unsupported operators.
- It requires too much tensor arena memory.
- It uses float operations that are too slow or memory-heavy.
- It contains shapes or kernels unsupported by the microcontroller runtime.

For Pico W deployment, strict INT8 models are usually the best first target.

## 25. Automated per-regime benchmark campaign

The automated campaign has two stages: generate all UF2 files on the host, then flash and benchmark them sequentially on one Pico W. Run both commands from the repository root.

### 25.1 Prerequisites

Before starting:

- Complete the SDK, TensorFlow Lite Micro, CMake, wiring, and serial-permission setup described above.
- Configure the template's `build/` directory successfully at least once. The generator performs incremental builds using the existing CMake configuration.
- Use a Python environment containing TensorFlow, FlatBuffers, and NumPy.
- Ensure that the host provides `stty`, `lsblk`, `udisksctl`, and `cp`. On Ubuntu, the `udisks2` package provides `udisksctl`.
- Provide exactly one `top_1_trial_*.tflite` model in each of the 14 regime directories for every port count from 3 through 7 under `studies/chapter06_mobilenet_nas/models/per_regime/nas/`.
- Connect exactly one Pico W. Do not keep `minicom`, `screen`, or another program attached to its serial port during the automated run.
- Keep USB VBUS isolated from the measured power path as described in Section 3.

### 25.2 Generate the UF2 files

Run:

```bash
python3 studies/chapter06_mobilenet_nas/firmware/generate_per_regime_uf2.py
```

For each model, the generator:

1. Rewrites dynamic broadcast shapes as constant batch-one shapes.
2. Runs `LOG` in float between dequantization and quantization operations.
3. Runs `GREATER` with float inputs to avoid unsupported INT8 multiplier scales.
4. Regenerates `model_data.cc` and `model_data.h`.
5. Builds `pico_tflm_benchmark.uf2` and copies it into the matching port folder.

The generator requires 14 regimes for each of the five port counts, producing 70 files under:

```text
uf2_files/nas_pqt_mobv3_per_regime/
├── 3_ports/
├── 4_ports/
├── 5_ports/
├── 6_ports/
└── 7_ports/
```

The dissertation's representative regimes have a filename prefix:

- `easy_` for `kappa5_mu5_m50`.
- `medium_` for `kappa5_mu2_m2`.
- `hard_` for `kappa5_mu2_m0`.

The generator restores the original `model.tflite`, `model_data.cc`, and `model_data.h` after the campaign build, including when generation fails.

### 25.3 Run the hardware benchmarks

Turn on the wired Pico W and run:

```bash
python3 studies/chapter06_mobilenet_nas/firmware/run_uf2_benchmarks.py
```

The runner first verifies that all 70 UF2 files exist. For each file, it then:

1. Requests the Pico SDK USB bootloader through the serial port at 1200 baud.
2. Waits for the `RPI-RP2` mass-storage device, mounts it when needed, and copies the UF2.
3. Waits for the serial device to return and captures the complete benchmark output.
4. Accepts the run only when `benchmark_status=complete`, `ina219_status=ok`, a positive `power_sample_runs` value, and a numeric `energy_mean_nJ` are present.
5. Saves the raw serial log and updates the summary CSV before continuing with the next firmware.

An individual firmware error or timeout is recorded and the campaign continues. A USB discovery or flashing error stops the campaign because the next model cannot be loaded reliably.

### 25.4 Results

The runner writes the consolidated table to:

```text
studies/chapter06_mobilenet_nas/results/pico_tflm_benchmark_nas_pqt_mobv3_per_regime/benchmark_results.csv
```

Raw serial logs are stored in the `3_ports/` through `7_ports/` subfolders beside that CSV. Rerunning the command starts a new in-memory result table and overwrites logs with matching filenames.

### 25.5 USB reset recovery

The automatic reset works only while the current firmware is responsive enough to service USB. If the board is blocked during model allocation or inference, hold `BOOTSEL` while reconnecting the Pico, manually copy one generated UF2 to the mounted `RPI-RP2` drive, wait for that benchmark to finish, and rerun the automated command.

## 26. References

- Raspberry Pi Pico C/C++ SDK documentation: https://www.raspberrypi.com/documentation/microcontrollers/c_sdk.html
- Raspberry Pi Pico SDK repository: https://github.com/raspberrypi/pico-sdk
- Raspberry Pi Pico TensorFlow Lite Micro port: https://github.com/raspberrypi/pico-tflmicro
- LiteRT for Microcontrollers build and convert guide: https://ai.google.dev/edge/litert/microcontrollers/build_convert
- TI INA219 datasheet: https://www.ti.com/lit/ds/symlink/ina219.pdf
