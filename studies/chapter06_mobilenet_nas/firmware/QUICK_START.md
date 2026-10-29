# Quick Start

## 1. Connect the INA219 and Pico W

Turn off the external power supply before connecting anything.

| From | To |
| --- | --- |
| External 5 V positive | INA219 VIN+ |
| INA219 VIN- | Pico W VSYS, physical pin 39 |
| External 5 V ground | Pico W GND, physical pin 38 |
| INA219 GND | Pico W GND, physical pin 38 |
| INA219 VCC | Pico W 3V3 OUT, physical pin 36 |
| INA219 SDA | Pico W GP4, physical pin 6 |
| INA219 SCL | Pico W GP5, physical pin 7 |

For energy measurements, connect the Pico W to the computer with a data-only USB cable or isolate USB VBUS.

## 2. Add the TFLite model

Run from the companion repository root:

```bash
cd studies/chapter06_mobilenet_nas/firmware
cp /absolute/path/to/model.tflite ./model.tflite
python3 make_model_data.py
ls -lh model.tflite model_data.cc model_data.h
```

## 3. Build

```bash
export PICO_SDK_PATH="$HOME/pico/pico-sdk"
export PICO_TFLMICRO_PATH="$HOME/pico/pico-tflmicro"
cmake -S . -B build -DPICO_BOARD=pico_w
cmake --build build --parallel "$(nproc)"
ls -lh build/pico_tflm_benchmark.uf2
```

## 4. Put the Pico W in BOOTSEL mode

1. Disconnect the Pico W USB cable.
2. Hold `BOOTSEL`.
3. Connect the USB cable while holding `BOOTSEL`.
4. Release `BOOTSEL`.
5. Run:

```bash
lsblk -f
```

## 5. Copy the firmware to the Pico W

```bash
cp build/pico_tflm_benchmark.uf2 "/media/$USER/RPI-RP2/"
sync
```

## 6. Open the serial output

```bash
ls -l /dev/ttyACM*
minicom -D /dev/ttyACM0 -b 115200
```

Reset the Pico W after `minicom` opens.

Exit `minicom` with `Ctrl+A`, then `X`.

## 7. Check the INA219 result

```text
ina219_stage=i2c_init_complete
ina219_stage=gpio_setup_complete
ina219_stage=config_write_begin
ina219_config_write_result=3
ina219_status=ok
benchmark_stage=tflm_init
```

I2C result codes:

```text
 3 = configuration write succeeded
-1 = device or address not acknowledged
-2 = transaction timed out
```
