For your **Raspberry Pi Pico W + INA219**, assuming the shunt resistor marked **R100 / 0.100 Ω**, keep this in the code:

```cpp
constexpr int32_t kIna219ShuntMicroOhms = 100000;
```

That means:

```text
100000 µΩ = 0.1 Ω
```

If your resistor is literally **100 Ω**, do **not** use it for Pico power measurement. That would create a huge voltage drop and the Pico W probably would not run correctly.

## Recommended connection

Use the INA219 as a **high-side current sensor**, placed in series with the Pico W supply line.

### INA219 logic/I2C pins

| INA219 pin | Pico W pin | Purpose                 |
| ---------- | ---------- | ----------------------- |
| VCC        | 3V3 OUT    | Powers the INA219 logic |
| GND        | GND        | Common ground           |
| SDA        | GP4        | I2C data                |
| SCL        | GP5        | I2C clock               |

This matches the code:

```cpp
constexpr uint kIna219SdaPin = 4;
constexpr uint kIna219SclPin = 5;
```

## Power measurement wiring

Use an external 5 V supply through the INA219 into **VSYS**:

```text
External 5 V +  -> INA219 VIN+
INA219 VIN-    -> Pico W VSYS
External 5 V -  -> Pico W GND
```

So the full connection is:

```text
External 5 V +  ──> INA219 VIN+
INA219 VIN-    ──> Pico W VSYS

External GND   ──> Pico W GND
INA219 GND     ──> Pico W GND

INA219 VCC     ──> Pico W 3V3 OUT
INA219 SDA     ──> Pico W GP4
INA219 SCL     ──> Pico W GP5
```

The INA219 measures the voltage drop across the shunt between **VIN+** and **VIN-**, and measures the bus voltage from **VIN-** to ground. This is the intended high-side measurement topology. ([Texas Instruments][1])

## Very important USB warning

If you connect the Pico W to your PC through a normal USB cable while also powering VSYS through the INA219, USB **VBUS may bypass the INA219**. Then your measured current will be wrong.

For valid energy measurement, use one of these:

1. Use a **data-only USB cable**.
2. Cut or isolate the USB 5 V wire.
3. Use UART serial instead of USB serial.
4. Use an external power analyzer setup where all Pico W current passes through the INA219.

For initial debugging, you may keep normal USB connected, but do **not** trust the energy result.

## Pin reference on Pico W

Use these Pico W pins:

```text
Pin 40: VBUS
Pin 39: VSYS
Pin 38: GND
Pin 36: 3V3 OUT
Pin 6:  GP4, I2C0 SDA
Pin 7:  GP5, I2C0 SCL
```

Power the board through **VSYS**, not through 3V3, unless you have a well-regulated 3.3 V supply and know exactly what you are doing. The Pico W has onboard power circuitry intended to accept input through VSYS. The Pico documentation identifies the Pico W power rails and GPIO functions. ([Raspberry Pi][2])

## Check before powering

Before connecting the Pico W:

1. Measure external supply voltage, should be around **5 V**.
2. Confirm polarity:

   ```text
   Supply + -> VIN+
   VIN- -> VSYS
   Supply - -> GND
   ```
3. Confirm INA219 VCC is connected to **3V3 OUT**, not 5 V.
4. Confirm all grounds are common.
5. Confirm your code uses:

   ```cpp
   constexpr int32_t kIna219ShuntMicroOhms = 100000;
   ```

## Expected behavior

If wired correctly, the serial output should show:

```text
ina219_status=ok
ina219_initial_bus_mV=...
ina219_initial_current_uA=...
ina219_initial_power_uW=...
```

If current is negative, VIN+ and VIN- are probably reversed.

## References

1. TI INA219 datasheet, high-side shunt and bus-voltage measurement topology. ([Texas Instruments][1])
2. Raspberry Pi Pico W documentation and pin/power reference. ([Raspberry Pi][2])

[1]: https://www.ti.com/lit/ds/symlink/ina219.pdf?utm_source=chatgpt.com "INA219 Zerø-Drift, Bidirectional Current/Power Monitor With I"
[2]: https://www.raspberrypi.com/?utm_source=chatgpt.com "Raspberry Pi"
