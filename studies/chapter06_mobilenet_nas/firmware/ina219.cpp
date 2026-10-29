#include "ina219.h"

#include <stdint.h>
#include <stdio.h>

#include "pico/stdlib.h"

namespace {

constexpr uint8_t kRegisterConfig = 0x00;
constexpr uint8_t kRegisterShuntVoltage = 0x01;
constexpr uint8_t kRegisterBusVoltage = 0x02;

constexpr uint16_t kConfig32V320mV12BitContinuous = 0x399F;
constexpr uint kI2cTransactionTimeoutUs = 100000;

static bool write_register_u16(
    i2c_inst_t* i2c,
    uint8_t address,
    uint8_t register_address,
    uint16_t value
) {
    uint8_t buffer[3] = {
        register_address,
        static_cast<uint8_t>((value >> 8) & 0xFF),
        static_cast<uint8_t>(value & 0xFF),
    };

    printf("ina219_stage=config_write_begin\n");
    fflush(stdout);

    // Bound the transaction so a locked bus reports a timeout over serial
    // instead of stopping all subsequent benchmark diagnostics.
    int written = i2c_write_timeout_us(
        i2c,
        address,
        buffer,
        sizeof(buffer),
        false,
        kI2cTransactionTimeoutUs
    );
    printf("ina219_config_write_result=%d\n", written);
    fflush(stdout);
    return written == static_cast<int>(sizeof(buffer));
}

static bool read_register_u16(
    i2c_inst_t* i2c,
    uint8_t address,
    uint8_t register_address,
    uint16_t* value
) {
    // Bound both halves of the repeated-start transaction so a disconnected
    // sensor or locked bus cannot freeze the benchmark during sampling.
    int written = i2c_write_timeout_us(
        i2c,
        address,
        &register_address,
        1,
        true,
        kI2cTransactionTimeoutUs
    );
    if (written != 1) {
        return false;
    }

    uint8_t buffer[2] = {0, 0};
    int read = i2c_read_timeout_us(
        i2c,
        address,
        buffer,
        sizeof(buffer),
        false,
        kI2cTransactionTimeoutUs
    );
    if (read != static_cast<int>(sizeof(buffer))) {
        return false;
    }

    *value = (static_cast<uint16_t>(buffer[0]) << 8) | static_cast<uint16_t>(buffer[1]);
    return true;
}

}  // namespace

bool ina219_init(
    i2c_inst_t* i2c,
    uint sda_pin,
    uint scl_pin,
    uint32_t baudrate_hz,
    uint8_t address
) {
    i2c_init(i2c, baudrate_hz);
    printf("ina219_stage=i2c_init_complete\n");
    fflush(stdout);

    gpio_set_function(sda_pin, GPIO_FUNC_I2C);
    gpio_set_function(scl_pin, GPIO_FUNC_I2C);
    gpio_pull_up(sda_pin);
    gpio_pull_up(scl_pin);
    printf("ina219_stage=gpio_setup_complete\n");
    fflush(stdout);

    sleep_ms(10);

    bool configured = write_register_u16(
        i2c,
        address,
        kRegisterConfig,
        kConfig32V320mV12BitContinuous
    );

    if (!configured) {
        return false;
    }

    sleep_ms(2);
    return true;
}

bool ina219_read_sample(
    i2c_inst_t* i2c,
    uint8_t address,
    int32_t shunt_micro_ohms,
    Ina219Sample* sample
) {
    if (shunt_micro_ohms <= 0 || sample == nullptr) {
        return false;
    }

    uint16_t shunt_raw_u16 = 0;
    uint16_t bus_raw = 0;

    if (!read_register_u16(i2c, address, kRegisterShuntVoltage, &shunt_raw_u16)) {
        return false;
    }

    if (!read_register_u16(i2c, address, kRegisterBusVoltage, &bus_raw)) {
        return false;
    }

    int16_t shunt_raw = static_cast<int16_t>(shunt_raw_u16);

    sample->shunt_uV = static_cast<int32_t>(shunt_raw) * 10;
    sample->bus_mV = static_cast<int32_t>(bus_raw >> 3) * 4;
    sample->current_uA = static_cast<int32_t>(
        (static_cast<int64_t>(sample->shunt_uV) * 1000000LL) /
        static_cast<int64_t>(shunt_micro_ohms)
    );
    sample->power_uW = static_cast<int32_t>(
        (static_cast<int64_t>(sample->bus_mV) *
        static_cast<int64_t>(sample->current_uA)) /
        1000LL
    );
    sample->conversion_ready = (bus_raw & 0x0002) != 0;
    sample->overflow = (bus_raw & 0x0001) != 0;

    return true;
}
