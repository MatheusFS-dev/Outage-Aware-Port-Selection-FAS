#pragma once

#include <stdint.h>

#include "hardware/i2c.h"

struct Ina219Sample {
    int32_t shunt_uV;
    int32_t bus_mV;
    int32_t current_uA;
    int32_t power_uW;
    bool conversion_ready;
    bool overflow;
};

bool ina219_init(
    i2c_inst_t* i2c,
    uint sda_pin,
    uint scl_pin,
    uint32_t baudrate_hz,
    uint8_t address
);

bool ina219_read_sample(
    i2c_inst_t* i2c,
    uint8_t address,
    int32_t shunt_micro_ohms,
    Ina219Sample* sample
);
