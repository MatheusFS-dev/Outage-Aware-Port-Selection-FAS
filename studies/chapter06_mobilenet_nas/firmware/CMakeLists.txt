cmake_minimum_required(VERSION 3.13)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_tflm_benchmark C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_subdirectory($ENV{PICO_TFLMICRO_PATH} pico-tflmicro EXCLUDE_FROM_ALL)

add_executable(
    pico_tflm_benchmark
    main.cpp
    benchmark_metrics.cpp
    ina219.cpp
    model_data.cc
)

target_include_directories(
    pico_tflm_benchmark
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_compile_options(
    pico_tflm_benchmark
    PRIVATE
    -fno-rtti
    -fno-exceptions
    -fno-threadsafe-statics
)

target_compile_definitions(
    pico_tflm_benchmark
    PRIVATE
    PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS=10000
)

target_link_libraries(
    pico_tflm_benchmark
    pico_stdlib
    hardware_clocks
    hardware_i2c
    pico-tflmicro
    m
)

pico_enable_stdio_usb(pico_tflm_benchmark 1)
pico_enable_stdio_uart(pico_tflm_benchmark 0)

pico_add_extra_outputs(pico_tflm_benchmark)
