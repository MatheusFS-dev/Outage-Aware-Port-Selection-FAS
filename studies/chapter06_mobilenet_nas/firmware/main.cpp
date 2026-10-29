#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/i2c.h"
#include "pico/stdlib.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "benchmark_metrics.h"
#include "ina219.h"
#include "model_data.h"

#define INA219_I2C_PORT i2c0

namespace {

constexpr int kWarmupRuns = 5;
constexpr int kBenchmarkRuns = 100;
constexpr int kPowerSampleRuns = 20;
constexpr int kTensorArenaSize = 224 * 1024;

constexpr bool kEnableIna219 = true;
constexpr uint8_t kIna219Address = 0x40;
constexpr uint kIna219SdaPin = 4;
constexpr uint kIna219SclPin = 5;
constexpr uint32_t kIna219I2cBaudrateHz = 100000;
constexpr int32_t kIna219ShuntMicroOhms = 100000;
constexpr uint32_t kPowerLoadWindowUs = 5000;
constexpr uint32_t kApplicationPeriodUs = 0;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];
uint32_t latencies_us[kBenchmarkRuns];
int32_t power_samples_uW[kPowerSampleRuns];

}  // namespace

static void print_tensor_shape(const char* name, const TfLiteTensor* tensor) {
    printf("%s_shape=[", name);
    for (int index = 0; index < tensor->dims->size; index++) {
        printf("%d", tensor->dims->data[index]);
        if (index + 1 < tensor->dims->size) {
            printf(",");
        }
    }
    printf("]\n");
}

static bool fill_dummy_input(TfLiteTensor* input) {
    if (input->type == kTfLiteInt8) {
        for (size_t index = 0; index < input->bytes; index++) {
            input->data.int8[index] = static_cast<int8_t>(input->params.zero_point);
        }
        return true;
    }

    if (input->type == kTfLiteUInt8) {
        for (size_t index = 0; index < input->bytes; index++) {
            input->data.uint8[index] = static_cast<uint8_t>(input->params.zero_point);
        }
        return true;
    }

    if (input->type == kTfLiteFloat32) {
        size_t count = input->bytes / sizeof(float);
        for (size_t index = 0; index < count; index++) {
            input->data.f[index] = 0.0f;
        }
        return true;
    }

    printf("unsupported_input_type=%d\n", static_cast<int>(input->type));
    return false;
}

static void sort_u32(uint32_t* values, int count) {
    for (int index = 1; index < count; index++) {
        uint32_t key = values[index];
        int previous = index - 1;

        while (previous >= 0 && values[previous] > key) {
            values[previous + 1] = values[previous];
            previous--;
        }

        values[previous + 1] = key;
    }
}

static void sort_i32(int32_t* values, int count) {
    for (int index = 1; index < count; index++) {
        int32_t key = values[index];
        int previous = index - 1;

        while (previous >= 0 && values[previous] > key) {
            values[previous + 1] = values[previous];
            previous--;
        }

        values[previous + 1] = key;
    }
}

static uint32_t print_latency_stats(
    const uint32_t* values,
    int count,
    uint64_t measurement_elapsed_us
) {
    uint64_t sum = 0;
    uint32_t min_value = values[0];
    uint32_t max_value = values[0];
    uint32_t sorted[kBenchmarkRuns];

    for (int index = 0; index < count; index++) {
        uint32_t value = values[index];
        sorted[index] = value;
        sum += value;

        if (value < min_value) {
            min_value = value;
        }

        if (value > max_value) {
            max_value = value;
        }
    }

    uint32_t mean = static_cast<uint32_t>(sum / count);

    uint64_t squared_error_sum = 0;
    for (int index = 0; index < count; index++) {
        int64_t error = static_cast<int64_t>(values[index]) - static_cast<int64_t>(mean);
        squared_error_sum += static_cast<uint64_t>(error * error);
    }

    uint32_t std = static_cast<uint32_t>(sqrt(static_cast<double>(squared_error_sum / count)));

    sort_u32(sorted, count);

    uint32_t p50 = sorted[nearest_rank_index(count, 50)];
    uint32_t p95 = sorted[nearest_rank_index(count, 95)];

    printf("runs=%d\n", count);
    printf("latency_mean_us=%lu\n", static_cast<unsigned long>(mean));
    printf("latency_std_us=%lu\n", static_cast<unsigned long>(std));
    printf("latency_min_us=%lu\n", static_cast<unsigned long>(min_value));
    printf("latency_p50_us=%lu\n", static_cast<unsigned long>(p50));
    printf("latency_p95_us=%lu\n", static_cast<unsigned long>(p95));
    printf("latency_max_us=%lu\n", static_cast<unsigned long>(max_value));

    uint32_t occupancy_percent_x100 = ratio_percent_x100(
        sum,
        measurement_elapsed_us
    );
    printf(
        "cpu_core0_inference_occupancy_percent_x100=%lu\n",
        static_cast<unsigned long>(occupancy_percent_x100)
    );
    printf(
        "latency_measurement_elapsed_us=%llu\n",
        static_cast<unsigned long long>(measurement_elapsed_us)
    );
    printf(
        "latency_total_invoke_us=%llu\n",
        static_cast<unsigned long long>(sum)
    );
    printf("cpu_occupancy_scope=invoke_time_over_latency_measurement_window\n");

    if constexpr (kApplicationPeriodUs > 0) {
        uint32_t application_occupancy_percent_x100 = ratio_percent_x100(
            mean,
            kApplicationPeriodUs
        );
        printf(
            "application_cpu_core0_occupancy_percent_x100=%lu\n",
            static_cast<unsigned long>(application_occupancy_percent_x100)
        );
    } else {
        printf("application_cpu_core0_occupancy_percent_x100=not_defined_set_kApplicationPeriodUs\n");
    }

    return mean;
}

static int32_t print_power_stats(const char* prefix, const int32_t* values, int count) {
    int64_t sum = 0;
    int32_t min_value = values[0];
    int32_t max_value = values[0];
    int32_t sorted[kBenchmarkRuns];

    for (int index = 0; index < count; index++) {
        int32_t value = values[index];
        sorted[index] = value;
        sum += value;

        if (value < min_value) {
            min_value = value;
        }

        if (value > max_value) {
            max_value = value;
        }
    }

    int32_t mean = static_cast<int32_t>(sum / count);

    int64_t squared_error_sum = 0;
    for (int index = 0; index < count; index++) {
        int64_t error = static_cast<int64_t>(values[index]) - static_cast<int64_t>(mean);
        squared_error_sum += error * error;
    }

    int32_t std = static_cast<int32_t>(sqrt(static_cast<double>(squared_error_sum / count)));

    sort_i32(sorted, count);

    printf("%s_mean_uW=%ld\n", prefix, static_cast<long>(mean));
    printf("%s_std_uW=%ld\n", prefix, static_cast<long>(std));
    printf("%s_min_uW=%ld\n", prefix, static_cast<long>(min_value));
    printf("%s_p50_uW=%ld\n", prefix, static_cast<long>(sorted[nearest_rank_index(count, 50)]));
    printf("%s_p95_uW=%ld\n", prefix, static_cast<long>(sorted[nearest_rank_index(count, 95)]));
    printf("%s_max_uW=%ld\n", prefix, static_cast<long>(max_value));

    return mean;
}

static bool read_ina219_sample_or_report(const char* label, Ina219Sample* sample) {
    bool read_ok = ina219_read_sample(
        INA219_I2C_PORT,
        kIna219Address,
        kIna219ShuntMicroOhms,
        sample
    );

    if (!read_ok) {
        printf("%s=ina219_read_failed\n", label);
        return false;
    }

    printf("%s_bus_mV=%ld\n", label, static_cast<long>(sample->bus_mV));
    printf("%s_shunt_uV=%ld\n", label, static_cast<long>(sample->shunt_uV));
    printf("%s_current_uA=%ld\n", label, static_cast<long>(sample->current_uA));
    printf("%s_power_uW=%ld\n", label, static_cast<long>(sample->power_uW));
    printf("%s_conversion_ready=%d\n", label, sample->conversion_ready ? 1 : 0);
    printf("%s_overflow=%d\n", label, sample->overflow ? 1 : 0);

    return true;
}

int main() {
    stdio_init_all();
    sleep_ms(2500);

    printf("\nPico W TFLM dummy benchmark with INA219\n");
    printf("boot_status=stdio_ready\n");
    fflush(stdout);

    bool ina219_ok = false;

    if (kEnableIna219) {
        printf("benchmark_stage=ina219_init\n");
        fflush(stdout);

        ina219_ok = ina219_init(
            INA219_I2C_PORT,
            kIna219SdaPin,
            kIna219SclPin,
            kIna219I2cBaudrateHz,
            kIna219Address
        );

        printf("ina219_enabled=1\n");
        printf("ina219_i2c_port=i2c0\n");
        printf("ina219_address=0x%02x\n", kIna219Address);
        printf("ina219_sda_pin=%u\n", kIna219SdaPin);
        printf("ina219_scl_pin=%u\n", kIna219SclPin);
        printf("ina219_i2c_baudrate_hz=%lu\n", static_cast<unsigned long>(kIna219I2cBaudrateHz));
        printf("ina219_shunt_micro_ohms=%ld\n", static_cast<long>(kIna219ShuntMicroOhms));

        if (!ina219_ok) {
            printf("ina219_status=not_found_or_config_failed\n");
        } else {
            Ina219Sample initial_sample = {};
            printf("ina219_status=ok\n");
            read_ina219_sample_or_report("ina219_initial", &initial_sample);
        }
    } else {
        printf("ina219_enabled=0\n");
    }
    fflush(stdout);

    printf("benchmark_stage=tflm_init\n");
    fflush(stdout);
    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("schema_version_mismatch model=%d runtime=%d\n", model->version(), TFLITE_SCHEMA_VERSION);
        while (true) {
            sleep_ms(1000);
        }
    }

    printf("benchmark_stage=operator_resolver_setup\n");
    fflush(stdout);
    static tflite::MicroMutableOpResolver<24> resolver;
    TfLiteStatus resolver_status = kTfLiteOk;

    // Register only the operators encoded in model.tflite because this Pico
    // TFLM port does not provide AllOpsResolver.
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddAdd();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddBroadcastTo();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddConv2D();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddDepthwiseConv2D();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddDequantize();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddFullyConnected();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddGather();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddGreater();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddHardSwish();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddLog();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddLogistic();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddMean();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddMul();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddNeg();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddPack();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddQuantize();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddRelu();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddReshape();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddRsqrt();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddSelectV2();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddShape();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddSquare();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddStridedSlice();
    }
    if (resolver_status == kTfLiteOk) {
        resolver_status = resolver.AddSub();
    }

    if (resolver_status != kTfLiteOk) {
        printf("operator_resolver_setup_failed check_model_ops\n");
        while (true) {
            sleep_ms(1000);
        }
    }

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    printf("benchmark_stage=allocate_tensors\n");
    fflush(stdout);
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("allocate_tensors_failed increase_kTensorArenaSize_or_check_ops\n");
        while (true) {
            sleep_ms(1000);
        }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("model_flash_bytes=%d\n", g_model_len);
    printf(
        "cpu_clock_reported_hz=%lu\n",
        static_cast<unsigned long>(clock_get_hz(clk_sys))
    );
    printf(
        "cpu_clock_measured_khz=%lu\n",
        static_cast<unsigned long>(frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_SYS))
    );
    printf("cpu_cores=2\n");
    printf("cpu_benchmark_core=0\n");
    printf("cpu_core1_used_by_benchmark=0\n");
    printf("tensor_arena_configured_bytes=%d\n", kTensorArenaSize);
    printf("tensor_arena_used_bytes=%u\n", static_cast<unsigned int>(interpreter.arena_used_bytes()));
    printf(
        "benchmark_result_buffers_bytes=%u\n",
        static_cast<unsigned int>(sizeof(latencies_us) + sizeof(power_samples_uW))
    );
    printf("memory_scope=tflm_arena_and_benchmark_buffers_not_total_firmware_ram\n");
    printf("input_bytes=%u\n", static_cast<unsigned int>(input->bytes));
    printf("output_bytes=%u\n", static_cast<unsigned int>(output->bytes));
    print_tensor_shape("input", input);
    print_tensor_shape("output", output);

    if (!fill_dummy_input(input)) {
        while (true) {
            sleep_ms(1000);
        }
    }

    printf("benchmark_stage=warmup\n");
    fflush(stdout);
    for (int run = 0; run < kWarmupRuns; run++) {
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("warmup_invoke_failed run=%d\n", run);
            while (true) {
                sleep_ms(1000);
            }
        }
    }

    // Keep sensor traffic outside this loop so elapsed occupancy and individual
    // latency describe repeated inference rather than I2C instrumentation.
    printf("benchmark_stage=latency_measurement\n");
    fflush(stdout);
    uint64_t latency_measurement_start_us = time_us_64();
    for (int run = 0; run < kBenchmarkRuns; run++) {
        uint64_t start_us = time_us_64();
        TfLiteStatus invoke_status = interpreter.Invoke();
        uint64_t end_us = time_us_64();

        if (invoke_status != kTfLiteOk) {
            printf("benchmark_invoke_failed run=%d\n", run);
            while (true) {
                sleep_ms(1000);
            }
        }

        latencies_us[run] = static_cast<uint32_t>(end_us - start_us);
    }
    uint64_t latency_measurement_elapsed_us = (
        time_us_64() - latency_measurement_start_us
    );

    uint32_t latency_mean_us = print_latency_stats(
        latencies_us,
        kBenchmarkRuns,
        latency_measurement_elapsed_us
    );

    int power_sample_count = 0;
    uint32_t power_inference_count = 0;

    if (ina219_ok) {
        printf("benchmark_stage=power_measurement\n");
        fflush(stdout);

        for (int run = 0; run < kPowerSampleRuns; run++) {
            // Sustain inference longer than one complete INA219 shunt-plus-bus
            // conversion so the sampled register represents active load.
            uint64_t load_start_us = time_us_64();
            do {
                if (interpreter.Invoke() != kTfLiteOk) {
                    printf("power_measurement_invoke_failed run=%d\n", run);
                    while (true) {
                        sleep_ms(1000);
                    }
                }
                power_inference_count++;
            } while (time_us_64() - load_start_us < kPowerLoadWindowUs);

            Ina219Sample sample = {};
            bool sample_ok = ina219_read_sample(
                INA219_I2C_PORT,
                kIna219Address,
                kIna219ShuntMicroOhms,
                &sample
            );

            // Only accepted samples contribute to aggregate board power; a
            // rejected sample is reported but not silently replaced or retried.
            bool sample_valid = (
                sample_ok &&
                sample.conversion_ready &&
                !sample.overflow &&
                sample.power_uW > 0
            );
            if (sample_valid) {
                power_samples_uW[power_sample_count] = sample.power_uW;
                power_sample_count++;
            } else {
                printf(
                    "ina219_sample_rejected run=%d read_ok=%d conversion_ready=%d overflow=%d power_uW=%ld\n",
                    run,
                    sample_ok ? 1 : 0,
                    sample.conversion_ready ? 1 : 0,
                    sample.overflow ? 1 : 0,
                    static_cast<long>(sample.power_uW)
                );
            }
        }
    }

    if (power_sample_count > 0) {
        int32_t power_mean_uW = print_power_stats(
            "power_continuous_inference",
            power_samples_uW,
            power_sample_count
        );
        int64_t energy_mean_nJ = power_latency_to_energy_nj(
            power_mean_uW,
            latency_mean_us
        );
        printf("power_sample_runs=%d\n", power_sample_count);
        printf("power_inference_count=%lu\n", static_cast<unsigned long>(power_inference_count));
        printf("power_load_window_us=%lu\n", static_cast<unsigned long>(kPowerLoadWindowUs));
        printf("energy_mean_nJ=%lld\n", static_cast<long long>(energy_mean_nJ));
        printf("energy_method=mean_continuous_inference_power_times_mean_latency\n");
        printf("energy_warning=aggregate_board_level_estimate_not_synchronized_waveform_integral\n");
    } else {
        printf("power_sample_runs=0\n");
        printf("energy_mean_nJ=not_measured_check_ina219_wiring_or_disable_usb_power_bypass\n");
    }

    printf("benchmark_status=complete\n");
    fflush(stdout);

    uint32_t heartbeat_count = 0;
    while (true) {
        printf("heartbeat=%lu benchmark_status=idle\n", static_cast<unsigned long>(heartbeat_count));
        fflush(stdout);
        heartbeat_count++;
        sleep_ms(5000);
    }
}
