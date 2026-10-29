#include "benchmark_metrics.h"

int nearest_rank_index(int count, int percentile) {
    int rank = (count * percentile + 99) / 100;
    return rank - 1;
}

uint32_t ratio_percent_x100(uint64_t numerator, uint64_t denominator) {
    return static_cast<uint32_t>((numerator * 10000ULL) / denominator);
}

int64_t power_latency_to_energy_nj(int32_t power_uW, uint32_t latency_us) {
    return (
        static_cast<int64_t>(power_uW) * static_cast<int64_t>(latency_us)
    ) / 1000LL;
}
