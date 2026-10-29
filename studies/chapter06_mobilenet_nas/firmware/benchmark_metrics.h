#pragma once

#include <stdint.h>

/** Return the zero-based nearest-rank percentile index.
 *
 * Args:
 *     count (int): Number of sorted observations. Must be positive.
 *     percentile (int): Requested percentile in the inclusive range 1 to 100.
 *
 * Returns:
 *     int: Zero-based index of the nearest-rank observation.
 *
 * Raises:
 *     None: Callers must satisfy the documented argument constraints.
 *
 * Example:
 *     `nearest_rank_index(100, 95)` returns `94`.
 */
int nearest_rank_index(int count, int percentile);

/** Convert a ratio to a percentage scaled by 100.
 *
 * Args:
 *     numerator (uint64_t): Ratio numerator.
 *     denominator (uint64_t): Positive ratio denominator.
 *
 * Returns:
 *     uint32_t: Percentage multiplied by 100, truncated to an integer.
 *
 * Raises:
 *     None: Callers must provide a nonzero denominator.
 *
 * Example:
 *     `ratio_percent_x100(985, 1000)` returns `9850`, meaning 98.50%.
 */
uint32_t ratio_percent_x100(uint64_t numerator, uint64_t denominator);

/** Convert average power and latency to energy.
 *
 * Args:
 *     power_uW (int32_t): Positive average board power in microwatts.
 *     latency_us (uint32_t): Mean inference latency in microseconds.
 *
 * Returns:
 *     int64_t: Aggregate estimated board energy in nanojoules.
 *
 * Raises:
 *     None: Inputs are integer measurements and conversion cannot fail.
 *
 * Example:
 *     `power_latency_to_energy_nj(50000, 1000)` returns `50000`.
 */
int64_t power_latency_to_energy_nj(int32_t power_uW, uint32_t latency_us);
