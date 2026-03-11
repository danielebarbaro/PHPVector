<?php

declare(strict_types=1);

namespace PHPVector\Benchmark;

/**
 * Statistical helpers for benchmark measurements.
 */
final class Stats
{
    /**
     * Compute a percentile value using linear interpolation.
     *
     * @param float[] $sorted Values sorted ascending.
     * @param float   $p      Percentile in [0, 100].
     */
    public static function percentile(array $sorted, float $p): float
    {
        $n = count($sorted);
        if ($n === 0) {
            return 0.0;
        }
        if ($n === 1) {
            return (float) $sorted[0];
        }

        $idx  = ($p / 100.0) * ($n - 1);
        $lo   = (int) floor($idx);
        $hi   = (int) ceil($idx);

        if ($lo === $hi) {
            return (float) $sorted[$lo];
        }

        $frac = $idx - $lo;
        return $sorted[$lo] * (1.0 - $frac) + $sorted[$hi] * $frac;
    }

    /** @param float[] $values */
    public static function mean(array $values): float
    {
        $n = count($values);
        return $n === 0 ? 0.0 : array_sum($values) / $n;
    }

    /**
     * Convert an array of nanosecond hrtime() measurements into a latency
     * statistics map (all values in milliseconds).
     *
     * @param int[] $ns  Raw nanosecond timings.
     * @return array{min:float, mean:float, p50:float, p95:float, p99:float, max:float}
     */
    public static function latencyStats(array $ns): array
    {
        $ms = array_map(static fn(int $v): float => $v / 1_000_000.0, $ns);
        sort($ms);

        return [
            'min'  => (float) min($ms),
            'mean' => self::mean($ms),
            'p50'  => self::percentile($ms, 50),
            'p95'  => self::percentile($ms, 95),
            'p99'  => self::percentile($ms, 99),
            'max'  => (float) max($ms),
        ];
    }
}
