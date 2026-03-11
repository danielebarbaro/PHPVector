<?php

declare(strict_types=1);

namespace PHPVector\HNSW;

/**
 * Max-heap ordered by the distance value stored in element[0].
 * `extract()` always returns the element with the *largest* distance first.
 *
 * Elements are stored as [float $distance, int $nodeId].
 *
 * @extends \SplHeap<array{float, int}>
 */
final class MaxDistanceHeap extends \SplHeap
{
    /**
     * @param array{float, int} $value1
     * @param array{float, int} $value2
     */
    protected function compare(mixed $value1, mixed $value2): int
    {
        // Return positive when value1 has *higher* priority (larger distance).
        return $value1[0] <=> $value2[0];
    }
}
