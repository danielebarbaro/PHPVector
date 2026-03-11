<?php

declare(strict_types=1);

namespace PHPVector\HNSW;

/**
 * Min-heap ordered by the distance value stored in element[0].
 * `extract()` always returns the element with the *smallest* distance first.
 *
 * Elements are stored as [float $distance, int $nodeId].
 *
 * @extends \SplHeap<array{float, int}>
 */
final class MinDistanceHeap extends \SplHeap
{
    /**
     * SplHeap extracts the element for which compare() returns the largest positive value.
     * For a min-heap we want lower distances extracted first, so we invert the comparison.
     *
     * @param array{float, int} $value1
     * @param array{float, int} $value2
     */
    protected function compare(mixed $value1, mixed $value2): int
    {
        // Return positive when value1 has *higher* priority (smaller distance).
        return $value2[0] <=> $value1[0];
    }
}
