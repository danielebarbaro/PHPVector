<?php

declare(strict_types=1);

namespace PHPVector\HNSW;

/**
 * Internal node inside the HNSW graph.
 *
 * Connections are stored per layer as arrays of neighbour node-IDs.
 * Using plain int[] (not a set) keeps memory low; duplicates are prevented
 * by the insertion algorithm.
 */
final class Node
{
    /**
     * Connections per layer: connections[layer] = [nodeId, nodeId, …]
     * Layer 0 is the base layer (most dense); higher layers are sparser.
     *
     * @var array<int, int[]>
     */
    public array $connections = [];

    /**
     * @param int     $id       Index into the owning Index::$nodes array.
     * @param float[] $vector   Dense embedding vector.
     * @param int     $maxLayer Highest layer this node participates in.
     */
    public function __construct(
        public readonly int $id,
        public readonly array $vector,
        public readonly int $maxLayer,
    ) {
        for ($l = 0; $l <= $maxLayer; $l++) {
            $this->connections[$l] = [];
        }
    }
}
