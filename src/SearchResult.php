<?php

declare(strict_types=1);

namespace PHPVector;

/**
 * A single item returned from any search method.
 * Higher `score` always means a better match.
 */
final class SearchResult
{
    public function __construct(
        /** The matched document. */
        public readonly Document $document,
        /**
         * Relevance score (higher = more relevant).
         *  - Vector search  : 1 − distance  (range depends on Distance function)
         *  - BM25 search    : raw BM25 score
         *  - Hybrid search  : RRF or weighted-combination score
         */
        public readonly float $score,
        /** 1-based rank within the result list. */
        public readonly int $rank,
    ) {}
}
