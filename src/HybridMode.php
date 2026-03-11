<?php

declare(strict_types=1);

namespace PHPVector;

enum HybridMode
{
    /**
     * Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank(d)).
     * Rank-based, robust to score scale differences.
     */
    case RRF;

    /**
     * Weighted linear combination of normalised vector and BM25 scores.
     * Requires explicit vectorWeight + textWeight parameters.
     */
    case Weighted;
}
