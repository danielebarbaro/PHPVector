<?php

declare(strict_types=1);

namespace PHPVector\BM25;

/**
 * Configuration for the BM25 (Okapi BM25) scoring function.
 *
 * Score for document D given query Q:
 *
 *   score(D, Q) = Σᵢ IDF(qᵢ) ·  tf(qᵢ, D) · (k1 + 1)
 *                               ─────────────────────────────────────────────
 *                               tf(qᵢ, D) + k1 · (1 − b + b · |D| / avgdl)
 *
 * where:
 *   IDF(q) = ln( (N − n(q) + 0.5) / (n(q) + 0.5) + 1 )
 *   N      = total number of documents
 *   n(q)   = number of documents containing term q
 *   avgdl  = average document length (in tokens)
 */
final class Config
{
    /**
     * Term-frequency saturation parameter.
     * Controls how quickly TF saturates (diminishing returns for repeated terms).
     * Typical range: 1.2 – 2.0. Default: 1.5.
     */
    public readonly float $k1;

    /**
     * Length normalisation parameter.
     * 0 = no length normalisation, 1 = full normalisation.
     * Typical value: 0.75.
     */
    public readonly float $b;

    public function __construct(float $k1 = 1.5, float $b = 0.75)
    {
        if ($k1 < 0) {
            throw new \InvalidArgumentException('k1 must be ≥ 0.');
        }
        if ($b < 0.0 || $b > 1.0) {
            throw new \InvalidArgumentException('b must be in [0, 1].');
        }
        $this->k1 = $k1;
        $this->b  = $b;
    }
}
