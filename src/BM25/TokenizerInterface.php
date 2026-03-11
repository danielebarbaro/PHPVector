<?php

declare(strict_types=1);

namespace PHPVector\BM25;

/**
 * Converts a raw text string into a sequence of normalised tokens.
 *
 * Implement this interface to plug in stemming, stop-word removal,
 * language-specific normalisation, or any custom tokenisation logic.
 */
interface TokenizerInterface
{
    /**
     * Tokenise `$text` and return an array of normalised tokens.
     * Duplicate tokens in the output are intentional — term frequency depends on them.
     *
     * @return string[]
     */
    public function tokenize(string $text): array;
}
