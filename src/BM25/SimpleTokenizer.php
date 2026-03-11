<?php

declare(strict_types=1);

namespace PHPVector\BM25;

/**
 * A lightweight, language-agnostic tokenizer.
 *
 * Pipeline:
 *  1. Lower-case the input (Unicode-aware via `mb_strtolower`).
 *  2. Split on any run of non-alphanumeric characters.
 *  3. Optionally remove a configurable stop-word list.
 *  4. Drop tokens shorter than `$minTokenLength`.
 */
final class SimpleTokenizer implements TokenizerInterface
{
    /** @var array<string, true> */
    private readonly array $stopWords;

    /**
     * @param string[] $stopWords     Words to discard (case-insensitive).
     * @param int      $minTokenLength Minimum token length to keep (default: 2).
     */
    public function __construct(
        array $stopWords = self::DEFAULT_STOP_WORDS,
        private readonly int $minTokenLength = 2,
    ) {
        $this->stopWords = array_fill_keys(
            array_map('mb_strtolower', $stopWords),
            true,
        );
    }

    /** {@inheritdoc} */
    public function tokenize(string $text): array
    {
        $text   = mb_strtolower($text, 'UTF-8');
        $tokens = preg_split('/[^\p{L}\p{N}]+/u', $text, -1, PREG_SPLIT_NO_EMPTY) ?: [];

        $result = [];
        foreach ($tokens as $token) {
            if (
                mb_strlen($token, 'UTF-8') >= $this->minTokenLength
                && !isset($this->stopWords[$token])
            ) {
                $result[] = $token;
            }
        }
        return $result;
    }

    /**
     * Common English stop words.
     * Replace or extend via the constructor for other languages or domains.
     */
    public const DEFAULT_STOP_WORDS = [
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'it', 'as', 'be', 'was', 'are',
        'were', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'not',
        'that', 'this', 'from', 'so', 'if', 'up', 'out', 'no', 'its',
        'then', 'than', 'into', 'can', 'will', 'just', 'about', 'also',
    ];
}
