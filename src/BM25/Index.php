<?php

declare(strict_types=1);

namespace PHPVector\BM25;

use PHPVector\Document;
use PHPVector\SearchResult;

/**
 * BM25 (Okapi BM25) inverted-index for full-text search.
 *
 * Data structures:
 *   $invertedIndex[term][docId] = term frequency (count of term in document)
 *   $docLengths[docId]          = number of tokens in the document
 *   $totalTokens                = running sum used to compute avgdl
 *
 * All document IDs used internally are the integer node-IDs assigned by
 * VectorDatabase (not the user-visible Document::$id).
 */
final class Index
{
    /**
     * Inverted index: term → [nodeId → termFrequency]
     * @var array<string, array<int, int>>
     */
    private array $invertedIndex = [];

    /**
     * Per-document token counts.
     * @var array<int, int>
     */
    private array $docLengths = [];

    /** Running sum of all document lengths (used for avgdl). */
    private int $totalTokens = 0;

    /** @var array<int, Document> nodeId → Document */
    private array $documents = [];

    public function __construct(
        private readonly Config $config = new Config(),
        private readonly TokenizerInterface $tokenizer = new SimpleTokenizer(),
    ) {}

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Index a document.
     *
     * @param int      $nodeId   Internal node-ID used to correlate with the HNSW index.
     * @param Document $document Document to index.
     */
    public function addDocument(int $nodeId, Document $document): void
    {
        if ($document->text === null || $document->text === '') {
            return;
        }

        $tokens = $this->tokenizer->tokenize($document->text);
        if (empty($tokens)) {
            return;
        }

        $this->documents[$nodeId]   = $document;
        $this->docLengths[$nodeId]  = count($tokens);
        $this->totalTokens         += count($tokens);

        // Count per-term frequencies for this document.
        $termFreqs = array_count_values($tokens);
        foreach ($termFreqs as $term => $tf) {
            $this->invertedIndex[$term][$nodeId] = $tf;
        }
    }

    /**
     * Search the index and return scored documents.
     *
     * @param string $query Raw query text.
     * @param int    $k     Maximum number of results.
     *
     * @return SearchResult[]  Sorted by score descending.
     */
    public function search(string $query, int $k = 10): array
    {
        if (empty($this->documents)) {
            return [];
        }

        $queryTokens = $this->tokenizer->tokenize($query);
        if (empty($queryTokens)) {
            return [];
        }

        $numDocs = count($this->documents);
        $avgDl   = $this->totalTokens / $numDocs;
        $scores  = [];

        // Deduplicate query tokens to avoid summing the same IDF twice.
        foreach (array_unique($queryTokens) as $term) {
            if (!isset($this->invertedIndex[$term])) {
                continue;
            }

            $postings = $this->invertedIndex[$term];
            $df       = count($postings);

            // Smoothed IDF (Robertson–Jones with +1 to keep positive for common terms).
            $idf = log(($numDocs - $df + 0.5) / ($df + 0.5) + 1.0);

            foreach ($postings as $nodeId => $tf) {
                $dl    = $this->docLengths[$nodeId];
                $k1    = $this->config->k1;
                $b     = $this->config->b;

                // BM25 term-frequency normalisation.
                $tfNorm = ($tf * ($k1 + 1.0))
                    / ($tf + $k1 * (1.0 - $b + $b * $dl / $avgDl));

                $scores[$nodeId] = ($scores[$nodeId] ?? 0.0) + $idf * $tfNorm;
            }
        }

        if (empty($scores)) {
            return [];
        }

        arsort($scores);

        $results = [];
        $rank    = 1;
        foreach (array_slice($scores, 0, $k, true) as $nodeId => $score) {
            $results[] = new SearchResult(
                document: $this->documents[$nodeId],
                score: $score,
                rank: $rank++,
            );
        }
        return $results;
    }

    /**
     * Return the raw BM25 scores for a query without wrapping in SearchResult.
     * Used by VectorDatabase for hybrid score fusion.
     *
     * @return array<int, float>  nodeId → BM25 score (sorted descending)
     */
    public function scoreAll(string $query): array
    {
        if (empty($this->documents)) {
            return [];
        }

        $queryTokens = $this->tokenizer->tokenize($query);
        if (empty($queryTokens)) {
            return [];
        }

        $numDocs = count($this->documents);
        $avgDl   = $this->totalTokens / $numDocs;
        $scores  = [];

        foreach (array_unique($queryTokens) as $term) {
            if (!isset($this->invertedIndex[$term])) {
                continue;
            }

            $postings = $this->invertedIndex[$term];
            $df       = count($postings);
            $idf      = log(($numDocs - $df + 0.5) / ($df + 0.5) + 1.0);

            foreach ($postings as $nodeId => $tf) {
                $dl     = $this->docLengths[$nodeId];
                $k1     = $this->config->k1;
                $b      = $this->config->b;
                $tfNorm = ($tf * ($k1 + 1.0))
                    / ($tf + $k1 * (1.0 - $b + $b * $dl / $avgDl));

                $scores[$nodeId] = ($scores[$nodeId] ?? 0.0) + $idf * $tfNorm;
            }
        }

        arsort($scores);
        return $scores;
    }

    /** Number of documents currently indexed. */
    public function count(): int
    {
        return count($this->documents);
    }

    /** Vocabulary size (unique terms in the index). */
    public function vocabularySize(): int
    {
        return count($this->invertedIndex);
    }

    /**
     * Export the BM25 index state as plain PHP arrays.
     *
     * @return array{
     *   totalTokens: int,
     *   docLengths: array<int, int>,
     *   invertedIndex: array<string, array<int, int>>
     * }
     */
    public function exportState(): array
    {
        return [
            'totalTokens'   => $this->totalTokens,
            'docLengths'    => $this->docLengths,
            'invertedIndex' => $this->invertedIndex,
        ];
    }

    /**
     * Restore BM25 state from plain arrays.
     * Document objects are supplied by the caller to avoid storing them twice.
     *
     * @param array{
     *   totalTokens: int,
     *   docLengths: array<int, int>,
     *   invertedIndex: array<string, array<int, int>>
     * } $state
     * @param array<int, Document> $documents  nodeId → Document (from HNSW index)
     */
    public function importState(array $state, array $documents): void
    {
        $this->totalTokens   = $state['totalTokens'];
        $this->docLengths    = $state['docLengths'];
        $this->invertedIndex = $state['invertedIndex'];
        $this->documents     = $documents;
    }
}
