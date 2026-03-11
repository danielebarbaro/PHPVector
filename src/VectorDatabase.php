<?php

declare(strict_types=1);

namespace PHPVector;

use PHPVector\BM25\Config as BM25Config;
use PHPVector\BM25\Index as BM25Index;
use PHPVector\BM25\TokenizerInterface;
use PHPVector\BM25\SimpleTokenizer;
use PHPVector\HNSW\Config as HNSWConfig;
use PHPVector\HNSW\Index as HNSWIndex;
use PHPVector\Persistence\BinarySerializer;

/**
 * High-level façade combining HNSW vector search with BM25 full-text search.
 *
 * Supports three retrieval modes:
 *  1. **Vector search** — approximate nearest-neighbour via HNSW.
 *  2. **Text search**   — BM25 ranked full-text search.
 *  3. **Hybrid search** — fuse both result sets with either
 *                         Reciprocal Rank Fusion (RRF) or a weighted linear combination.
 *
 * Quick start
 * -----------
 * ```php
 * $db = new VectorDatabase();
 *
 * $db->addDocument(new Document(id: 1, vector: [0.1, 0.9, ...], text: 'PHP vector database'));
 * $db->addDocument(new Document(id: 2, vector: [0.8, 0.2, ...], text: 'Approximate nearest neighbour'));
 *
 * $results = $db->hybridSearch(vector: $queryVec, text: 'vector search php', k: 5);
 * ```
 */
final class VectorDatabase
{
    private readonly HNSWIndex $hnswIndex;
    private readonly BM25Index $bm25Index;
    private readonly HNSWConfig $hnswConfig;

    /**
     * Internal sequence counter: maps sequential int IDs → Documents.
     * Both HNSWIndex and BM25Index use this integer as node-ID.
     */
    private int $nextId = 0;

    /** @var array<int, Document> nodeId → Document (mirrors HNSWIndex::$documents) */
    private array $nodeIdToDoc = [];

    /** @var array<string|int, int> user document ID → nodeId */
    private array $docIdToNodeId = [];

    public function __construct(
        HNSWConfig $hnswConfig = new HNSWConfig(),
        BM25Config $bm25Config = new BM25Config(),
        TokenizerInterface $tokenizer = new SimpleTokenizer(),
    ) {
        $this->hnswConfig = $hnswConfig;
        $this->hnswIndex  = new HNSWIndex($hnswConfig);
        $this->bm25Index  = new BM25Index($bm25Config, $tokenizer);
    }

    // ------------------------------------------------------------------
    // Indexing
    // ------------------------------------------------------------------

    /**
     * Add a single document.
     *
     * @throws \RuntimeException if a document with the same ID already exists.
     */
    public function addDocument(Document $document): void
    {
        if (isset($this->docIdToNodeId[$document->id])) {
            throw new \RuntimeException(
                sprintf('Document with id "%s" already exists.', $document->id)
            );
        }

        $nodeId = $this->nextId++;
        $this->nodeIdToDoc[$nodeId]          = $document;
        $this->docIdToNodeId[$document->id]  = $nodeId;

        $this->hnswIndex->insert($document);
        $this->bm25Index->addDocument($nodeId, $document);
    }

    /**
     * Add multiple documents in one call.
     *
     * @param Document[] $documents
     */
    public function addDocuments(array $documents): void
    {
        foreach ($documents as $doc) {
            $this->addDocument($doc);
        }
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /**
     * Pure vector search via HNSW.
     *
     * @param float[]  $vector  Query embedding.
     * @param int      $k       Number of results.
     * @param int|null $ef      Candidate list size (≥ k; null = use index default).
     *
     * @return SearchResult[]
     */
    public function vectorSearch(array $vector, int $k = 10, ?int $ef = null): array
    {
        return $this->hnswIndex->search($vector, $k, $ef);
    }

    /**
     * Pure BM25 full-text search.
     *
     * @return SearchResult[]
     */
    public function textSearch(string $query, int $k = 10): array
    {
        return $this->bm25Index->search($query, $k);
    }

    /**
     * Hybrid search: fuse vector similarity and BM25 results.
     *
     * @param float[]    $vector        Query embedding (used for HNSW leg).
     * @param string     $text          Query text (used for BM25 leg).
     * @param int        $k             Final number of results.
     * @param int        $fetchK        Number of candidates fetched from each leg before fusion.
     *                                  Higher values improve recall at the cost of latency.
     *                                  Defaults to max(k * 3, 50).
     * @param HybridMode $mode          Fusion strategy.
     * @param float      $vectorWeight  Weight for vector scores (Weighted mode only).
     * @param float      $textWeight    Weight for BM25 scores (Weighted mode only).
     * @param int        $rrfK          RRF constant k (RRF mode only). Typical value: 60.
     *
     * @return SearchResult[]
     */
    public function hybridSearch(
        array $vector,
        string $text,
        int $k = 10,
        ?int $fetchK = null,
        HybridMode $mode = HybridMode::RRF,
        float $vectorWeight = 0.5,
        float $textWeight = 0.5,
        int $rrfK = 60,
    ): array {
        $fetchK ??= max($k * 3, 50);

        $vectorResults = $this->hnswIndex->search($vector, $fetchK);
        $textScores    = $this->bm25Index->scoreAll($text);

        return match ($mode) {
            HybridMode::RRF      => $this->fuseRRF($vectorResults, $textScores, $k, $rrfK),
            HybridMode::Weighted => $this->fuseWeighted($vectorResults, $textScores, $k, $vectorWeight, $textWeight),
        };
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /**
     * Persist the full database state to a binary .phpv file.
     *
     * @throws \RuntimeException on write failure.
     */
    public function persist(string $path): void
    {
        $hnswState = $this->hnswIndex->exportState();
        $bm25State = $this->bm25Index->exportState();

        $state = [
            'distance'      => self::encodeDistance($this->hnswConfig->distance),
            'dimension'     => $hnswState['dimension'] ?? 0,
            'nodeCount'     => count($hnswState['nodes']),
            'entryPoint'    => $hnswState['entryPoint'],
            'maxLayer'      => $hnswState['maxLayer'],
            'nextId'        => $this->nextId,
            'docIdToNodeId' => $this->docIdToNodeId,
            'nodes'         => $hnswState['nodes'],
            'documents'     => $hnswState['documents'],
            'bm25'          => $bm25State,
        ];

        (new BinarySerializer())->persist($path, $state);
    }

    /**
     * Load a VectorDatabase from a previously persisted .phpv file.
     *
     * The supplied $hnswConfig **must** use the same distance metric as the
     * one used when the file was written; an exception is thrown otherwise.
     *
     * @throws \RuntimeException on read failure or distance metric mismatch.
     */
    public static function load(
        string $path,
        HNSWConfig $hnswConfig = new HNSWConfig(),
        BM25Config $bm25Config = new BM25Config(),
        TokenizerInterface $tokenizer = new SimpleTokenizer(),
    ): self {
        $state = (new BinarySerializer())->load($path);

        $distCode = self::encodeDistance($hnswConfig->distance);
        if ($distCode !== $state['distance']) {
            throw new \RuntimeException(sprintf(
                'Distance mismatch: config uses %s (code %d) but file was built with code %d.',
                $hnswConfig->distance->name,
                $distCode,
                $state['distance'],
            ));
        }

        $db = new self($hnswConfig, $bm25Config, $tokenizer);
        $db->nextId        = $state['nextId'];
        $db->docIdToNodeId = $state['docIdToNodeId'];

        // Rebuild Document objects and the nodeId → Document map.
        foreach ($state['documents'] as $nodeId => $docData) {
            $db->nodeIdToDoc[$nodeId] = new Document(
                id:       $docData['id'],
                vector:   $state['nodes'][$nodeId]['vector'],
                text:     $docData['text'],
                metadata: $docData['metadata'],
            );
        }

        // Restore HNSW graph (creates its own Document copies internally).
        $db->hnswIndex->importState($state);

        // Restore BM25 index, sharing VectorDatabase's Document objects.
        $db->bm25Index->importState($state['bm25'], $db->nodeIdToDoc);

        return $db;
    }

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------

    /** Total number of documents stored. */
    public function count(): int
    {
        return $this->nextId;
    }

    // ------------------------------------------------------------------
    // Fusion strategies
    // ------------------------------------------------------------------

    /**
     * Reciprocal Rank Fusion.
     *
     *   RRF(d) = Σᵣ  1 / (k + rankᵣ(d))
     *
     * This is rank-based and scale-invariant — no normalisation needed.
     *
     * @param SearchResult[]       $vectorResults
     * @param array<int, float>    $textScores     nodeId → BM25 score (pre-sorted desc)
     * @param int                  $k
     * @param int                  $rrfK
     *
     * @return SearchResult[]
     */
    private function fuseRRF(
        array $vectorResults,
        array $textScores,
        int $k,
        int $rrfK,
    ): array {
        $fused = [];

        // Vector ranks (1-based).
        foreach ($vectorResults as $sr) {
            $nodeId = $this->docIdToNodeId[$sr->document->id];
            $fused[$nodeId] = ($fused[$nodeId] ?? 0.0) + 1.0 / ($rrfK + $sr->rank);
        }

        // BM25 ranks (1-based; $textScores is already sorted descending by scoreAll()).
        $bm25Rank = 1;
        foreach ($textScores as $nodeId => $score) {
            $fused[$nodeId] = ($fused[$nodeId] ?? 0.0) + 1.0 / ($rrfK + $bm25Rank);
            $bm25Rank++;
        }

        arsort($fused);

        return $this->buildSearchResults(array_slice($fused, 0, $k, true));
    }

    /**
     * Weighted linear combination of min-max normalised scores.
     *
     *   combined(d) = α · vecNorm(d) + β · bm25Norm(d)
     *
     * @param SearchResult[]    $vectorResults
     * @param array<int, float> $textScores
     *
     * @return SearchResult[]
     */
    private function fuseWeighted(
        array $vectorResults,
        array $textScores,
        int $k,
        float $vectorWeight,
        float $textWeight,
    ): array {
        // Normalise vector scores to [0, 1].
        $vecNorm = $this->minMaxNormalise(
            array_combine(
                array_map(fn($sr) => $this->docIdToNodeId[$sr->document->id], $vectorResults),
                array_column($vectorResults, 'score'),
            )
        );

        // Normalise BM25 scores to [0, 1].
        $bm25Norm = $this->minMaxNormalise($textScores);

        // Collect all candidate node IDs from both legs.
        $allIds = array_unique(array_merge(array_keys($vecNorm), array_keys($bm25Norm)));

        $fused = [];
        foreach ($allIds as $nodeId) {
            $fused[$nodeId] =
                $vectorWeight * ($vecNorm[$nodeId] ?? 0.0) +
                $textWeight   * ($bm25Norm[$nodeId] ?? 0.0);
        }

        arsort($fused);

        return $this->buildSearchResults(array_slice($fused, 0, $k, true));
    }

    /**
     * Min-max normalise a nodeId → score map to [0, 1].
     *
     * @param array<int, float> $scores
     * @return array<int, float>
     */
    private function minMaxNormalise(array $scores): array
    {
        if (empty($scores)) {
            return [];
        }

        $min = min($scores);
        $max = max($scores);

        if ($max === $min) {
            return array_fill_keys(array_keys($scores), 1.0);
        }

        $range = $max - $min;
        $out   = [];
        foreach ($scores as $nodeId => $score) {
            $out[$nodeId] = ($score - $min) / $range;
        }
        return $out;
    }

    /**
     * Convert a nodeId → fusedScore map into ranked SearchResult objects.
     *
     * @param array<int, float> $fused  Already sorted descending, sliced to k.
     * @return SearchResult[]
     */
    private function buildSearchResults(array $fused): array
    {
        $results = [];
        $rank    = 1;
        foreach ($fused as $nodeId => $score) {
            $results[] = new SearchResult(
                document: $this->nodeIdToDoc[$nodeId],
                score: $score,
                rank: $rank++,
            );
        }
        return $results;
    }

    // ------------------------------------------------------------------
    // Distance codec
    // ------------------------------------------------------------------

    private static function encodeDistance(Distance $d): int
    {
        return match ($d) {
            Distance::Cosine     => 0,
            Distance::Euclidean  => 1,
            Distance::DotProduct => 2,
            Distance::Manhattan  => 3,
        };
    }
}
