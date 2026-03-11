<?php

declare(strict_types=1);

namespace PHPVector\Tests;

use PHPUnit\Framework\TestCase;
use PHPVector\BM25\Config as BM25Config;
use PHPVector\BM25\SimpleTokenizer;
use PHPVector\Document;
use PHPVector\HNSW\Config as HNSWConfig;
use PHPVector\HybridMode;
use PHPVector\VectorDatabase;

final class PersistenceTest extends TestCase
{
    private string $tmpFile;

    protected function setUp(): void
    {
        $this->tmpFile = tempnam(sys_get_temp_dir(), 'phpvtest_') . '.phpv';
    }

    protected function tearDown(): void
    {
        if (file_exists($this->tmpFile)) {
            unlink($this->tmpFile);
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private function makeDb(): VectorDatabase
    {
        return new VectorDatabase(
            hnswConfig: new HNSWConfig(M: 8, efConstruction: 50, efSearch: 20),
            bm25Config: new BM25Config(),
            tokenizer:  new SimpleTokenizer([]),
        );
    }

    private function loadDb(): VectorDatabase
    {
        return VectorDatabase::load(
            path:       $this->tmpFile,
            hnswConfig: new HNSWConfig(M: 8, efConstruction: 50, efSearch: 20),
            bm25Config: new BM25Config(),
            tokenizer:  new SimpleTokenizer([]),
        );
    }

    // ------------------------------------------------------------------
    // Round-trip: vector / text / hybrid search parity
    // ------------------------------------------------------------------

    public function testRoundTripVectorSearch(): void
    {
        $db = $this->makeDb();

        // Distinct cluster vectors so the nearest-neighbour is deterministic.
        $db->addDocument(new Document(id: 1, vector: [1.0, 0.0, 0.0, 0.0], text: 'alpha one'));
        $db->addDocument(new Document(id: 2, vector: [0.0, 1.0, 0.0, 0.0], text: 'beta two'));
        $db->addDocument(new Document(id: 3, vector: [0.0, 0.0, 1.0, 0.0], text: 'gamma three'));
        $db->addDocument(new Document(id: 4, vector: [0.0, 0.0, 0.0, 1.0], text: 'delta four'));

        $query   = [1.0, 0.1, 0.0, 0.0];
        $before  = $db->vectorSearch($query, k: 1);

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        $after = $loaded->vectorSearch($query, k: 1);

        self::assertSame($before[0]->document->id, $after[0]->document->id);
    }

    public function testRoundTripTextSearch(): void
    {
        $db = $this->makeDb();

        $db->addDocument(new Document(id: 'a', vector: [0.1, 0.9], text: 'machine learning vector search'));
        $db->addDocument(new Document(id: 'b', vector: [0.5, 0.5], text: 'database storage systems'));
        $db->addDocument(new Document(id: 'c', vector: [0.9, 0.1], text: 'neural network deep learning'));

        $query  = 'machine learning';
        $before = $db->textSearch($query, k: 1);

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        $after = $loaded->textSearch($query, k: 1);

        self::assertSame($before[0]->document->id, $after[0]->document->id);
    }

    public function testRoundTripHybridSearchRRF(): void
    {
        $db = $this->makeDb();

        $db->addDocument(new Document(id: 10, vector: [1.0, 0.0], text: 'php vector database search'));
        $db->addDocument(new Document(id: 20, vector: [0.0, 1.0], text: 'sql relational storage'));

        $query  = [1.0, 0.0];
        $text   = 'vector';
        $before = $db->hybridSearch($query, $text, k: 1, mode: HybridMode::RRF);

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        $after = $loaded->hybridSearch($query, $text, k: 1, mode: HybridMode::RRF);

        self::assertSame($before[0]->document->id, $after[0]->document->id);
    }

    public function testRoundTripHybridSearchWeighted(): void
    {
        $db = $this->makeDb();

        $db->addDocument(new Document(id: 10, vector: [1.0, 0.0], text: 'php vector database search'));
        $db->addDocument(new Document(id: 20, vector: [0.0, 1.0], text: 'sql relational storage'));

        $query  = [1.0, 0.0];
        $text   = 'vector';
        $before = $db->hybridSearch($query, $text, k: 1, mode: HybridMode::Weighted);

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        $after = $loaded->hybridSearch($query, $text, k: 1, mode: HybridMode::Weighted);

        self::assertSame($before[0]->document->id, $after[0]->document->id);
    }

    // ------------------------------------------------------------------
    // Document fidelity
    // ------------------------------------------------------------------

    public function testCountMatchesAfterRoundTrip(): void
    {
        $db = $this->makeDb();

        for ($i = 0; $i < 10; $i++) {
            $db->addDocument(new Document(
                id:     $i,
                vector: [sin($i), cos($i)],
                text:   "document number {$i}",
            ));
        }

        $db->persist($this->tmpFile);
        $loaded = VectorDatabase::load($this->tmpFile, new HNSWConfig(M: 8, efConstruction: 50, efSearch: 20));

        self::assertSame(10, $loaded->count());
    }

    public function testMetadataAndTextPreservedExactly(): void
    {
        $db = $this->makeDb();

        $db->addDocument(new Document(
            id:       'doc-meta',
            vector:   [0.5, 0.5],
            text:     'some full-text content',
            metadata: ['key' => 'value', 'nested' => ['a' => 1, 'b' => true], 'num' => 3.14],
        ));
        $db->addDocument(new Document(
            id:     'no-text',
            vector: [0.1, 0.9],
            text:   null,
        ));
        $db->addDocument(new Document(
            id:     'empty-meta',
            vector: [0.9, 0.1],
            text:   'only text',
        ));

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        // Retrieve 'doc-meta' via vector search (it's the only document close to [0.5, 0.5]).
        $results = $loaded->vectorSearch([0.5, 0.5], k: 3);
        $byId    = [];
        foreach ($results as $r) {
            $byId[$r->document->id] = $r->document;
        }

        self::assertArrayHasKey('doc-meta', $byId);
        $restored = $byId['doc-meta'];
        self::assertSame('some full-text content', $restored->text);
        self::assertSame('value', $restored->metadata['key']);
        self::assertSame(['a' => 1, 'b' => true], $restored->metadata['nested']);
        self::assertEqualsWithDelta(3.14, $restored->metadata['num'], 1e-9);

        self::assertArrayHasKey('no-text', $byId);
        self::assertNull($byId['no-text']->text);
        self::assertSame([], $byId['no-text']->metadata);

        self::assertArrayHasKey('empty-meta', $byId);
        self::assertSame([], $byId['empty-meta']->metadata);
    }

    public function testStringAndIntDocumentIdsPreserved(): void
    {
        $db = $this->makeDb();

        $db->addDocument(new Document(id: 42,    vector: [1.0, 0.0]));
        $db->addDocument(new Document(id: 'str', vector: [0.0, 1.0]));

        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        $intResult = $loaded->vectorSearch([1.0, 0.0], k: 1);
        self::assertSame(42, $intResult[0]->document->id);
        self::assertIsInt($intResult[0]->document->id);

        $strResult = $loaded->vectorSearch([0.0, 1.0], k: 1);
        self::assertSame('str', $strResult[0]->document->id);
        self::assertIsString($strResult[0]->document->id);
    }

    // ------------------------------------------------------------------
    // File format
    // ------------------------------------------------------------------

    public function testFileBeginsWithMagicBytes(): void
    {
        $db = $this->makeDb();
        $db->addDocument(new Document(id: 1, vector: [1.0, 0.0]));
        $db->persist($this->tmpFile);

        $handle = fopen($this->tmpFile, 'rb');
        self::assertNotFalse($handle);
        $magic = fread($handle, 4);
        fclose($handle);

        self::assertSame('PHPV', $magic);
    }

    public function testEmptyDatabaseRoundTrip(): void
    {
        $db = $this->makeDb();
        $db->persist($this->tmpFile);
        $loaded = $this->loadDb();

        self::assertSame(0, $loaded->count());
        self::assertSame([], $loaded->vectorSearch([1.0, 0.0], k: 5));
        self::assertSame([], $loaded->textSearch('anything', k: 5));
    }

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    public function testLoadThrowsOnDistanceMismatch(): void
    {
        $db = new VectorDatabase(
            hnswConfig: new HNSWConfig(distance: \PHPVector\Distance::Cosine),
        );
        $db->addDocument(new Document(id: 1, vector: [1.0, 0.0]));
        $db->persist($this->tmpFile);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessageMatches('/[Dd]istance/');

        VectorDatabase::load(
            path:       $this->tmpFile,
            hnswConfig: new HNSWConfig(distance: \PHPVector\Distance::Euclidean),
        );
    }

    public function testLoadThrowsOnInvalidFile(): void
    {
        file_put_contents($this->tmpFile, 'not a valid phpv file at all');

        $this->expectException(\RuntimeException::class);
        VectorDatabase::load($this->tmpFile);
    }
}
