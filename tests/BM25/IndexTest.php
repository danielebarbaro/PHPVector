<?php

declare(strict_types=1);

namespace PHPVector\Tests\BM25;

use PHPUnit\Framework\TestCase;
use PHPVector\BM25\Config;
use PHPVector\BM25\Index;
use PHPVector\BM25\SimpleTokenizer;
use PHPVector\Document;

final class IndexTest extends TestCase
{
    private function makeIndex(): Index
    {
        return new Index(new Config(k1: 1.5, b: 0.75), new SimpleTokenizer([]));
    }

    // ------------------------------------------------------------------
    // Basic functionality
    // ------------------------------------------------------------------

    public function testEmptyIndexReturnsNoResults(): void
    {
        $index = $this->makeIndex();
        self::assertSame([], $index->search('hello world', 5));
    }

    public function testUnmatchedQueryReturnsNoResults(): void
    {
        $index = $this->makeIndex();
        $index->addDocument(0, new Document(id: 1, vector: [], text: 'the quick brown fox'));

        self::assertSame([], $index->search('elephant zebra', 5));
    }

    public function testSingleDocumentIsReturned(): void
    {
        $index = $this->makeIndex();
        $doc   = new Document(id: 1, vector: [], text: 'hello world');
        $index->addDocument(0, $doc);

        $results = $index->search('hello', 5);

        self::assertCount(1, $results);
        self::assertSame($doc, $results[0]->document);
        self::assertGreaterThan(0.0, $results[0]->score);
    }

    public function testMostRelevantDocumentRanksFirst(): void
    {
        $index = $this->makeIndex();

        $index->addDocument(0, new Document(id: 1, vector: [], text: 'php vector search engine'));
        $index->addDocument(1, new Document(id: 2, vector: [], text: 'vector vector vector search php php php'));
        $index->addDocument(2, new Document(id: 3, vector: [], text: 'database management system'));

        $results = $index->search('php vector', 3);

        self::assertNotEmpty($results);
        // The document about "database management" should not rank first.
        self::assertNotSame(3, $results[0]->document->id);
    }

    public function testResultsAreSortedByScoreDescending(): void
    {
        $index = $this->makeIndex();

        $index->addDocument(0, new Document(id: 'a', vector: [], text: 'apple banana cherry'));
        $index->addDocument(1, new Document(id: 'b', vector: [], text: 'banana cherry cherry cherry'));
        $index->addDocument(2, new Document(id: 'c', vector: [], text: 'cherry cherry cherry cherry cherry'));

        $results = $index->search('cherry', 3);

        self::assertCount(3, $results);
        for ($i = 1; $i < count($results); $i++) {
            self::assertGreaterThanOrEqual($results[$i]->score, $results[$i - 1]->score);
        }
    }

    public function testKLimitsResultCount(): void
    {
        $index = $this->makeIndex();

        for ($i = 0; $i < 10; $i++) {
            $index->addDocument($i, new Document(id: $i, vector: [], text: "document number $i with common word"));
        }

        $results = $index->search('common word', 3);
        self::assertCount(3, $results);
    }

    // ------------------------------------------------------------------
    // Ranking properties
    // ------------------------------------------------------------------

    public function testHigherTermFrequencyScoresHigher(): void
    {
        $index = $this->makeIndex();

        // doc1: term appears once; doc2: term appears many times.
        $index->addDocument(0, new Document(id: 1, vector: [], text: 'machine learning is cool'));
        $index->addDocument(1, new Document(id: 2, vector: [], text: 'machine machine machine machine machine'));

        $results = $index->search('machine', 2);

        self::assertSame(2, $results[0]->document->id, 'Higher TF should rank higher');
    }

    public function testRareTermScoresHigherThanCommonTerm(): void
    {
        $index = $this->makeIndex();

        // 'vector' appears in only one doc; 'data' appears in all.
        $index->addDocument(0, new Document(id: 1, vector: [], text: 'data vector search'));
        $index->addDocument(1, new Document(id: 2, vector: [], text: 'data analysis'));
        $index->addDocument(2, new Document(id: 3, vector: [], text: 'data science'));

        $allScores = $index->scoreAll('vector data');

        // doc #1 (nodeId 0) should score highest because it contains the rare term 'vector'.
        self::assertArrayHasKey(0, $allScores);
        self::assertEquals(max($allScores), $allScores[0]);
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    public function testDocumentWithNullTextIsSkipped(): void
    {
        $index = $this->makeIndex();
        $index->addDocument(0, new Document(id: 1, vector: [])); // no text
        $index->addDocument(1, new Document(id: 2, vector: [], text: 'hello world'));

        self::assertSame(1, $index->count());
    }

    public function testRankIsOneBased(): void
    {
        $index = $this->makeIndex();
        $index->addDocument(0, new Document(id: 1, vector: [], text: 'alpha beta'));
        $index->addDocument(1, new Document(id: 2, vector: [], text: 'alpha gamma'));

        $results = $index->search('alpha', 5);
        foreach ($results as $i => $sr) {
            self::assertSame($i + 1, $sr->rank);
        }
    }

    // ------------------------------------------------------------------
    // SimpleTokenizer
    // ------------------------------------------------------------------

    public function testTokenizerStopWordsAreRemoved(): void
    {
        $tokenizer = new SimpleTokenizer(stopWords: ['the', 'is', 'a']);
        $tokens    = $tokenizer->tokenize('The sky is a beautiful blue');

        self::assertNotContains('the', $tokens);
        self::assertNotContains('is', $tokens);
        self::assertNotContains('a', $tokens);
        self::assertContains('sky', $tokens);
        self::assertContains('beautiful', $tokens);
        self::assertContains('blue', $tokens);
    }

    public function testTokenizerMinLengthFiltersShortTokens(): void
    {
        $tokenizer = new SimpleTokenizer(stopWords: [], minTokenLength: 4);
        $tokens    = $tokenizer->tokenize('hi there it is good');

        foreach ($tokens as $token) {
            self::assertGreaterThanOrEqual(4, mb_strlen($token));
        }
    }

    public function testTokenizerLowerCases(): void
    {
        $tokenizer = new SimpleTokenizer(stopWords: []);
        $tokens    = $tokenizer->tokenize('PHP Vector DATABASE');

        self::assertContains('php', $tokens);
        self::assertContains('vector', $tokens);
        self::assertContains('database', $tokens);
    }
}
