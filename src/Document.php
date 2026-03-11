<?php

declare(strict_types=1);

namespace PHPVector;

/**
 * Represents a single item stored in the database.
 *
 * @phpstan-type Metadata array<string, mixed>
 */
final class Document
{
    /**
     * @param string|int    $id       Unique identifier (user-supplied or auto-assigned).
     * @param float[]       $vector   Dense embedding vector.
     * @param string|null   $text     Raw text content used for BM25 indexing (optional).
     * @param array<string, mixed> $metadata Arbitrary key-value payload returned with results.
     */
    public function __construct(
        public readonly string|int $id,
        public readonly array $vector,
        public readonly ?string $text = null,
        public readonly array $metadata = [],
    ) {}
}
