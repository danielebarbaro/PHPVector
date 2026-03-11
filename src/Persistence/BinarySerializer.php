<?php

declare(strict_types=1);

namespace PHPVector\Persistence;

/**
 * Binary serializer for the PHPVector (.phpv) file format.
 *
 * Encodes and decodes the full VectorDatabase state using pack/unpack for
 * maximum throughput — one file_put_contents write, one file_get_contents read.
 *
 * Format: HEADER · ID-MAPPING · NODE SECTION · DOCUMENT SECTION · BM25 SECTION
 * See plan documentation for the exact byte layout.
 */
final class BinarySerializer
{
    private const MAGIC            = 'PHPV';
    private const VERSION          = 1;
    private const NULL_ENTRY_POINT = 0xFFFFFFFF;

    /**
     * Encode $state to binary and write to $path (single syscall).
     *
     * @param array{
     *   distance: int,
     *   dimension: int,
     *   nodeCount: int,
     *   entryPoint: int|null,
     *   maxLayer: int,
     *   nextId: int,
     *   docIdToNodeId: array<string|int, int>,
     *   nodes: array<int, array{maxLayer: int, vector: float[], connections: array<int, int[]>}>,
     *   documents: array<int, array{id: string|int, text: string|null, metadata: array}>,
     *   bm25: array{totalTokens: int, docLengths: array<int,int>, invertedIndex: array<string, array<int,int>>}
     * } $state
     */
    public function persist(string $path, array $state): void
    {
        $buf = '';

        // ── HEADER (26 bytes) ────────────────────────────────────────────
        $buf .= self::MAGIC;
        $buf .= pack('CC', self::VERSION, $state['distance']);
        $buf .= pack('NNNNN',
            $state['dimension'],
            $state['nodeCount'],
            $state['entryPoint'] ?? self::NULL_ENTRY_POINT,
            $state['maxLayer'],
            $state['nextId'],
        );

        // ── ID-MAPPING ───────────────────────────────────────────────────
        $json = json_encode($state['docIdToNodeId'], JSON_THROW_ON_ERROR);
        $buf .= pack('N', strlen($json)) . $json;

        // ── NODE SECTION ─────────────────────────────────────────────────
        $dim = $state['dimension'];
        foreach ($state['nodes'] as $nodeId => $node) {
            $buf .= pack('NN', $nodeId, $node['maxLayer']);
            if ($dim > 0) {
                $buf .= pack('d*', ...$node['vector']);
            }
            for ($l = 0; $l <= $node['maxLayer']; $l++) {
                $conns = $node['connections'][$l] ?? [];
                $cnt   = count($conns);
                $buf  .= pack('N', $cnt);
                if ($cnt > 0) {
                    $buf .= pack('N*', ...$conns);
                }
            }
        }

        // ── DOCUMENT SECTION (same iteration order as nodes) ─────────────
        foreach ($state['documents'] as $doc) {
            $id = $doc['id'];
            if (is_int($id)) {
                $buf .= pack('Cq', 0, $id);
            } else {
                $idBytes = (string) $id;
                $buf .= pack('CN', 1, strlen($idBytes)) . $idBytes;
            }

            $text = $doc['text'];
            if ($text === null || $text === '') {
                $buf .= pack('N', 0);
            } else {
                $buf .= pack('N', strlen($text)) . $text;
            }

            $meta = $doc['metadata'];
            if (empty($meta)) {
                $buf .= pack('N', 0);
            } else {
                $metaJson = json_encode($meta, JSON_THROW_ON_ERROR);
                $buf .= pack('N', strlen($metaJson)) . $metaJson;
            }
        }

        // ── BM25 SECTION ─────────────────────────────────────────────────
        $bm25 = $state['bm25'];
        $buf .= pack('N', $bm25['totalTokens']);

        $docLengths = $bm25['docLengths'];
        $buf .= pack('N', count($docLengths));
        foreach ($docLengths as $nodeId => $length) {
            $buf .= pack('NN', $nodeId, $length);
        }

        $invertedIndex = $bm25['invertedIndex'];
        $buf .= pack('N', count($invertedIndex));
        foreach ($invertedIndex as $term => $postings) {
            $termBytes = (string) $term;
            $buf .= pack('n', strlen($termBytes)) . $termBytes;
            $buf .= pack('N', count($postings));
            foreach ($postings as $postNodeId => $tf) {
                $buf .= pack('NN', $postNodeId, $tf);
            }
        }

        if (file_put_contents($path, $buf) === false) {
            throw new \RuntimeException("Failed to write to: {$path}");
        }
    }

    /**
     * Read a .phpv file and return the decoded state array.
     *
     * @return array Same shape as the $state parameter of persist().
     * @throws \RuntimeException on I/O error or format mismatch.
     */
    public function load(string $path): array
    {
        $data = @file_get_contents($path);
        if ($data === false) {
            throw new \RuntimeException("Failed to read: {$path}");
        }

        $off = 0;

        // ── HEADER ───────────────────────────────────────────────────────
        if (substr($data, $off, 4) !== self::MAGIC) {
            throw new \RuntimeException("Not a PHPVector file: {$path}");
        }
        $off += 4;

        $version  = ord($data[$off]);
        $distance = ord($data[$off + 1]);
        $off += 2;

        if ($version !== self::VERSION) {
            throw new \RuntimeException("Unsupported file version: {$version}");
        }

        [$dimension, $nodeCount, $entryPointRaw, $maxLayer, $nextId]
            = array_values(unpack('N5', $data, $off));
        $off += 20;

        $entryPoint = ($entryPointRaw === self::NULL_ENTRY_POINT) ? null : (int) $entryPointRaw;

        // ── ID-MAPPING ───────────────────────────────────────────────────
        [$jsonLen] = array_values(unpack('N', $data, $off));
        $off += 4;
        $docIdToNodeId = json_decode(substr($data, $off, $jsonLen), true, 512, JSON_THROW_ON_ERROR);
        $off += $jsonLen;

        // ── NODE SECTION ─────────────────────────────────────────────────
        $nodes      = [];
        $nodeIdList = [];

        for ($i = 0; $i < $nodeCount; $i++) {
            [$nodeId, $nodeMaxLayer] = array_values(unpack('N2', $data, $off));
            $off += 8;

            if ($dimension > 0) {
                $vector = array_values(unpack('d' . $dimension, $data, $off));
                $off += $dimension * 8;
            } else {
                $vector = [];
            }

            $connections = [];
            for ($l = 0; $l <= $nodeMaxLayer; $l++) {
                [$connCount] = array_values(unpack('N', $data, $off));
                $off += 4;
                if ($connCount > 0) {
                    $connections[$l] = array_values(unpack('N' . $connCount, $data, $off));
                    $off += $connCount * 4;
                } else {
                    $connections[$l] = [];
                }
            }

            $nodes[(int) $nodeId] = [
                'maxLayer'    => (int) $nodeMaxLayer,
                'vector'      => $vector,
                'connections' => $connections,
            ];
            $nodeIdList[] = (int) $nodeId;
        }

        // ── DOCUMENT SECTION (positional — same order as nodes) ──────────
        $documents = [];

        foreach ($nodeIdList as $nodeId) {
            $idType = ord($data[$off]);
            $off += 1;

            if ($idType === 0) {
                [$id] = array_values(unpack('q', $data, $off));
                $off += 8;
            } else {
                [$idLen] = array_values(unpack('N', $data, $off));
                $off += 4;
                $id  = substr($data, $off, $idLen);
                $off += $idLen;
            }

            [$textLen] = array_values(unpack('N', $data, $off));
            $off += 4;
            if ($textLen > 0) {
                $text = substr($data, $off, $textLen);
                $off += $textLen;
            } else {
                $text = null;
            }

            [$metaLen] = array_values(unpack('N', $data, $off));
            $off += 4;
            if ($metaLen > 0) {
                $metadata = json_decode(substr($data, $off, $metaLen), true, 512, JSON_THROW_ON_ERROR);
                $off += $metaLen;
            } else {
                $metadata = [];
            }

            $documents[$nodeId] = [
                'id'       => $id,
                'text'     => $text,
                'metadata' => $metadata,
            ];
        }

        // ── BM25 SECTION ─────────────────────────────────────────────────
        [$totalTokens] = array_values(unpack('N', $data, $off));
        $off += 4;

        [$docLenCount] = array_values(unpack('N', $data, $off));
        $off += 4;

        $docLengths = [];
        for ($i = 0; $i < $docLenCount; $i++) {
            [$dlNodeId, $dlLength] = array_values(unpack('N2', $data, $off));
            $off += 8;
            $docLengths[(int) $dlNodeId] = (int) $dlLength;
        }

        [$termCount] = array_values(unpack('N', $data, $off));
        $off += 4;

        $invertedIndex = [];
        for ($i = 0; $i < $termCount; $i++) {
            [$termLen] = array_values(unpack('n', $data, $off));
            $off += 2;
            $term = substr($data, $off, $termLen);
            $off += $termLen;

            [$postingCount] = array_values(unpack('N', $data, $off));
            $off += 4;

            $postings = [];
            for ($j = 0; $j < $postingCount; $j++) {
                [$postNodeId, $tf] = array_values(unpack('N2', $data, $off));
                $off += 8;
                $postings[(int) $postNodeId] = (int) $tf;
            }
            $invertedIndex[$term] = $postings;
        }

        return [
            'distance'      => (int) $distance,
            'dimension'     => (int) $dimension,
            'nodeCount'     => (int) $nodeCount,
            'entryPoint'    => $entryPoint,
            'maxLayer'      => (int) $maxLayer,
            'nextId'        => (int) $nextId,
            'docIdToNodeId' => $docIdToNodeId,
            'nodes'         => $nodes,
            'documents'     => $documents,
            'bm25'          => [
                'totalTokens'   => (int) $totalTokens,
                'docLengths'    => $docLengths,
                'invertedIndex' => $invertedIndex,
            ],
        ];
    }
}
