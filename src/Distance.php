<?php

declare(strict_types=1);

namespace PHPVector;

enum Distance
{
    /** 1 − cosine_similarity. Range [0, 2]. Best for normalized embeddings. */
    case Cosine;

    /** Euclidean (L2) distance. Best for raw, unnormalized vectors. */
    case Euclidean;

    /** Negative dot-product. Best when vectors are already unit-normalized (equivalent to Cosine, but faster). */
    case DotProduct;

    /** Manhattan (L1) distance. Robust to outliers. */
    case Manhattan;
}
