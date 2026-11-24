### ANIMAL-10N: is the difference larger than 2 std?

| scenario                             | comparison                 | epoch   |   Δ top-1 [p.p.] |   larger std [p.p.] | verdict                |
|:-------------------------------------|:---------------------------|:--------|-----------------:|--------------------:|:-----------------------|
| A1: original labels (~8% real noise) | HET (R=5) − HET-diag (R=0) | best    |             0.07 |                0.64 | within noise (< 2 std) |
| A1: original labels (~8% real noise) | HET (R=5) − HET-diag (R=0) | last    |             0.16 |                0.7  | within noise (< 2 std) |
| A1: original labels (~8% real noise) | HET (R=5) − CE             | best    |             0.32 |                0.55 | within noise (< 2 std) |
| A1: original labels (~8% real noise) | HET (R=5) − CE             | last    |             0.24 |                0.42 | within noise (< 2 std) |
| A1: original labels (~8% real noise) | HET-diag (R=0) − CE        | best    |             0.25 |                0.64 | within noise (< 2 std) |
| A1: original labels (~8% real noise) | HET-diag (R=0) − CE        | last    |             0.08 |                0.7  | within noise (< 2 std) |
| A1: original labels (~8% real noise) | LS (0.1) − CE              | best    |             1.11 |                0.23 | difference             |
| A1: original labels (~8% real noise) | LS (0.1) − CE              | last    |             0.95 |                0.26 | difference             |
| A2: + 20% pair flips                 | HET (R=5) − HET-diag (R=0) | best    |            -1.03 |                1.44 | within noise (< 2 std) |
| A2: + 20% pair flips                 | HET (R=5) − HET-diag (R=0) | last    |            -0.03 |                0.39 | within noise (< 2 std) |
| A2: + 20% pair flips                 | HET (R=5) − CE             | best    |             1.61 |                1.44 | within noise (< 2 std) |
| A2: + 20% pair flips                 | HET (R=5) − CE             | last    |            -0.53 |                0.87 | within noise (< 2 std) |
| A2: + 20% pair flips                 | HET-diag (R=0) − CE        | best    |             2.65 |                0.74 | difference             |
| A2: + 20% pair flips                 | HET-diag (R=0) − CE        | last    |            -0.5  |                0.87 | within noise (< 2 std) |
| A2: + 20% pair flips                 | LS (0.1) − CE              | best    |             1.34 |                0.26 | difference             |
| A2: + 20% pair flips                 | LS (0.1) − CE              | last    |             0.9  |                0.87 | within noise (< 2 std) |
