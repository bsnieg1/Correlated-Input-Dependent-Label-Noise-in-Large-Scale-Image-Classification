### Multispecies Campus UJ: clean test set (expert labels, unseen photo sequences)

| training labels           | method                          | seeds   | top-1 best [%]   | top-1 last [%]   | balanced acc [%]   | NLL           | ECE [%]      |
|:--------------------------|:--------------------------------|:--------|:-----------------|:-----------------|:-------------------|:--------------|:-------------|
| -                         | SpeciesNet itself (no training) | -       | 69.41            | -                | 56.71              | -             | -            |
| Step 1: SpeciesNet labels | CE                              | 3       | 55.68 ± 0.45     | 55.60 ± 0.76     | 35.90 ± 0.70       | 2.012 ± 0.069 | 26.16 ± 1.47 |
| Step 1: SpeciesNet labels | LS (0.1)                        | 3       | 55.03 ± 1.11     | 54.81 ± 1.60     | 34.77 ± 0.60       | 1.638 ± 0.045 | 12.90 ± 1.50 |
| Step 1: SpeciesNet labels | HET-diag (R=0)                  | 3       | 56.91 ± 0.55     | 55.60 ± 0.66     | 37.60 ± 2.06       | 2.051 ± 0.079 | 25.53 ± 0.42 |
| Step 1: SpeciesNet labels | HET (R=5)                       | 3       | 56.25 ± 1.23     | 56.54 ± 1.03     | 37.38 ± 1.75       | 2.126 ± 0.070 | 24.01 ± 1.29 |
| Step 1: SpeciesNet labels | HET, softmax(μ)                 | 3       | 56.62 ± 1.64     | -                | 37.55 ± 2.12       | 2.395 ± 0.102 | 28.72 ± 1.65 |
| Step 2: expert labels     | CE                              | 3       | 70.93 ± 1.85     | 71.58 ± 0.43     | 59.31 ± 3.56       | 1.232 ± 0.060 | 14.97 ± 1.98 |
| Step 2: expert labels     | LS (0.1)                        | 3       | 70.50 ± 1.13     | 71.80 ± 0.43     | 57.57 ± 1.08       | 1.086 ± 0.017 | 5.13 ± 0.58  |
| Step 2: expert labels     | HET-diag (R=0)                  | 3       | 71.22 ± 0.90     | 71.22 ± 1.19     | 60.26 ± 3.95       | 1.224 ± 0.094 | 14.45 ± 0.62 |
| Step 2: expert labels     | HET (R=5)                       | 3       | 72.38 ± 1.27     | 72.09 ± 1.96     | 60.78 ± 2.98       | 1.218 ± 0.047 | 13.88 ± 1.69 |
| Step 2: expert labels     | HET, softmax(μ)                 | 3       | 72.38 ± 1.27     | -                | 61.06 ± 2.93       | 1.357 ± 0.038 | 15.89 ± 1.81 |
