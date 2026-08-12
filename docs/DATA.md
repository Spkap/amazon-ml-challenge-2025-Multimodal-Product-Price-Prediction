# Data and artifacts

[← Project overview](../README.md) · [Setup](SETUP.md)

## Dataset contract

The prepared challenge data contains **74,999 training rows** and **75,000 test rows**.

| File | Required columns |
|---|---|
| Training | `sample_id`, `catalog_content`, `image_link`, `price` |
| Test | `sample_id`, `catalog_content`, `image_link` |
| Submission | `sample_id`, `price` |

`sample_id` is the alignment key. The pipeline stores ordered IDs beside every feature matrix,
checks them before concatenation, and writes predictions in the exact test-row order.

## Feature matrix contract

| Columns | Block | Shape |
|---:|---|---:|
| `0:1024` | SigLIP2 image projection | 1,024 |
| `1024:2048` | SigLIP2 text projection | 1,024 |
| `2048:2049` | Image–text cosine similarity | 1 |
| `2049:2433` | MiniLM text embedding | 384 |
| `2433:2450` | Engineered catalogue features | 17 |

Train and test matrices are saved as `float32`. Their manifests record ordered feature names,
source hashes, ordered-ID hashes, dimensions, dtypes, checkpoint revisions, and the canonical
configuration hash. Writes are atomic, and downstream stages verify artifact checksums before
reusing saved work.

## Local layout

```text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   └── kaggle/                  # Optional authenticated downloads
└── samples/                     # Public fictional fixtures

artifacts/final/
├── embeddings/
├── features/
│   ├── train.npy
│   ├── test.npy
│   ├── target.npy
│   ├── train_ids.csv
│   ├── test_ids.csv
│   ├── folds.csv
│   └── manifest.json
└── ensemble/
    ├── models/
    ├── oof_predictions.npy             # Four OOF prediction columns
    ├── test_predictions.npy            # Four test prediction columns
    ├── blended_oof_predictions.npy
    ├── blended_test_predictions.npy
    ├── model_columns.json
    ├── weights.json
    ├── metrics.json
    └── submission.csv
```

These directories are local working data. Git retains the public fictional fixtures and metadata,
but ignores generated matrices, images, checkpoints, predictions, and submissions.

## Recovered artifact manifest

[`archive/recovered/final-artifacts.json`](../archive/recovered/final-artifacts.json) records the
checksums, shapes, dtypes, row counts, and Kaggle identifiers associated with the final feature
artifacts. The manifest verifies provenance without adding large binaries to Git.

## Data validation

```bash
amazon-ml-price validate-data \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv
```

Validation rejects missing columns, duplicated identifiers, invalid targets, and malformed row
order before feature extraction begins.

Feature assembly additionally rejects:

- mismatched ordered-ID files;
- incompatible configuration or source hashes;
- missing feature blocks;
- incorrect dimensions or dtypes;
- non-finite feature values.

Unavailable product images remain in place and receive a zero image vector. A separate
unavailability mask records those rows without changing the 2,450-column matrix.

## Data distribution

This repository does not redistribute the full challenge tables, product-image corpus, feature
arrays, fitted models, or competition submissions. Obtain challenge-scale inputs through an
authorized source and follow its terms of use.

Never commit Kaggle credentials, cookies, signed download URLs, private dataset references, or
credential-bearing logs. Review notebook outputs and generated manifests before publishing them.
