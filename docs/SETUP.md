# Setup and usage

[← Project overview](../README.md) · [Data and artifacts](DATA.md)

Python 3.10 or newer is required. Feature extraction downloads the selected Hugging Face
checkpoints unless they are already available in the local cache.

## Install

Install the complete pipeline with `uv`:

```bash
uv sync --extra all
source .venv/bin/activate
```

Or use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,dev]'
```

Available extras are `models`, `embeddings`, `kaggle`, `dev`, and `all`. LightGBM and XGBoost use
OpenMP; on macOS, install the runtime with `brew install libomp`.

## Prepare the data

Place authorized challenge files at:

```text
data/raw/train.csv
data/raw/test.csv
```

Validate their schema, identifiers, and training targets:

```bash
amazon-ml-price validate-data \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv
```

The repository also includes fictional fixtures under `data/samples/` for inspecting the schema.

## Run the complete pipeline

```bash
amazon-ml-run \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --artifact-dir artifacts/final \
  --config configs/final.json
```

The runner extracts SigLIP2 and MiniLM embeddings, builds the 17 catalogue features, assembles the
2,450-column matrices, creates the shared five-fold plan, trains the four models, learns ensemble
weights from out-of-fold predictions, refits each model, and writes the validated submission.
Completed stages are reused only when their configuration, source, ordered-ID, shape, dtype, and
SHA-256 contracts still match. Partial or stale artifacts are rebuilt.

## Run individual stages

```bash
amazon-ml-price extract-embeddings \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --output artifacts/final/embeddings \
  --config configs/final.json

amazon-ml-price build-features \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --embeddings artifacts/final/embeddings \
  --output artifacts/final/features \
  --config configs/final.json

amazon-ml-price train-ensemble \
  --features artifacts/final/features \
  --output artifacts/final/ensemble \
  --config configs/final.json

amazon-ml-price make-submission \
  --test-ids artifacts/final/features/test_ids.csv \
  --predictions artifacts/final/ensemble/blended_test_predictions.npy \
  --output artifacts/final/ensemble/submission.csv
```

The default embedding models are:

- `google/siglip2-large-patch16-384`, producing 1,024-dimensional image and text projections;
- `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional text embeddings.

Omit `--device` for automatic CUDA, Apple MPS, or CPU selection, or pass `cuda`, `mps`, or `cpu`
explicitly. Batch sizes can be adjusted without changing the feature contract.

## Outputs

```text
artifacts/final/
├── embeddings/       # SigLIP2 and MiniLM blocks, ordered IDs, and manifests
├── features/         # 2,450-column matrices, targets, fold map, and manifest
└── ensemble/
    ├── models/       # Four fitted models and model manifests
    └── …             # Base/blended predictions, weights, metrics, and submission.csv
```

Generated arrays, model files, downloaded images, and submissions are ignored by Git.

## Download selected Kaggle assets

The helper uses the official Kaggle client and its standard authentication configuration:

```bash
python -m pip install -e '.[kaggle]'
python scripts/download_kaggle_assets.py owner/dataset --list-only

python scripts/download_kaggle_assets.py owner/dataset \
  --file path/in/dataset.csv \
  --dest data/raw/kaggle
```

Downloads are limited to 100 MiB by default. Larger files require `--allow-large` and sufficient
local disk space. See [Data and artifacts](DATA.md) before storing or publishing challenge data.
