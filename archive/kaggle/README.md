# Kaggle notebooks and metadata

[← Project overview](../../README.md) · [Final artifact manifest](../recovered/final-artifacts.json)

This directory contains the project notebooks and their Kaggle metadata.

## Notebooks

Each notebook is accompanied by a `.metadata.json` file containing its Kaggle identity, runtime
configuration, privacy setting, and attached sources.

| Kaggle reference | Local notebook | Focus |
|---|---|---|
| [`sourabhkap/mlc-sourabh`](https://www.kaggle.com/code/sourabhkap/mlc-sourabh) | [`mlc-sourabh.ipynb`](notebooks/mlc-sourabh.ipynb) | Vision-language embeddings, image–text similarity, and boosted regression |
| [`sourabhkap/amazon-ml-challenge`](https://www.kaggle.com/code/sourabhkap/amazon-ml-challenge) | [`amazon-ml-challenge.ipynb`](notebooks/amazon-ml-challenge.ipynb) | Catalogue parsing, image representations, and price regression |
| [`sourabhkap/amlc-25`](https://www.kaggle.com/code/sourabhkap/amlc-25) | [`amlc-25.ipynb`](notebooks/amlc-25.ipynb) | Full-data analysis, dimensionality reduction, LightGBM, and stacking |
| [`harshjavajunkie/csv-files`](https://www.kaggle.com/code/harshjavajunkie/csv-files) | [`csv-files.ipynb`](notebooks/csv-files.ipynb) | Structured catalogue features, MiniLM, and multimodal representations |

## Indexes

- [`manifests/datasets.json`](manifests/datasets.json) records Kaggle dataset references, versions,
  file names, and byte counts.
- [`manifests/notebooks.json`](manifests/notebooks.json) records cell counts, local byte counts, and
  SHA-256 checksums.
- [`manifests/NOTEBOOK_OUTPUTS.md`](manifests/NOTEBOOK_OUTPUTS.md) indexes saved notebook outputs.
- [`../recovered/final-artifacts.json`](../recovered/final-artifacts.json) records the final
  2,450-feature artifact contract and checksums.

Large arrays and downloaded data are not committed. Use
[`../../scripts/download_kaggle_assets.py`](../../scripts/download_kaggle_assets.py) with an
authenticated Kaggle account and an explicit size limit.
