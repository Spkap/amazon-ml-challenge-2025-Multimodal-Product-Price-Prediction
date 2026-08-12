# Kaggle notebook output manifest

The authenticated Kaggle API reports **1,020 output files** for
[`sourabhkap/mlc-sourabh`](https://www.kaggle.com/code/sourabhkap/mlc-sourabh), version 1: 1,000
product images and the 20 files listed below. The API does not report output sizes, so the table
uses `not reported`. [`datasets.json`](datasets.json) contains the dataset sizes reported by Kaggle.

The notebook is private. Temporary signed download URLs are not stored in Git.

| File | Size | Purpose |
|---|---:|---|
| `features_all_combined.npy` | not reported | Fused train feature matrix used by XGBoost |
| `image_dataset_for_kaggle/dataset-metadata.json` | not reported | Metadata for the generated 1,000-image sample dataset |
| `image_dataset_for_kaggle/sampled_data.csv` | not reported | Deterministic 1,000-row development sample |
| `image_pixel_values.pt` | not reported | Preprocessed train image tensors |
| `sample_ids_order.pt` | not reported | Train sample-ID order used to preserve alignment |
| `siglip_combined_features.npy` | not reported | Concatenated image and text embeddings |
| `siglip_combined_with_similarity.npy` | not reported | Concatenated embeddings plus cosine similarity |
| `siglip_image_embeddings.npy` | not reported | Raw 768-dimensional train image embeddings |
| `siglip_image_normalized.npy` | not reported | L2-normalized train image embeddings |
| `siglip_similarity_scores.npy` | not reported | Per-row image-text cosine similarity |
| `siglip_text_embeddings.npy` | not reported | Raw 768-dimensional train text embeddings |
| `siglip_text_normalized.npy` | not reported | L2-normalized train text embeddings |
| `test_features_all_combined.npy` | not reported | Fused sample-test feature matrix |
| `test_image_pixel_values.pt` | not reported | Preprocessed sample-test image tensors |
| `test_sample_ids_order.pt` | not reported | Sample-test ID order |
| `test_text_attention_mask.npy` | not reported | Sample-test tokenizer attention mask |
| `test_text_input_ids.npy` | not reported | Sample-test tokenizer IDs |
| `text_attention_mask.npy` | not reported | Train tokenizer attention mask |
| `text_input_ids.npy` | not reported | Train tokenizer IDs |
| `xgboost_final_model.json` | not reported | Persisted XGBoost regressor |
