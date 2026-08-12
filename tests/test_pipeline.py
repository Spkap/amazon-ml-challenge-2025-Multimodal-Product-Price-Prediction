import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from amazon_ml_price.data import (
    align_by_sample_id,
    build_submission,
    load_pipeline_config,
    ordered_id_hash,
    save_npy,
    validate_frame,
    validate_npy_artifact,
)
from amazon_ml_price.features import (
    CATALOGUE_FEATURE_NAMES,
    FEATURE_BLOCKS,
    FINAL_FEATURE_DIM,
    CatalogFeatureExtractor,
    MiniLMEncoder,
    Siglip2Encoder,
    assemble_feature_artifacts,
    assemble_feature_matrix,
    cosine_similarity_column,
    extract_measurement,
    extract_pack_quantity,
    write_feature_block,
)
from amazon_ml_price.models import (
    MODEL_ORDER,
    _feature_matrix,
    blend_predictions,
    create_shared_folds,
    load_fold_assignments,
    optimize_smape_weights,
    save_fold_assignments,
    smape,
    validate_shared_folds,
)


def test_canonical_config_and_feature_offsets():
    config, digest = load_pipeline_config("configs/final.json")
    assert config["feature_width"] == FINAL_FEATURE_DIM == 2450
    assert [block[2] - block[1] for block in FEATURE_BLOCKS] == [1024, 1024, 1, 384, 17]
    assert [block["name"] for block in config["feature_blocks"]] == [
        block[0] for block in FEATURE_BLOCKS
    ]
    assert config["model_order"] == list(MODEL_ORDER)
    assert len(digest) == 64


def test_catalogue_feature_contract_and_train_only_frequencies():
    train = pd.Series([
        "Item Name: Acme Coffee\nBrand: Acme\nDescription: Bold roast\n1.5 kg pack of 2",
        "Item Name: Acme Tea\nBrand: Acme\nProduct Class: Drinks\n500 g",
    ])
    extractor = CatalogFeatureExtractor()
    matrix = extractor.fit_transform(train)
    unseen = extractor.transform(pd.Series(["Brand: New\nProduct Class: Other\n250 g"]))
    assert matrix.shape == (2, 17)
    assert matrix.columns.tolist() == list(CATALOGUE_FEATURE_NAMES)
    assert matrix.loc[0, "pack_quantity"] == 2
    assert matrix.loc[0, "base_unit_value"] == 1500
    assert matrix.loc[0, "total_declared_value"] == 3000
    assert unseen.loc[0, "brand_frequency"] == 0
    assert unseen.loc[0, "product_class_frequency"] == 0
    assert extract_pack_quantity("six bars, pack of 6") == 6
    assert extract_measurement("500 millilitres") == (500.0, "ml")


def test_embedding_width_guards_and_zero_cosine():
    with pytest.raises(ValueError, match="1024"):
        Siglip2Encoder(expected_dim=768)
    with pytest.raises(ValueError, match="384"):
        MiniLMEncoder(expected_dim=512)
    image = np.zeros((2, 1024), dtype=np.float32)
    text = np.zeros((2, 1024), dtype=np.float32)
    text[:, 0] = 1
    cosine = cosine_similarity_column(image, text)
    assert cosine.shape == (2, 1)
    np.testing.assert_array_equal(cosine, 0)


def test_final_matrix_block_order_and_validation():
    blocks = {
        "siglip2_image": np.full((2, 1024), 1, dtype=np.float32),
        "siglip2_text": np.full((2, 1024), 2, dtype=np.float32),
        "siglip2_cosine": np.full((2, 1), 3, dtype=np.float32),
        "minilm_text": np.full((2, 384), 4, dtype=np.float32),
        "catalogue": np.full((2, 17), 5, dtype=np.float32),
    }
    matrix = assemble_feature_matrix(**blocks)
    assert matrix.shape == (2, 2450)
    for (_, start, stop), expected in zip(FEATURE_BLOCKS, range(1, 6), strict=True):
        np.testing.assert_array_equal(matrix[:, start:stop], expected)
    blocks["catalogue"] = np.zeros((2, 18), dtype=np.float32)
    with pytest.raises(ValueError, match="17"):
        assemble_feature_matrix(**blocks)


def test_feature_sidecars_reject_reordered_ids_and_hashes(tmp_path):
    ids = ["a", "b"]
    paths = {}
    widths = dict(zip([item[0] for item in FEATURE_BLOCKS], [1024, 1024, 1, 384, 17]))
    for name, width in widths.items():
        path = tmp_path / f"{name}.npy"
        write_feature_block(
            path,
            np.zeros((2, width), dtype=np.float32),
            ids,
            block_name=name,
            config_hash="config",
            source_hash="source",
        )
        paths[name] = path
    matrix, manifest = assemble_feature_artifacts(
        paths, ids, config_hash="config", source_hash="source"
    )
    assert matrix.shape == (2, 2450)
    assert manifest["ordered_id_hash"] == ordered_id_hash(ids)
    with pytest.raises(ValueError, match="Ordered sample IDs"):
        assemble_feature_artifacts(
            paths, list(reversed(ids)), config_hash="config", source_hash="source"
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        assemble_feature_artifacts(
            paths, ids, config_hash="different", source_hash="source"
        )
    save_npy(paths["catalogue"], np.ones((2, 17), dtype=np.float32))
    with pytest.raises(ValueError, match="checksum mismatch"):
        assemble_feature_artifacts(
            paths, ids, config_hash="config", source_hash="source"
        )


def test_shared_folds_cover_every_row_once(tmp_path):
    ids = [f"sample-{index}" for index in range(25)]
    folds = create_shared_folds(ids, n_splits=5, random_state=42)
    validate_shared_folds(folds, ids)
    assert sorted(np.bincount(folds.folds).tolist()) == [5, 5, 5, 5, 5]
    validation_rows = np.concatenate([valid for _, valid in folds.split()])
    np.testing.assert_array_equal(np.sort(validation_rows), np.arange(25))
    with pytest.raises(ValueError, match="row order"):
        validate_shared_folds(folds, list(reversed(ids)))
    path = tmp_path / "folds.csv"
    save_fold_assignments(folds, path, config_hash="config")
    restored = load_fold_assignments(path, expected_ids=ids, config_hash="config")
    np.testing.assert_array_equal(restored.folds, folds.folds)
    assert restored.manifest_hash == folds.manifest_hash


def test_smape_optimizer_respects_four_model_contract():
    target = np.array([10.0, 20.0, 30.0, 40.0])
    predictions = np.column_stack([
        target,
        target * 1.1,
        target * 0.9,
        target + 2,
    ])
    weights = optimize_smape_weights(predictions, target, minimum_weight=0.01)
    assert weights.shape == (4,)
    assert np.all(weights >= 0.01 - 1e-8)
    assert weights.sum() == pytest.approx(1.0)
    blended = blend_predictions(predictions, weights)
    assert np.isfinite(blended).all() and (blended > 0).all()
    assert smape(target, blended) <= smape(target, predictions.mean(axis=1))


def test_metric_and_dense_sparse_feature_contracts():
    assert smape(np.array([0.0, 10.0]), np.array([0.0, 10.0])) == 0.0
    assert smape(np.array([100.0]), np.array([50.0])) == pytest.approx(66.6666667)
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    assert sparse.isspmatrix_csr(_feature_matrix(matrix))
    with pytest.raises(ValueError, match="two-dimensional"):
        _feature_matrix(np.array([1.0, 2.0]))


def test_schema_alignment_and_submission_contracts():
    frame = pd.DataFrame({"sample_id": [2, 1], "value": [20, 10]})
    assert align_by_sample_id([1, 2], frame)["value"].tolist() == [10, 20]
    invalid = pd.DataFrame({
        "sample_id": [1], "catalog_content": ["x"], "image_link": ["x.jpg"], "price": [0]
    })
    with pytest.raises(ValueError, match="greater than zero"):
        validate_frame(invalid, training=True)
    with pytest.raises(ValueError, match="greater than zero"):
        build_submission([1, 2], [1.0, np.nan])
    result = build_submission([1, 2], [2.5, 3.0])
    assert result.columns.tolist() == ["sample_id", "price"]


def test_config_rejects_feature_contract_drift(tmp_path):
    with open("configs/final.json", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["feature_blocks"][-1]["stop"] = 2451
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="feature block contract"):
        load_pipeline_config(path)


def test_atomic_numpy_artifact_contract(tmp_path):
    path = tmp_path / "matrix.npy"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_npy(path, values)
    restored = validate_npy_artifact(
        path,
        expected_shape=(3, 4),
        expected_dtype="float32",
    )
    np.testing.assert_array_equal(restored, values)
    assert not list(tmp_path.glob(".matrix.npy.*"))
