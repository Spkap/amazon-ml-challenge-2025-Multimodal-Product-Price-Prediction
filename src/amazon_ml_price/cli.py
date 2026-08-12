"""Command-line workflow for the final multimodal price-prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import (
    canonical_json_hash,
    file_sha256,
    load_challenge_csv,
    load_pipeline_config,
    ordered_id_hash,
    read_ordered_ids,
    save_npy,
    validate_npy_artifact,
    write_json,
    write_ordered_ids,
    write_submission,
)
from .features import (
    CATALOGUE_FEATURE_NAMES,
    CatalogFeatureExtractor,
    MiniLMEncoder,
    Siglip2Encoder,
    assemble_feature_artifacts,
    load_feature_block,
    write_feature_block,
)
from .models import (
    create_shared_folds,
    load_fold_assignments,
    save_ensemble_predictions,
    save_fold_assignments,
    save_model_predictions,
    train_four_model_ensemble,
)

DEFAULT_CONFIG = "configs/final.json"


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="canonical pipeline JSON")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amazon-ml-price")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data", help="validate challenge CSV files")
    validate.add_argument("--train-csv")
    validate.add_argument("--test-csv")

    extract = commands.add_parser(
        "extract-embeddings", help="extract aligned SigLIP2 and MiniLM feature blocks"
    )
    extract.add_argument("--train-csv", required=True)
    extract.add_argument("--test-csv", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--cache-dir")
    extract.add_argument("--device")
    _add_config(extract)

    build = commands.add_parser(
        "build-features", help="assemble the canonical 2,450-column matrices"
    )
    build.add_argument("--train-csv", required=True)
    build.add_argument("--test-csv", required=True)
    build.add_argument("--embeddings", required=True)
    build.add_argument("--output", required=True)
    _add_config(build)

    train = commands.add_parser(
        "train-ensemble", help="train four base models and the OOF-weighted ensemble"
    )
    train.add_argument("--features", required=True)
    train.add_argument("--output", required=True)
    _add_config(train)

    submission = commands.add_parser(
        "make-submission", help="write an ID-aligned positive-price submission"
    )
    submission.add_argument("--test-ids", required=True)
    submission.add_argument("--predictions", required=True)
    submission.add_argument("--output", required=True)
    return parser


def _embedding_paths(root: Path, split: str) -> dict[str, Path]:
    return {
        "siglip2_image": root / f"{split}_siglip2_image.npy",
        "siglip2_text": root / f"{split}_siglip2_text.npy",
        "siglip2_cosine": root / f"{split}_siglip2_cosine.npy",
        "minilm_text": root / f"{split}_minilm_text.npy",
    }


def _ensemble_artifacts_match(
    output: Path,
    *,
    config: dict[str, object],
    config_hash: str,
    feature_hash: str,
    fold_hash: str,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> bool:
    """Return whether a complete saved ensemble is safe to reuse."""
    try:
        weights = json.loads((output / "weights.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        columns = json.loads((output / "model_columns.json").read_text(encoding="utf-8"))
        model_order = list(config["model_order"])
        if (
            weights.get("config_hash") != config_hash
            or weights.get("feature_manifest_hash") != feature_hash
            or weights.get("fold_manifest_hash") != fold_hash
            or weights.get("model_order") != model_order
            or metrics.get("config_hash") != config_hash
            or metrics.get("feature_manifest_hash") != feature_hash
            or metrics.get("fold_manifest_hash") != fold_hash
            or set(metrics.get("models", {})) != set(model_order)
            or columns != {"config_hash": config_hash, "model_order": model_order}
        ):
            return False
        saved_folds = load_fold_assignments(
            output / "folds.csv",
            expected_ids=train_ids,
            config_hash=config_hash,
        )
        if saved_folds.manifest_hash != fold_hash:
            return False
        blend_weights = np.asarray(
            [weights["weights"][name] for name in model_order],
            dtype=np.float64,
        )
        minimum_weight = float(config["ensemble"]["minimum_weight"])
        if (
            not np.isfinite(blend_weights).all()
            or np.any(blend_weights < minimum_weight - 1e-8)
            or not np.isclose(blend_weights.sum(), 1.0)
        ):
            return False
        artifact_hashes = weights["artifacts"]
        validate_npy_artifact(
            output / "oof_predictions.npy",
            expected_shape=(len(train_ids), len(model_order)),
            expected_dtype="float32",
            expected_sha256=artifact_hashes["oof_predictions"],
        )
        validate_npy_artifact(
            output / "blended_oof_predictions.npy",
            expected_shape=(len(train_ids),),
            expected_dtype="float32",
            expected_sha256=artifact_hashes["blended_oof_predictions"],
        )
        validate_npy_artifact(
            output / "test_predictions.npy",
            expected_shape=(len(test_ids), len(model_order)),
            expected_dtype="float32",
            expected_sha256=artifact_hashes["test_predictions"],
        )
        blended_test = validate_npy_artifact(
            output / "blended_test_predictions.npy",
            expected_shape=(len(test_ids),),
            expected_dtype="float32",
            expected_sha256=artifact_hashes["blended_test_predictions"],
        )
        submission_path = output / "submission.csv"
        if artifact_hashes["submission"] != file_sha256(submission_path):
            return False
        submission = pd.read_csv(submission_path, dtype={"sample_id": str})
        if (
            list(submission.columns) != ["sample_id", "price"]
            or list(submission["sample_id"]) != list(map(str, test_ids))
            or not np.array_equal(
                submission["price"].to_numpy(dtype=np.float32),
                np.asarray(blended_test, dtype=np.float32),
            )
        ):
            return False
        for model_name in model_order:
            model_root = output / "models" / model_name
            manifest = json.loads(
                (model_root / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("model_name") != model_name
                or manifest.get("config_hash") != config_hash
                or manifest.get("feature_manifest_hash") != feature_hash
                or manifest.get("fold_manifest_hash") != fold_hash
            ):
                return False
            model_artifacts = manifest["artifacts"]
            validate_npy_artifact(
                model_root / model_artifacts["oof_predictions"]["file"],
                expected_shape=(len(train_ids),),
                expected_dtype="float32",
                expected_sha256=model_artifacts["oof_predictions"]["sha256"],
            )
            validate_npy_artifact(
                model_root / model_artifacts["test_predictions"]["file"],
                expected_shape=(len(test_ids),),
                expected_dtype="float32",
                expected_sha256=model_artifacts["test_predictions"]["sha256"],
            )
            model_artifact = model_artifacts["model"]
            if file_sha256(model_root / model_artifact["file"]) != model_artifact["sha256"]:
                return False
        return True
    except (AttributeError, IndexError, OSError, KeyError, TypeError, ValueError):
        return False


def _feature_artifacts_match(
    output: Path,
    *,
    config: dict[str, object],
    config_hash: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_csv: str,
    test_csv: str,
) -> bool:
    """Return whether a complete saved feature bundle is safe to reuse."""
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        expected_artifacts = {
            "train_features",
            "test_features",
            "target",
            "train_ids",
            "test_ids",
            "folds",
            "catalogue_extractor",
        }
        if (
            manifest.get("config_hash") != config_hash
            or set(artifacts) != expected_artifacts
            or manifest["splits"]["train"].get("source_hash") != file_sha256(train_csv)
            or manifest["splits"]["test"].get("source_hash") != file_sha256(test_csv)
        ):
            return False
        expected_bundle_hash = canonical_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        if manifest.get("manifest_hash") != expected_bundle_hash:
            return False
        train_ids = train["sample_id"].tolist()
        test_ids = test["sample_id"].tolist()
        saved_train_ids = read_ordered_ids(output / "train_ids.csv")
        saved_test_ids = read_ordered_ids(output / "test_ids.csv")
        if (
            list(map(str, saved_train_ids)) != list(map(str, train_ids))
            or list(map(str, saved_test_ids)) != list(map(str, test_ids))
            or artifacts["train_ids"]["sha256"]
            != file_sha256(output / "train_ids.csv")
            or artifacts["test_ids"]["sha256"]
            != file_sha256(output / "test_ids.csv")
            or artifacts["folds"]["sha256"] != file_sha256(output / "folds.csv")
            or artifacts["catalogue_extractor"]["sha256"]
            != file_sha256(output / "catalogue_extractor.joblib")
        ):
            return False
        for split, artifact_name in (
            ("train", "train_features"),
            ("test", "test_features"),
        ):
            split_manifest = manifest["splits"][split]
            expected_manifest_hash = canonical_json_hash(
                {
                    key: value
                    for key, value in split_manifest.items()
                    if key != "manifest_hash"
                }
            )
            if (
                split_manifest.get("manifest_hash") != expected_manifest_hash
                or split_manifest.get("matrix_sha256")
                != artifacts[artifact_name]["sha256"]
                or split_manifest.get("ids_sha256")
                != artifacts[f"{split}_ids"]["sha256"]
            ):
                return False
        validate_npy_artifact(
            output / "train.npy",
            expected_shape=(len(train), int(config["feature_width"])),
            expected_dtype=str(config["feature_dtype"]),
            expected_sha256=artifacts["train_features"]["sha256"],
        )
        validate_npy_artifact(
            output / "test.npy",
            expected_shape=(len(test), int(config["feature_width"])),
            expected_dtype=str(config["feature_dtype"]),
            expected_sha256=artifacts["test_features"]["sha256"],
        )
        saved_target = validate_npy_artifact(
            output / "target.npy",
            expected_shape=(len(train),),
            expected_dtype="float32",
            expected_sha256=artifacts["target"]["sha256"],
        )
        if not np.array_equal(
            saved_target,
            train["price"].to_numpy(dtype=np.float32),
        ):
            return False
        folds = load_fold_assignments(
            output / "folds.csv",
            expected_ids=train_ids,
            config_hash=config_hash,
        )
        return folds.manifest_hash == manifest.get("fold_manifest_hash")
    except (AttributeError, IndexError, OSError, KeyError, TypeError, ValueError):
        return False


def _extract_embeddings(args: argparse.Namespace) -> int:
    config, config_hash = load_pipeline_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    siglip_config = config["embeddings"]["siglip2"]
    minilm_config = config["embeddings"]["minilm"]
    siglip = Siglip2Encoder(
        checkpoint=siglip_config["checkpoint"],
        revision=siglip_config["revision"],
        expected_dim=int(siglip_config["projection_dim"]),
        max_text_length=int(siglip_config["max_text_length"]),
        batch_size=int(siglip_config["batch_size"]),
        device=args.device,
        image_timeout=float(siglip_config["image_timeout_seconds"]),
        image_retries=int(siglip_config["image_retries"]),
        max_image_bytes=int(siglip_config["max_image_bytes"]),
    )
    minilm = MiniLMEncoder(
        checkpoint=minilm_config["checkpoint"],
        revision=minilm_config["revision"],
        expected_dim=int(minilm_config["embedding_dim"]),
        max_sequence_length=int(minilm_config["max_sequence_length"]),
        batch_size=int(minilm_config["batch_size"]),
        device=args.device,
    )
    for split, csv_path, training in (
        ("train", args.train_csv, True),
        ("test", args.test_csv, False),
    ):
        frame = load_challenge_csv(csv_path, training=training)
        ids = frame["sample_id"].tolist()
        texts = frame["catalog_content"].fillna("").tolist()
        source_hash = file_sha256(csv_path)
        expected_widths = {
            "siglip2_image": 1024,
            "siglip2_text": 1024,
            "siglip2_cosine": 1,
            "minilm_text": 384,
        }
        split_paths = _embedding_paths(output, split)
        try:
            for name, width in expected_widths.items():
                load_feature_block(
                    split_paths[name],
                    ids,
                    block_name=name,
                    expected_width=width,
                    config_hash=config_hash,
                    source_hash=source_hash,
                )
            unavailable_path = output / f"{split}_image_unavailable.npy"
            unavailable_manifest_path = output / f"{split}_image_unavailable.json"
            unavailable_manifest = json.loads(
                unavailable_manifest_path.read_text(encoding="utf-8")
            )
            unavailable = validate_npy_artifact(
                unavailable_path,
                expected_shape=(len(ids),),
                expected_dtype="uint8",
                expected_sha256=unavailable_manifest["sha256"],
                finite=False,
            )
            if (
                unavailable_manifest.get("config_hash") != config_hash
                or unavailable_manifest.get("source_hash") != source_hash
                or unavailable_manifest.get("ordered_id_hash") != ordered_id_hash(ids)
                or not np.isin(unavailable, (0, 1)).all()
            ):
                raise ValueError("Image-unavailability artifact does not match its inputs")
            continue
        except (FileNotFoundError, ValueError):
            pass
        images, siglip_text, cosine, failures = siglip.encode(
            frame["image_link"].fillna("").tolist(),
            texts,
            cache_dir=args.cache_dir,
        )
        minilm_text = minilm.encode(texts)
        blocks = {
            "siglip2_image": images,
            "siglip2_text": siglip_text,
            "siglip2_cosine": cosine,
            "minilm_text": minilm_text,
        }
        for name, matrix in blocks.items():
            metadata = {
                "checkpoint": (
                    minilm_config["checkpoint"] if name == "minilm_text"
                    else siglip_config["checkpoint"]
                ),
                "revision": (
                    minilm_config["revision"] if name == "minilm_text"
                    else siglip_config["revision"]
                ),
                "batch_size": (
                    minilm_config["batch_size"] if name == "minilm_text"
                    else siglip_config["batch_size"]
                ),
                "device": args.device or "auto",
            }
            write_feature_block(
                split_paths[name],
                matrix,
                ids,
                block_name=name,
                config_hash=config_hash,
                source_hash=source_hash,
                metadata=metadata,
            )
        unavailable_path = output / f"{split}_image_unavailable.npy"
        save_npy(unavailable_path, failures)
        write_json(
            output / f"{split}_image_unavailable.json",
            {
                "rows": len(ids),
                "dtype": "uint8",
                "config_hash": config_hash,
                "source_hash": source_hash,
                "ordered_id_hash": ordered_id_hash(ids),
                "sha256": file_sha256(unavailable_path),
                "unavailable_images": int(failures.sum()),
            },
        )
        del frame, images, siglip_text, cosine, minilm_text
    write_json(output / "config.snapshot.json", config)
    return 0


def _build_features(args: argparse.Namespace) -> int:
    config, config_hash = load_pipeline_config(args.config)
    output = Path(args.output)
    blocks_dir = output / "blocks"
    output.mkdir(parents=True, exist_ok=True)
    train = load_challenge_csv(args.train_csv, training=True)
    test = load_challenge_csv(args.test_csv, training=False)
    if _feature_artifacts_match(
        output,
        config=config,
        config_hash=config_hash,
        train=train,
        test=test,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
    ):
        return 0
    extractor = CatalogFeatureExtractor()
    train_catalogue = extractor.fit_transform(train["catalog_content"])
    test_catalogue = extractor.transform(test["catalog_content"])
    embedding_root = Path(args.embeddings)
    split_data = (
        ("train", train, train_catalogue, args.train_csv),
        ("test", test, test_catalogue, args.test_csv),
    )
    manifests: dict[str, object] = {}
    for split, frame, catalogue, csv_path in split_data:
        ids = frame["sample_id"].tolist()
        source_hash = file_sha256(csv_path)
        catalogue_path = blocks_dir / f"{split}_catalogue.npy"
        write_feature_block(
            catalogue_path,
            catalogue.to_numpy(dtype=np.float32),
            ids,
            block_name="catalogue",
            config_hash=config_hash,
            source_hash=source_hash,
            metadata={"feature_names": list(CATALOGUE_FEATURE_NAMES)},
        )
        paths = {**_embedding_paths(embedding_root, split), "catalogue": catalogue_path}
        matrix, manifest = assemble_feature_artifacts(
            paths,
            ids,
            config_hash=config_hash,
            source_hash=source_hash,
        )
        matrix_path = output / f"{split}.npy"
        ids_path = output / f"{split}_ids.csv"
        save_npy(matrix_path, matrix)
        write_ordered_ids(ids_path, ids)
        manifest["matrix_sha256"] = file_sha256(matrix_path)
        manifest["ids_sha256"] = file_sha256(ids_path)
        manifest["manifest_hash"] = canonical_json_hash(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        manifests[split] = manifest
    save_npy(output / "target.npy", train["price"].to_numpy(dtype=np.float32))
    folds = create_shared_folds(
        train["sample_id"].tolist(),
        n_splits=int(config["cross_validation"]["folds"]),
        random_state=int(config["cross_validation"]["seed"]),
    )
    save_fold_assignments(folds, output / "folds.csv", config_hash=config_hash)
    extractor_path = output / "catalogue_extractor.joblib"
    joblib.dump(extractor, extractor_path)
    combined_manifest = {
        "schema_version": config["schema_version"],
        "config_hash": config_hash,
        "feature_width": config["feature_width"],
        "feature_dtype": config["feature_dtype"],
        "feature_blocks": config["feature_blocks"],
        "catalogue_features": list(CATALOGUE_FEATURE_NAMES),
        "fold_manifest_hash": folds.manifest_hash,
        "splits": manifests,
        "artifacts": {
            "train_features": {
                "file": "train.npy",
                "sha256": file_sha256(output / "train.npy"),
            },
            "test_features": {
                "file": "test.npy",
                "sha256": file_sha256(output / "test.npy"),
            },
            "target": {
                "file": "target.npy",
                "sha256": file_sha256(output / "target.npy"),
            },
            "train_ids": {
                "file": "train_ids.csv",
                "sha256": file_sha256(output / "train_ids.csv"),
            },
            "test_ids": {
                "file": "test_ids.csv",
                "sha256": file_sha256(output / "test_ids.csv"),
            },
            "folds": {
                "file": "folds.csv",
                "sha256": file_sha256(output / "folds.csv"),
            },
            "catalogue_extractor": {
                "file": extractor_path.name,
                "sha256": file_sha256(extractor_path),
            },
        },
    }
    combined_manifest["manifest_hash"] = canonical_json_hash(combined_manifest)
    write_json(output / "manifest.json", combined_manifest)
    write_json(output / "config.snapshot.json", config)
    return 0


def _train_ensemble(args: argparse.Namespace) -> int:
    config, config_hash = load_pipeline_config(args.config)
    root = Path(args.features)
    train_ids = read_ordered_ids(root / "train_ids.csv")
    test_ids = read_ordered_ids(root / "test_ids.csv")
    feature_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if feature_manifest.get("config_hash") != config_hash:
        raise ValueError("Feature artifacts were built with a different configuration")
    expected_bundle_hash = canonical_json_hash(
        {
            key: value
            for key, value in feature_manifest.items()
            if key != "manifest_hash"
        }
    )
    if feature_manifest.get("manifest_hash") != expected_bundle_hash:
        raise ValueError("Feature bundle manifest hash mismatch")
    artifacts = feature_manifest["artifacts"]
    for split in ("train", "test"):
        split_manifest = feature_manifest["splits"][split]
        expected_manifest_hash = canonical_json_hash(
            {
                key: value
                for key, value in split_manifest.items()
                if key != "manifest_hash"
            }
        )
        if split_manifest.get("manifest_hash") != expected_manifest_hash:
            raise ValueError(f"{split} feature manifest hash mismatch")
    if (
        artifacts["train_ids"]["sha256"] != file_sha256(root / "train_ids.csv")
        or artifacts["test_ids"]["sha256"] != file_sha256(root / "test_ids.csv")
        or feature_manifest["splits"]["train"].get("ids_sha256")
        != artifacts["train_ids"]["sha256"]
        or feature_manifest["splits"]["test"].get("ids_sha256")
        != artifacts["test_ids"]["sha256"]
        or feature_manifest["splits"]["train"].get("matrix_sha256")
        != artifacts["train_features"]["sha256"]
        or feature_manifest["splits"]["test"].get("matrix_sha256")
        != artifacts["test_features"]["sha256"]
    ):
        raise ValueError("Feature artifact checksum mismatch")
    train = validate_npy_artifact(
        root / "train.npy",
        expected_shape=(len(train_ids), int(config["feature_width"])),
        expected_dtype=config["feature_dtype"],
        expected_sha256=artifacts["train_features"]["sha256"],
    )
    test = validate_npy_artifact(
        root / "test.npy",
        expected_shape=(len(test_ids), int(config["feature_width"])),
        expected_dtype=config["feature_dtype"],
        expected_sha256=artifacts["test_features"]["sha256"],
    )
    target = validate_npy_artifact(
        root / "target.npy",
        expected_shape=(len(train_ids),),
        expected_dtype="float32",
        expected_sha256=artifacts["target"]["sha256"],
    )
    feature_hash = feature_manifest["manifest_hash"]
    folds = load_fold_assignments(
        root / "folds.csv",
        expected_ids=train_ids,
        config_hash=config_hash,
    )
    if folds.manifest_hash != feature_manifest.get("fold_manifest_hash"):
        raise ValueError("Fold artifact does not match the feature manifest")
    output = Path(args.output)
    if _ensemble_artifacts_match(
        output,
        config=config,
        config_hash=config_hash,
        feature_hash=feature_hash,
        fold_hash=folds.manifest_hash,
        train_ids=train_ids,
        test_ids=test_ids,
    ):
        return 0
    result = train_four_model_ensemble(
        train,
        target,
        sample_ids=train_ids,
        test_features=test,
        folds=folds,
        n_splits=int(config["cross_validation"]["folds"]),
        random_state=int(config["cross_validation"]["seed"]),
        feature_manifest_hash=feature_hash,
        prediction_floor=float(config["target"]["prediction_floor"]),
        minimum_weight=float(config["ensemble"]["minimum_weight"]),
        model_parameters=config["models"],
    )
    for model_result in result.model_results:
        save_model_predictions(
            model_result,
            output / "models",
            config_hash=config_hash,
        )
    save_fold_assignments(
        result.fold_assignments,
        output / "folds.csv",
        config_hash=config_hash,
    )
    if result.test_predictions is None:
        raise RuntimeError("The ensemble did not produce test predictions")
    submission_path = output / "submission.csv"
    write_submission(test_ids, result.test_predictions, submission_path)
    save_ensemble_predictions(
        result,
        output,
        config_hash=config_hash,
        submission_path=submission_path,
    )
    return 0


def _make_submission(args: argparse.Namespace) -> int:
    ids = read_ordered_ids(args.test_ids)
    predictions_path = Path(args.predictions)
    if predictions_path.suffix == ".npy":
        predictions = np.load(predictions_path, allow_pickle=False).reshape(-1)
    else:
        frame = pd.read_csv(predictions_path, dtype={"sample_id": str})
        if list(frame.columns) != ["sample_id", "price"]:
            raise ValueError("Prediction CSV must contain exactly sample_id and price")
        indexed = frame.set_index(frame["sample_id"].astype(str))
        expected = [str(value) for value in ids]
        if not indexed.index.is_unique or set(indexed.index) != set(expected):
            raise ValueError("Prediction IDs do not match test IDs")
        predictions = indexed.loc[expected, "price"].to_numpy()
    write_submission(ids, predictions, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-data":
        if not args.train_csv and not args.test_csv:
            raise ValueError("Provide --train-csv, --test-csv, or both")
        if args.train_csv:
            load_challenge_csv(args.train_csv, training=True)
        if args.test_csv:
            load_challenge_csv(args.test_csv, training=False)
        return 0
    if args.command == "extract-embeddings":
        return _extract_embeddings(args)
    if args.command == "build-features":
        return _build_features(args)
    if args.command == "train-ensemble":
        return _train_ensemble(args)
    return _make_submission(args)


def _workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amazon-ml-run")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device")
    return parser


def workflow_main(argv: list[str] | None = None) -> int:
    args = _workflow_parser().parse_args(argv)
    root = Path(args.artifact_dir)
    embeddings = root / "embeddings"
    features = root / "features"
    ensemble = root / "ensemble"
    extract_args = [
        "extract-embeddings", "--train-csv", args.train_csv, "--test-csv", args.test_csv,
        "--output", str(embeddings), "--config", args.config,
    ]
    if args.cache_dir:
        extract_args.extend(("--cache-dir", args.cache_dir))
    if args.device:
        extract_args.extend(("--device", args.device))
    main(extract_args)
    main([
        "build-features", "--train-csv", args.train_csv, "--test-csv", args.test_csv,
        "--embeddings", str(embeddings), "--output", str(features), "--config", args.config,
    ])
    main([
        "train-ensemble", "--features", str(features), "--output", str(ensemble),
        "--config", args.config,
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
