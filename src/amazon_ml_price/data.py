"""Challenge schema, configuration, artifact, and submission contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChallengeSchema:
    id_column: str = "sample_id"
    text_column: str = "catalog_content"
    image_column: str = "image_link"
    target_column: str = "price"

    def required_columns(self, *, training: bool) -> tuple[str, ...]:
        base = (self.id_column, self.text_column, self.image_column)
        return base + ((self.target_column,) if training else ())


DEFAULT_SCHEMA = ChallengeSchema()


def canonical_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_hash(sample_ids: Iterable[object]) -> str:
    """Hash IDs in order using their stable CSV representation."""
    values = [str(value) for value in sample_ids]
    return canonical_json_hash(values)


def load_pipeline_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate the complete canonical pipeline contract."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported pipeline schema_version")
    blocks = config.get("feature_blocks", [])
    expected_start = 0
    for block in blocks:
        start = int(block["start"])
        stop = int(block["stop"])
        width = int(block["width"])
        if start != expected_start or stop - start != width:
            raise ValueError(f"Invalid feature block contract: {block}")
        expected_start = stop
    if expected_start != int(config.get("feature_width", -1)):
        raise ValueError("Feature blocks do not match feature_width")
    expected_names = [
        "siglip2_image",
        "siglip2_text",
        "siglip2_cosine",
        "minilm_text",
        "catalogue",
    ]
    if [block.get("name") for block in blocks] != expected_names:
        raise ValueError(f"Canonical feature block order must be {expected_names}")
    expected = [1024, 1024, 1, 384, 17]
    if [int(block["width"]) for block in blocks] != expected:
        raise ValueError(f"Canonical feature widths must be {expected}")
    if config.get("feature_dtype") != "float32":
        raise ValueError("Canonical features must use float32")
    if len(config.get("catalogue_features", [])) != 17:
        raise ValueError("The catalogue feature contract must contain 17 names")
    if len(set(config["catalogue_features"])) != 17:
        raise ValueError("Catalogue feature names must be unique")

    embeddings = config.get("embeddings", {})
    siglip2 = embeddings.get("siglip2", {})
    minilm = embeddings.get("minilm", {})
    if siglip2.get("checkpoint") != "google/siglip2-large-patch16-384":
        raise ValueError("The canonical SigLIP2 checkpoint is invalid")
    if int(siglip2.get("projection_dim", -1)) != 1024:
        raise ValueError("SigLIP2 must produce 1,024-dimensional projections")
    if int(siglip2.get("max_text_length", -1)) != 64:
        raise ValueError("SigLIP2 maximum text length must be 64")
    if minilm.get("checkpoint") != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("The canonical MiniLM checkpoint is invalid")
    if int(minilm.get("embedding_dim", -1)) != 384:
        raise ValueError("MiniLM must produce 384-dimensional embeddings")
    if int(minilm.get("max_sequence_length", -1)) != 64:
        raise ValueError("MiniLM maximum sequence length must be 64")
    if minilm.get("normalize_embeddings") is not False:
        raise ValueError("MiniLM embeddings must remain unnormalized")
    for model in (siglip2, minilm):
        revision = str(model.get("revision", ""))
        invalid_revision = len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        )
        if invalid_revision:
            raise ValueError("Embedding model revisions must be pinned Git commit hashes")

    cross_validation = config.get("cross_validation", {})
    if (
        int(cross_validation.get("folds", -1)) != 5
        or cross_validation.get("shuffle") is not True
        or int(cross_validation.get("seed", -1)) != 42
    ):
        raise ValueError("Cross-validation must use five shuffled folds with seed 42")
    target = config.get("target", {})
    if target.get("transform") != "log1p" or float(target.get("prediction_floor", 0)) <= 0:
        raise ValueError("Target configuration must use log1p and a positive prediction floor")
    model_order = ["lightgbm", "xgboost", "catboost", "mlp"]
    if config.get("model_order") != model_order:
        raise ValueError(f"Canonical model order must be {model_order}")
    if set(config.get("models", {})) != set(model_order):
        raise ValueError("Configuration must define exactly the four ensemble models")
    models = config["models"]
    positive_parameters = {
        "lightgbm": ("n_estimators", "learning_rate", "num_leaves"),
        "xgboost": ("n_estimators", "learning_rate", "max_depth"),
        "catboost": ("iterations", "learning_rate", "depth"),
        "mlp": (
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
        ),
    }
    for model_name, parameter_names in positive_parameters.items():
        if any(float(models[model_name].get(name, 0)) <= 0 for name in parameter_names):
            raise ValueError(f"{model_name} parameters must be positive")
    hidden_sizes = models["mlp"].get("hidden_sizes", [])
    dropout = models["mlp"].get("dropout", [])
    if (
        len(hidden_sizes) != 2
        or any(int(width) <= 0 for width in hidden_sizes)
        or len(dropout) != 2
        or any(not 0 <= float(rate) < 1 for rate in dropout)
    ):
        raise ValueError("MLP must define two positive hidden widths and valid dropout rates")
    ensemble = config.get("ensemble", {})
    minimum_weight = float(ensemble.get("minimum_weight", -1))
    if (
        ensemble.get("method") != "slsqp_smape"
        or ensemble.get("sum_to_one") is not True
        or ensemble.get("initializations") != "equal_and_near_vertices"
        or not 0 < minimum_weight < 0.25
    ):
        raise ValueError("Invalid constrained SMAPE ensemble configuration")
    return config, canonical_json_hash(config)


def validate_npy_artifact(
    path: str | Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    expected_sha256: str | None = None,
    finite: bool = True,
) -> np.ndarray:
    """Validate a saved NumPy artifact without loading it fully into memory."""
    source = Path(path)
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape:
        raise ValueError(f"Unexpected shape for {source}: {values.shape} != {expected_shape}")
    if str(values.dtype) != expected_dtype:
        raise ValueError(f"Unexpected dtype for {source}: {values.dtype} != {expected_dtype}")
    if finite and not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {source}")
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {source}")
    return values


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def save_npy(path: str | Path, values: np.ndarray) -> Path:
    """Atomically save a NumPy array without pickle support."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_ordered_ids(path: str | Path, *, id_column: str = "sample_id") -> np.ndarray:
    frame = pd.read_csv(path, dtype={id_column: str})
    if list(frame.columns) != [id_column]:
        raise ValueError(f"Expected one {id_column} column in {path}")
    values = frame[id_column]
    if values.isna().any() or values.duplicated().any():
        raise ValueError(f"{id_column} values must be non-null and unique")
    return values.to_numpy()


def write_ordered_ids(
    path: str | Path,
    sample_ids: Iterable[object],
    *,
    id_column: str = "sample_id",
) -> Path:
    values = pd.Series(list(sample_ids), name=id_column)
    if values.isna().any() or values.duplicated().any():
        raise ValueError(f"{id_column} values must be non-null and unique")
    return write_csv(path, values.to_frame())


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    """Atomically write a CSV data frame."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_frame(
    frame: pd.DataFrame,
    *,
    training: bool,
    schema: ChallengeSchema = DEFAULT_SCHEMA,
) -> None:
    missing = sorted(set(schema.required_columns(training=training)) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[schema.id_column].isna().any():
        raise ValueError(f"{schema.id_column} contains null values")
    duplicates = frame[schema.id_column].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, schema.id_column].head(5).tolist()
        raise ValueError(f"{schema.id_column} must be unique; duplicate examples: {examples}")
    if training:
        target = pd.to_numeric(frame[schema.target_column], errors="coerce")
        if target.isna().any() or not np.isfinite(target.to_numpy()).all() or (target <= 0).any():
            raise ValueError("Training price must be finite numeric values greater than zero")


def load_challenge_csv(
    path: str | Path,
    *,
    training: bool,
    schema: ChallengeSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_frame(frame, training=training, schema=schema)
    return frame


def align_by_sample_id(
    reference_ids: Iterable[object],
    frame: pd.DataFrame,
    *,
    schema: ChallengeSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    reference = pd.Index(list(reference_ids), name=schema.id_column)
    if reference.has_duplicates:
        raise ValueError("Reference sample IDs must be unique")
    if schema.id_column not in frame:
        raise ValueError(f"Derived frame is missing {schema.id_column}")
    indexed = frame.set_index(schema.id_column)
    if not indexed.index.is_unique:
        raise ValueError("Derived sample IDs must be unique")
    missing = reference.difference(indexed.index)
    extra = indexed.index.difference(reference)
    if len(missing) or len(extra):
        raise ValueError(
            f"sample_id set mismatch: {len(missing)} missing and {len(extra)} unexpected"
        )
    return indexed.loc[reference].reset_index()


def build_submission(sample_ids: Iterable[object], predictions: Iterable[float]) -> pd.DataFrame:
    ids = pd.Series(list(sample_ids), name="sample_id")
    values = np.asarray(list(predictions), dtype=np.float64)
    if len(ids) != len(values):
        raise ValueError(f"ID/prediction count mismatch: {len(ids)} != {len(values)}")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("sample_id values must be non-null and unique")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("All price predictions must be finite and greater than zero")
    return pd.DataFrame({"sample_id": ids, "price": values})


def write_submission(
    sample_ids: Iterable[object], predictions: Iterable[float], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_submission(sample_ids, predictions)
    write_csv(output, frame)
    round_trip = pd.read_csv(output, dtype={"sample_id": str})
    if (
        list(round_trip.columns) != ["sample_id", "price"]
        or len(round_trip) != len(frame)
        or list(map(str, round_trip["sample_id"])) != list(map(str, frame["sample_id"]))
        or not np.array_equal(
            round_trip["price"].to_numpy(dtype=np.float64),
            frame["price"].to_numpy(dtype=np.float64),
        )
    ):
        raise RuntimeError("Written submission failed round-trip validation")
    return output
