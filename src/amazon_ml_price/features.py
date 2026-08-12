"""Deterministic multimodal feature extraction and assembly."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    canonical_json_hash,
    file_sha256,
    ordered_id_hash,
    read_ordered_ids,
    save_npy,
    write_json,
    write_ordered_ids,
)

SIGLIP2_DIM = 1024
MINILM_DIM = 384
CATALOGUE_DIM = 17
FINAL_FEATURE_DIM = 2450
DEFAULT_SIGLIP2_CHECKPOINT = "google/siglip2-large-patch16-384"
DEFAULT_SIGLIP2_REVISION = "1b426889ea62b5a72bf9839009a1b184bfc9c178"
DEFAULT_MINILM_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024

FEATURE_BLOCKS = (
    ("siglip2_image", 0, 1024),
    ("siglip2_text", 1024, 2048),
    ("siglip2_cosine", 2048, 2049),
    ("minilm_text", 2049, 2433),
    ("catalogue", 2433, 2450),
)
CATALOGUE_FEATURE_NAMES = (
    "pack_quantity",
    "base_unit_value",
    "total_declared_value",
    "character_count",
    "word_count",
    "numbered_bullet_count",
    "line_count",
    "digit_count",
    "unique_word_count",
    "item_name_word_count",
    "description_word_count",
    "measurement_mention_count",
    "has_brand",
    "has_pack_expression",
    "unit_frequency",
    "brand_frequency",
    "product_class_frequency",
)

_NUMBER = r"(?P<value>\d+(?:\.\d+)?)"
_UNIT_RE = re.compile(
    rf"{_NUMBER}\s*(?P<unit>kilograms?|kg|milligrams?|mg|grams?|g|"
    r"millilit(?:er|re)s?|ml|lit(?:er|re)s?|l|fl\.?\s*oz|oz|ounces?|"
    r"lb|pounds?|cm|mm|met(?:er|re)s?|m|counts?|ct|pieces?|pcs?)\b",
    re.IGNORECASE,
)
_PACK_PATTERNS = (
    re.compile(r"\b(?:pack|set|case)\s+of\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*[- ]?(?:pack|count|ct|pcs?|pieces?)\b", re.IGNORECASE),
    re.compile(r"\b(?:pack|qty|quantity)\s*[:x-]?\s*(\d+)\b", re.IGNORECASE),
)
_NUMBERED_BULLET_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+|\bbullet\s*point\s*\d+\s*:",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?P<label>item\s*name|product\s*name|title|brand|"
    r"product\s*class|category|description)\s*[:\-]\s*(?P<value>[^\n|]{0,500})",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_UNIT_ALIASES = {
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
    "milligram": "mg", "milligrams": "mg", "liter": "l", "liters": "l",
    "litre": "l", "litres": "l", "milliliter": "ml", "milliliters": "ml",
    "millilitre": "ml", "millilitres": "ml", "ounce": "oz", "ounces": "oz",
    "fl oz": "fl_oz", "fl. oz": "fl_oz", "pound": "lb", "pounds": "lb",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "count": "count", "counts": "count", "ct": "count", "piece": "count",
    "pieces": "count", "pc": "count", "pcs": "count",
}
_UNIT_CONVERSIONS = {
    "kg": (1000.0, "g"), "g": (1.0, "g"), "mg": (0.001, "g"),
    "lb": (453.59237, "g"), "oz": (28.349523125, "g"),
    "l": (1000.0, "ml"), "ml": (1.0, "ml"), "fl_oz": (29.5735295625, "ml"),
    "m": (1000.0, "mm"), "cm": (10.0, "mm"), "mm": (1.0, "mm"),
    "count": (1.0, "count"),
}


def _clean_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _labelled_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _LABEL_RE.finditer(text):
        key = re.sub(r"\s+", "_", match.group("label").lower())
        fields.setdefault(key, match.group("value").strip())
    return fields


def _normalise_category(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.strip().lower())
    return clean or "unknown"


def extract_pack_quantity(text: object) -> int:
    clean = _clean_text(text)
    values = [int(match.group(1)) for pattern in _PACK_PATTERNS if (match := pattern.search(clean))]
    return max(values, default=1)


def extract_measurements(text: object) -> list[tuple[float, str]]:
    measurements: list[tuple[float, str]] = []
    for match in _UNIT_RE.finditer(_clean_text(text)):
        unit = re.sub(r"\s+", " ", match.group("unit").lower()).strip()
        unit = _UNIT_ALIASES.get(unit, unit)
        multiplier, canonical_unit = _UNIT_CONVERSIONS.get(unit, (1.0, unit))
        measurements.append((float(match.group("value")) * multiplier, canonical_unit))
    return measurements


def extract_measurement(text: object) -> tuple[float, str]:
    measurements = extract_measurements(text)
    return measurements[0] if measurements else (0.0, "unknown")


def extract_brand(text: object) -> str:
    clean = _clean_text(text)
    fields = _labelled_fields(clean)
    if fields.get("brand"):
        return _normalise_category(fields["brand"])
    item_name = fields.get("item_name") or fields.get("product_name") or fields.get("title")
    first_line = item_name or (clean.splitlines()[0] if clean else "")
    first_token = re.sub(r"[^a-zA-Z0-9-]", "", first_line.split()[0] if first_line else "")
    return _normalise_category(first_token)


def extract_product_class(text: object) -> str:
    fields = _labelled_fields(_clean_text(text))
    value = fields.get("product_class") or fields.get("category") or "unknown"
    return _normalise_category(value)


def _select_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _normalise_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def cosine_similarity_column(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    expected_dim: int = SIGLIP2_DIM,
) -> np.ndarray:
    image = np.asarray(image_embeddings, dtype=np.float32)
    text = np.asarray(text_embeddings, dtype=np.float32)
    if image.ndim != 2 or image.shape[1] != expected_dim:
        raise ValueError(f"Expected image embeddings shaped (n, {expected_dim}); got {image.shape}")
    if text.shape != image.shape:
        raise ValueError(f"Image/text embedding shape mismatch: {image.shape} != {text.shape}")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Embeddings must contain only finite values")
    similarity = np.sum(
        _normalise_rows(image) * _normalise_rows(text), axis=1, keepdims=True
    )
    return similarity.astype(np.float32)


def _download_bytes(
    source: str,
    *,
    timeout: float,
    retries: int,
    max_bytes: int,
) -> bytes:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        response = session.get(source, timeout=timeout, stream=True)
        response.raise_for_status()
        declared_size = int(response.headers.get("content-length", 0))
        if declared_size > max_bytes:
            raise OSError(f"Image exceeds {max_bytes} bytes: {source}")
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise OSError(f"Image exceeds {max_bytes} bytes: {source}")
            chunks.append(chunk)
        content = b"".join(chunks)
    if len(content) > max_bytes:
        raise OSError(f"Image exceeds {max_bytes} bytes: {source}")
    return content


def load_rgb_image(
    source: str,
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 15.0,
    retries: int = 3,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
):
    from PIL import Image

    source = str(source)
    if source.startswith(("http://", "https://")):
        cached: Path | None = None
        if cache_dir is not None:
            cached = Path(cache_dir) / f"{hashlib.sha256(source.encode()).hexdigest()}.image"
            cached.parent.mkdir(parents=True, exist_ok=True)
        if cached is not None and cached.exists():
            content = cached.read_bytes()
        else:
            try:
                content = _download_bytes(
                    source, timeout=timeout, retries=retries, max_bytes=max_bytes
                )
            except Exception as exc:
                raise OSError(f"Unable to download image: {source}") from exc
            if cached is not None:
                cached.write_bytes(content)
        if len(content) > max_bytes:
            raise OSError(f"Image exceeds {max_bytes} bytes: {source}")
        return Image.open(BytesIO(content)).convert("RGB")
    return Image.open(Path(source)).convert("RGB")


@dataclass
class Siglip2Encoder:
    checkpoint: str = DEFAULT_SIGLIP2_CHECKPOINT
    revision: str = DEFAULT_SIGLIP2_REVISION
    expected_dim: int = SIGLIP2_DIM
    max_text_length: int = 64
    batch_size: int = 16
    device: str | None = None
    image_timeout: float = 15.0
    image_retries: int = 3
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES

    def __post_init__(self) -> None:
        if self.expected_dim != SIGLIP2_DIM:
            raise ValueError(f"SigLIP2 projection width must be {SIGLIP2_DIM}")
        if self.batch_size <= 0 or self.max_text_length <= 0:
            raise ValueError("batch_size and max_text_length must be positive")

    def _load(self) -> None:
        if hasattr(self, "model"):
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = self.device or _select_device(torch)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        self.model = AutoModel.from_pretrained(self.checkpoint, revision=self.revision)
        projection_dim = getattr(getattr(self.model, "config", None), "projection_size", None)
        if projection_dim is not None and int(projection_dim) != self.expected_dim:
            raise ValueError(
                f"Checkpoint projection width is {projection_dim}; expected {self.expected_dim}"
            )
        self.model = self.model.to(self.device).eval()

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "projection_dim": self.expected_dim,
            "max_text_length": self.max_text_length,
            "batch_size": self.batch_size,
            "device": self.device or "auto",
        }

    def encode_texts(self, texts: Sequence[object]) -> np.ndarray:
        self._load()
        import torch

        text_values = list(texts)
        output = np.empty((len(text_values), self.expected_dim), dtype=np.float32)
        for start in range(0, len(text_values), self.batch_size):
            stop = min(start + self.batch_size, len(text_values))
            inputs = self.processor(
                text=[_clean_text(value) for value in text_values[start:stop]],
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                encoded = self.model.get_text_features(**inputs)
            batch = encoded.detach().float().cpu().numpy()
            if batch.shape != (stop - start, self.expected_dim):
                raise ValueError(f"SigLIP2 text output has invalid shape: {batch.shape}")
            output[start:stop] = batch
        return output

    def encode_images(
        self,
        image_sources: Sequence[object],
        *,
        cache_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._load()
        import torch

        sources = list(image_sources)
        output = np.zeros((len(sources), self.expected_dim), dtype=np.float32)
        failed = np.zeros(len(sources), dtype=np.uint8)
        for start in range(0, len(sources), self.batch_size):
            stop = min(start + self.batch_size, len(sources))
            valid_rows: list[int] = []
            images = []
            for row in range(start, stop):
                try:
                    image = load_rgb_image(
                        _clean_text(sources[row]),
                        cache_dir=cache_dir,
                        timeout=self.image_timeout,
                        retries=self.image_retries,
                        max_bytes=self.max_image_bytes,
                    )
                except (OSError, ValueError):
                    failed[row] = 1
                    continue
                valid_rows.append(row)
                images.append(image)
            if not images:
                continue
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                encoded = self.model.get_image_features(**inputs)
            batch = encoded.detach().float().cpu().numpy()
            if batch.shape != (len(valid_rows), self.expected_dim):
                raise ValueError(f"SigLIP2 image output has invalid shape: {batch.shape}")
            output[np.asarray(valid_rows)] = batch
            for image in images:
                image.close()
        return output, failed

    def encode(
        self,
        image_sources: Sequence[object],
        texts: Sequence[object],
        *,
        cache_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(image_sources) != len(texts):
            raise ValueError("image_sources and texts must have equal length")
        image, failed = self.encode_images(image_sources, cache_dir=cache_dir)
        text = self.encode_texts(texts)
        cosine = cosine_similarity_column(image, text, expected_dim=self.expected_dim)
        return image, text, cosine, failed


SiglipEncoder = Siglip2Encoder


@dataclass
class MiniLMEncoder:
    checkpoint: str = DEFAULT_MINILM_CHECKPOINT
    revision: str = DEFAULT_MINILM_REVISION
    expected_dim: int = MINILM_DIM
    max_sequence_length: int = 64
    batch_size: int = 256
    device: str | None = None

    def __post_init__(self) -> None:
        if self.expected_dim != MINILM_DIM:
            raise ValueError(f"MiniLM embedding width must be {MINILM_DIM}")
        if self.batch_size <= 0 or self.max_sequence_length <= 0:
            raise ValueError("batch_size and max_sequence_length must be positive")

    def _load(self) -> None:
        if hasattr(self, "model"):
            return
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = self.device or _select_device(torch)
        self.model = SentenceTransformer(
            self.checkpoint,
            revision=self.revision,
            device=self.device,
        )
        self.model.max_seq_length = self.max_sequence_length
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension != self.expected_dim:
            raise ValueError(f"MiniLM output width is {dimension}; expected {self.expected_dim}")

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "embedding_dim": self.expected_dim,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "device": self.device or "auto",
            "normalize_embeddings": False,
        }

    def encode(self, texts: Sequence[object]) -> np.ndarray:
        self._load()
        output = self.model.encode(
            [_clean_text(value) for value in texts],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        output = np.asarray(output, dtype=np.float32)
        if output.shape != (len(texts), self.expected_dim):
            raise ValueError(f"MiniLM output has invalid shape: {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("MiniLM embeddings must contain only finite values")
        return output


@dataclass
class CatalogFeatureExtractor:
    """Fit train-only frequency maps and produce the fixed 17-feature block."""

    unit_counts: Counter[str] = field(default_factory=Counter, init=False)
    brand_counts: Counter[str] = field(default_factory=Counter, init=False)
    product_class_counts: Counter[str] = field(default_factory=Counter, init=False)
    training_rows: int = field(default=0, init=False)

    @staticmethod
    def _records(values: Sequence[object] | pd.Series) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for value in values:
            text = _clean_text(value)
            fields = _labelled_fields(text)
            words = [word.lower() for word in _WORD_RE.findall(text)]
            measurements = extract_measurements(text)
            base_value, unit = measurements[0] if measurements else (0.0, "unknown")
            pack_quantity = extract_pack_quantity(text)
            item_name = (
                fields.get("item_name")
                or fields.get("product_name")
                or fields.get("title")
                or ""
            )
            description = fields.get("description", "")
            brand = extract_brand(text)
            product_class = extract_product_class(text)
            records.append({
                "text": text,
                "words": words,
                "measurements": measurements,
                "base_unit_value": base_value,
                "unit": unit,
                "pack_quantity": pack_quantity,
                "item_name": item_name,
                "description": description,
                "brand": brand,
                "product_class": product_class,
                "has_brand": float(fields.get("brand", "").strip() != ""),
                "has_pack_expression": float(
                    any(pattern.search(text) for pattern in _PACK_PATTERNS)
                ),
            })
        return records

    def fit(self, values: Sequence[object] | pd.Series) -> CatalogFeatureExtractor:
        records = self._records(values)
        self.training_rows = len(records)
        if not self.training_rows:
            raise ValueError("Cannot fit catalogue features on an empty training set")
        self.unit_counts = Counter(record["unit"] for record in records)
        self.brand_counts = Counter(record["brand"] for record in records)
        self.product_class_counts = Counter(record["product_class"] for record in records)
        return self

    def transform(self, values: Sequence[object] | pd.Series) -> pd.DataFrame:
        if self.training_rows <= 0:
            raise RuntimeError("CatalogFeatureExtractor must be fitted before transform")
        rows: list[list[float]] = []
        for record in self._records(values):
            text = record["text"]
            words = record["words"]
            pack_quantity = float(record["pack_quantity"])
            base_value = float(record["base_unit_value"])
            rows.append([
                pack_quantity,
                base_value,
                base_value * pack_quantity,
                float(len(text)),
                float(len(words)),
                float(len(_NUMBERED_BULLET_RE.findall(text))),
                float(len(text.splitlines())) if text else 0.0,
                float(sum(character.isdigit() for character in text)),
                float(len(set(words))),
                float(len(_WORD_RE.findall(record["item_name"]))),
                float(len(_WORD_RE.findall(record["description"]))),
                float(len(record["measurements"])),
                record["has_brand"],
                record["has_pack_expression"],
                float(self.unit_counts.get(record["unit"], 0)) / self.training_rows,
                float(self.brand_counts.get(record["brand"], 0)) / self.training_rows,
                float(self.product_class_counts.get(record["product_class"], 0))
                / self.training_rows,
            ])
        frame = pd.DataFrame(rows, columns=CATALOGUE_FEATURE_NAMES, dtype=np.float32)
        if frame.shape[1] != CATALOGUE_DIM or not np.isfinite(frame.to_numpy()).all():
            raise RuntimeError("Catalogue feature contract was violated")
        return frame

    def fit_transform(self, values: Sequence[object] | pd.Series) -> pd.DataFrame:
        return self.fit(values).transform(values)

    def frequency_manifest(self) -> dict[str, Any]:
        if self.training_rows <= 0:
            raise RuntimeError("CatalogFeatureExtractor has not been fitted")
        return {
            "training_rows": self.training_rows,
            "unit_counts": dict(sorted(self.unit_counts.items())),
            "brand_counts": dict(sorted(self.brand_counts.items())),
            "product_class_counts": dict(sorted(self.product_class_counts.items())),
        }


def fuse_siglip_features(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    expected_dim: int = SIGLIP2_DIM,
) -> np.ndarray:
    cosine = cosine_similarity_column(image_embeddings, text_embeddings, expected_dim=expected_dim)
    return np.concatenate(
        (
            np.asarray(image_embeddings, dtype=np.float32),
            np.asarray(text_embeddings, dtype=np.float32),
            cosine,
        ),
        axis=1,
    ).astype(np.float32)


def _sidecar_paths(matrix_path: str | Path) -> tuple[Path, Path, Path]:
    matrix = Path(matrix_path)
    return (
        matrix,
        matrix.with_suffix(".ids.csv"),
        matrix.with_suffix(".manifest.json"),
    )


def write_feature_block(
    matrix_path: str | Path,
    matrix: np.ndarray,
    sample_ids: Sequence[object],
    *,
    block_name: str,
    config_hash: str,
    source_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    expected_widths = {name: stop - start for name, start, stop in FEATURE_BLOCKS}
    if block_name not in expected_widths:
        raise ValueError(f"Unknown feature block: {block_name}")
    if values.ndim != 2 or values.shape[0] != len(sample_ids):
        raise ValueError("Feature rows must match sample IDs")
    if values.shape[1] != expected_widths[block_name]:
        raise ValueError(
            f"{block_name} must have {expected_widths[block_name]} columns; "
            f"found {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Feature blocks must contain only finite values")
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise ValueError("Feature sample IDs must be unique")
    output, ids_path, manifest_path = _sidecar_paths(matrix_path)
    if output.suffix != ".npy":
        raise ValueError("Feature matrix paths must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npy(output, values)
    write_ordered_ids(ids_path, sample_ids)
    manifest = {
        "schema_version": 1,
        "block": block_name,
        "matrix_file": output.name,
        "ids_file": ids_path.name,
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "dtype": "float32",
        "config_hash": config_hash,
        "source_hash": source_hash,
        "ordered_id_hash": ordered_id_hash(sample_ids),
        "matrix_sha256": file_sha256(output),
        "ids_sha256": file_sha256(ids_path),
        "metadata": dict(metadata or {}),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_feature_block(
    matrix_path: str | Path,
    expected_ids: Sequence[object],
    *,
    block_name: str,
    expected_width: int,
    config_hash: str,
    source_hash: str,
) -> np.ndarray:
    matrix_path, ids_path, manifest_path = _sidecar_paths(matrix_path)
    if not matrix_path.exists() or not ids_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Feature block or companion files are missing: {matrix_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_ids = read_ordered_ids(ids_path)
    expected_hash = ordered_id_hash(expected_ids)
    if manifest.get("block") != block_name:
        raise ValueError(f"Expected {block_name} block; found {manifest.get('block')}")
    if manifest.get("config_hash") != config_hash or manifest.get("source_hash") != source_hash:
        raise ValueError(f"Manifest hash mismatch for {block_name}")
    if (
        manifest.get("matrix_sha256") != file_sha256(matrix_path)
        or manifest.get("ids_sha256") != file_sha256(ids_path)
    ):
        raise ValueError(f"Artifact checksum mismatch for {block_name}")
    if (
        manifest.get("rows") != len(expected_ids)
        or manifest.get("columns") != expected_width
        or manifest.get("dtype") != "float32"
    ):
        raise ValueError(f"Manifest shape or dtype mismatch for {block_name}")
    if (
        manifest.get("ordered_id_hash") != expected_hash
        or ordered_id_hash(saved_ids) != expected_hash
    ):
        raise ValueError(f"Ordered sample IDs do not match for {block_name}")
    if list(map(str, saved_ids)) != list(map(str, expected_ids)):
        raise ValueError(f"Ordered sample IDs do not match for {block_name}")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (len(expected_ids), expected_width):
        raise ValueError(
            f"{block_name} must have shape {(len(expected_ids), expected_width)} and float32 dtype"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{block_name} contains non-finite values")
    return matrix


def assemble_feature_matrix(
    *,
    siglip2_image: np.ndarray,
    siglip2_text: np.ndarray,
    siglip2_cosine: np.ndarray,
    minilm_text: np.ndarray,
    catalogue: np.ndarray,
) -> np.ndarray:
    blocks = (
        np.asarray(siglip2_image, dtype=np.float32),
        np.asarray(siglip2_text, dtype=np.float32),
        np.asarray(siglip2_cosine, dtype=np.float32),
        np.asarray(minilm_text, dtype=np.float32),
        np.asarray(catalogue, dtype=np.float32),
    )
    widths = (SIGLIP2_DIM, SIGLIP2_DIM, 1, MINILM_DIM, CATALOGUE_DIM)
    if any(block.ndim != 2 for block in blocks):
        raise ValueError("Every feature block must be two-dimensional")
    row_counts = {block.shape[0] for block in blocks}
    if len(row_counts) != 1:
        raise ValueError("Feature block row counts do not match")
    for (name, _, _), block, width in zip(FEATURE_BLOCKS, blocks, widths, strict=True):
        if block.shape[1] != width:
            raise ValueError(f"{name} must have {width} columns; found {block.shape[1]}")
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains non-finite values")
    output = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if output.shape[1] != FINAL_FEATURE_DIM:
        raise RuntimeError(f"Expected {FINAL_FEATURE_DIM} final features; found {output.shape[1]}")
    return output


def assemble_feature_artifacts(
    block_paths: Mapping[str, str | Path],
    sample_ids: Sequence[object],
    *,
    config_hash: str,
    source_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    widths = dict(
        zip(
            (name for name, _, _ in FEATURE_BLOCKS),
            (1024, 1024, 1, 384, 17),
            strict=True,
        )
    )
    missing = sorted(set(widths) - set(block_paths))
    if missing:
        raise ValueError(f"Missing feature blocks: {missing}")
    loaded = {
        name: load_feature_block(
            block_paths[name],
            sample_ids,
            block_name=name,
            expected_width=width,
            config_hash=config_hash,
            source_hash=source_hash,
        )
        for name, width in widths.items()
    }
    matrix = assemble_feature_matrix(**loaded)
    manifest = {
        "schema_version": 1,
        "rows": int(matrix.shape[0]),
        "columns": FINAL_FEATURE_DIM,
        "dtype": "float32",
        "config_hash": config_hash,
        "source_hash": source_hash,
        "ordered_id_hash": ordered_id_hash(sample_ids),
        "feature_blocks": [
            {"name": name, "start": start, "stop": stop, "width": stop - start}
            for name, start, stop in FEATURE_BLOCKS
        ],
        "manifest_hash": canonical_json_hash({
            "config_hash": config_hash,
            "source_hash": source_hash,
            "ordered_id_hash": ordered_id_hash(sample_ids),
            "columns": FINAL_FEATURE_DIM,
        }),
    }
    return matrix, manifest
