"""Training and ensemble routines for the final four-model price pipeline.

Heavy model libraries are imported only by the trainer that needs them.  This
keeps data validation and manifest inspection available from a core install.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .data import file_sha256, save_npy, write_csv, write_json

MODEL_ORDER = ("lightgbm", "xgboost", "catboost", "mlp")
DEFAULT_PREDICTION_FLOOR = 0.01


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, percentage: bool = True) -> float:
    """Return symmetric mean absolute percentage error."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {truth.shape} != {pred.shape}")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("SMAPE inputs must contain only finite values")
    denominator = np.abs(truth) + np.abs(pred)
    terms = np.divide(
        2.0 * np.abs(pred - truth),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    value = float(np.mean(terms))
    return value * 100.0 if percentage else value


def _feature_matrix(values):
    """Validate dense features without accidentally densifying SciPy matrices."""
    if sparse.issparse(values):
        matrix = values
        finite = np.isfinite(matrix.data).all()
    else:
        matrix = np.asarray(values)
        finite = np.isfinite(matrix).all()
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must contain at least one row and one column")
    if not finite:
        raise ValueError("features must contain only finite values")
    return matrix


def _target_vector(values, *, expected_rows: int) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(target) != expected_rows:
        raise ValueError("features and target must contain the same number of rows")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("target must contain finite values greater than zero")
    return target


def _test_matrix(values, *, expected_columns: int):
    if values is None:
        return None
    test = _feature_matrix(values)
    if test.shape[1] != expected_columns:
        raise ValueError("train and test features must have the same number of columns")
    return test


def _positive_predictions(log_predictions, prediction_floor: float) -> np.ndarray:
    if prediction_floor <= 0:
        raise ValueError("prediction_floor must be greater than zero")
    predictions = np.expm1(np.asarray(log_predictions, dtype=np.float64).reshape(-1))
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return np.maximum(predictions, prediction_floor).astype(np.float32)


def _hash_folds(sample_ids: np.ndarray, folds: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id, fold in zip(sample_ids, folds, strict=True):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(fold)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class FoldAssignments:
    """Stable sample-to-fold mapping shared by every base learner."""

    sample_ids: np.ndarray
    folds: np.ndarray
    n_splits: int
    random_state: int
    manifest_hash: str

    def split(self):
        for fold in range(self.n_splits):
            valid = np.flatnonzero(self.folds == fold)
            train = np.flatnonzero(self.folds != fold)
            if valid.size == 0 or train.size == 0:
                raise ValueError(f"Fold {fold} does not contain a valid train/validation split")
            yield train, valid


def create_shared_folds(
    sample_ids: Sequence[object], *, n_splits: int = 5, random_state: int = 42
) -> FoldAssignments:
    """Create deterministic shuffled K-fold assignments bound to ordered IDs."""
    ids = np.asarray(sample_ids).reshape(-1)
    if ids.size == 0:
        raise ValueError("sample_ids must not be empty")
    normalized = np.asarray([str(value) for value in ids], dtype=str)
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError("sample_ids must be unique")
    if not 2 <= n_splits <= len(ids):
        raise ValueError("n_splits must be between 2 and the number of samples")
    folds = np.full(len(ids), -1, dtype=np.int16)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (_, valid_index) in enumerate(splitter.split(ids)):
        folds[valid_index] = fold
    if (folds < 0).any():
        raise RuntimeError("Every sample must receive exactly one fold assignment")
    return FoldAssignments(ids, folds, n_splits, random_state, _hash_folds(ids, folds))


def validate_shared_folds(assignments: FoldAssignments, sample_ids: Sequence[object]) -> None:
    """Reject a fold file that is incomplete or bound to a different row order."""
    ids = np.asarray(sample_ids).reshape(-1)
    if not np.array_equal(np.asarray(assignments.sample_ids).astype(str), ids.astype(str)):
        raise ValueError("Fold sample IDs do not match the feature row order")
    if len(assignments.folds) != len(ids):
        raise ValueError("Fold assignments and features have different row counts")
    expected = set(range(assignments.n_splits))
    if set(np.unique(assignments.folds).tolist()) != expected:
        raise ValueError("Fold assignments must cover every configured fold")
    if assignments.manifest_hash != _hash_folds(ids, assignments.folds):
        raise ValueError("Fold manifest hash does not match its contents")


@dataclass
class ModelPredictions:
    """Common output contract for every final base learner."""

    model_name: str
    oof_predictions: np.ndarray
    full_fit_test_predictions: np.ndarray | None
    fold_smape: list[float]
    best_iterations_or_epochs: list[int]
    parameters: dict[str, object]
    feature_manifest_hash: str
    fold_manifest_hash: str
    final_model: object | None = field(default=None, repr=False)

    def validate(self, *, expected_rows: int, expected_test_rows: int | None = None) -> None:
        if self.model_name not in MODEL_ORDER:
            raise ValueError(f"Unknown final model: {self.model_name}")
        oof = np.asarray(self.oof_predictions).reshape(-1)
        if len(oof) != expected_rows or not np.isfinite(oof).all() or (oof <= 0).any():
            raise ValueError(f"{self.model_name} OOF predictions are incomplete or invalid")
        if len(self.fold_smape) != len(self.best_iterations_or_epochs):
            raise ValueError("Fold scores and selected iteration counts must align")
        if expected_test_rows is not None:
            if self.full_fit_test_predictions is None:
                raise ValueError(f"{self.model_name} is missing full-fit test predictions")
            test = np.asarray(self.full_fit_test_predictions).reshape(-1)
            if len(test) != expected_test_rows or not np.isfinite(test).all() or (test <= 0).any():
                raise ValueError(f"{self.model_name} test predictions are invalid")


def _training_inputs(features, target, test_features, folds):
    x = _feature_matrix(features)
    y = _target_vector(target, expected_rows=x.shape[0])
    test = _test_matrix(test_features, expected_columns=x.shape[1])
    if folds is None:
        folds = create_shared_folds(np.arange(len(y)))
    validate_shared_folds(folds, folds.sample_ids)
    if len(folds.sample_ids) != len(y):
        raise ValueError("Fold assignments and training features have different row counts")
    return x, y, test, folds


def _median_selection(values: Sequence[int], *, minimum: int = 1) -> int:
    if not values:
        raise ValueError("At least one selected iteration is required")
    return max(minimum, int(np.median(np.asarray(values, dtype=np.float64))))


def train_lightgbm_oof(
    features,
    target,
    *,
    test_features=None,
    folds: FoldAssignments | None = None,
    feature_manifest_hash: str = "",
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    random_state: int = 42,
    **overrides,
) -> ModelPredictions:
    """Train LightGBM on shared folds and refit once on all labelled rows."""
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("LightGBM is unavailable; install the models extra.") from exc
    x, y, test, folds = _training_inputs(features, target, test_features, folds)
    parameters: dict[str, Any] = {
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    parameters.update(overrides)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds", 100))
    oof = np.zeros(len(y), dtype=np.float32)
    scores: list[float] = []
    selections: list[int] = []
    for train_index, valid_index in folds.split():
        model = lgb.LGBMRegressor(**parameters)
        model.fit(
            x[train_index],
            np.log1p(y[train_index]),
            eval_set=[(x[valid_index], np.log1p(y[valid_index]))],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        predicted = _positive_predictions(model.predict(x[valid_index]), prediction_floor)
        oof[valid_index] = predicted
        scores.append(smape(y[valid_index], predicted))
        selections.append(int(getattr(model, "best_iteration_", parameters["n_estimators"])))
    final_parameters = dict(parameters)
    final_parameters["n_estimators"] = _median_selection(selections)
    final_model = lgb.LGBMRegressor(**final_parameters)
    final_model.fit(x, np.log1p(y))
    test_predictions = (
        None
        if test is None
        else _positive_predictions(final_model.predict(test), prediction_floor)
    )
    result = ModelPredictions(
        "lightgbm", oof, test_predictions, scores, selections,
        {**final_parameters, "early_stopping_rounds": early_stopping_rounds},
        feature_manifest_hash, folds.manifest_hash, final_model,
    )
    result.validate(
        expected_rows=len(y), expected_test_rows=None if test is None else test.shape[0]
    )
    return result


def train_xgboost_oof(
    features,
    target,
    *,
    test_features=None,
    folds: FoldAssignments | None = None,
    feature_manifest_hash: str = "",
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    random_state: int = 42,
    **overrides,
) -> ModelPredictions:
    """Train XGBoost on shared folds and refit once on all labelled rows."""
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost is unavailable; install the models extra.") from exc
    x, y, test, folds = _training_inputs(features, target, test_features, folds)
    parameters: dict[str, Any] = {
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "tree_method": "hist",
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
    }
    parameters.update(overrides)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds", 100))
    oof = np.zeros(len(y), dtype=np.float32)
    scores: list[float] = []
    selections: list[int] = []
    for train_index, valid_index in folds.split():
        model = XGBRegressor(**parameters, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            x[train_index], np.log1p(y[train_index]),
            eval_set=[(x[valid_index], np.log1p(y[valid_index]))], verbose=False,
        )
        predicted = _positive_predictions(model.predict(x[valid_index]), prediction_floor)
        oof[valid_index] = predicted
        scores.append(smape(y[valid_index], predicted))
        best = getattr(model, "best_iteration", None)
        selections.append(int(best) + 1 if best is not None else int(parameters["n_estimators"]))
    final_parameters = dict(parameters)
    final_parameters["n_estimators"] = _median_selection(selections)
    final_model = XGBRegressor(**final_parameters)
    final_model.fit(x, np.log1p(y), verbose=False)
    test_predictions = (
        None
        if test is None
        else _positive_predictions(final_model.predict(test), prediction_floor)
    )
    result = ModelPredictions(
        "xgboost", oof, test_predictions, scores, selections,
        {**final_parameters, "early_stopping_rounds": early_stopping_rounds},
        feature_manifest_hash, folds.manifest_hash, final_model,
    )
    result.validate(
        expected_rows=len(y), expected_test_rows=None if test is None else test.shape[0]
    )
    return result


def train_catboost_oof(
    features,
    target,
    *,
    test_features=None,
    folds: FoldAssignments | None = None,
    feature_manifest_hash: str = "",
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    random_state: int = 42,
    **overrides,
) -> ModelPredictions:
    """Train CatBoost on shared folds and refit once on all labelled rows."""
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:
        raise RuntimeError("CatBoost is unavailable; install the models extra.") from exc
    x, y, test, folds = _training_inputs(features, target, test_features, folds)
    parameters: dict[str, Any] = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 8,
        "loss_function": "RMSE",
        "random_seed": random_state,
        "allow_writing_files": False,
        "verbose": False,
    }
    parameters.update(overrides)
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds", 100))
    oof = np.zeros(len(y), dtype=np.float32)
    scores: list[float] = []
    selections: list[int] = []
    for train_index, valid_index in folds.split():
        model = CatBoostRegressor(**parameters)
        model.fit(
            x[train_index], np.log1p(y[train_index]),
            eval_set=(x[valid_index], np.log1p(y[valid_index])),
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        predicted = _positive_predictions(model.predict(x[valid_index]), prediction_floor)
        oof[valid_index] = predicted
        scores.append(smape(y[valid_index], predicted))
        best = model.get_best_iteration()
        selections.append(
            int(best) + 1
            if best is not None and best >= 0
            else int(parameters["iterations"])
        )
    final_parameters = dict(parameters)
    final_parameters["iterations"] = _median_selection(selections)
    final_model = CatBoostRegressor(**final_parameters)
    final_model.fit(x, np.log1p(y), verbose=False)
    test_predictions = (
        None
        if test is None
        else _positive_predictions(final_model.predict(test), prediction_floor)
    )
    result = ModelPredictions(
        "catboost", oof, test_predictions, scores, selections,
        {**final_parameters, "early_stopping_rounds": early_stopping_rounds},
        feature_manifest_hash, folds.manifest_hash, final_model,
    )
    result.validate(
        expected_rows=len(y), expected_test_rows=None if test is None else test.shape[0]
    )
    return result


def _resolve_torch_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mlp_oof(
    features,
    target,
    *,
    test_features=None,
    folds: FoldAssignments | None = None,
    feature_manifest_hash: str = "",
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    random_state: int = 42,
    hidden_sizes: Sequence[int] = (512, 128),
    dropout: Sequence[float] = (0.2, 0.1),
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 50,
    patience: int = 6,
    device: str = "auto",
) -> ModelPredictions:
    """Train a fold-normalized PyTorch MLP and refit it on all labelled rows."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is unavailable; install the models extra.") from exc
    x, y, test, folds = _training_inputs(features, target, test_features, folds)
    if sparse.issparse(x) or sparse.issparse(test):
        raise TypeError("The MLP requires dense feature matrices")
    x = np.asarray(x, dtype=np.float32)
    test = None if test is None else np.asarray(test, dtype=np.float32)
    if len(hidden_sizes) != 2 or len(dropout) != 2:
        raise ValueError("The final MLP requires two hidden sizes and two dropout values")
    if min(batch_size, max_epochs, patience) <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    torch_device = _resolve_torch_device(torch, device)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    class ArrayDataset(Dataset):
        def __init__(self, values, labels, mean, scale):
            self.values, self.labels, self.mean, self.scale = values, labels, mean, scale

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            value = (self.values[index] - self.mean) / self.scale
            label = np.float32(self.labels[index])
            return torch.from_numpy(value), torch.tensor(label)

    class PriceMLP(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(width, int(hidden_sizes[0])),
                nn.GELU(),
                nn.Dropout(float(dropout[0])),
                nn.Linear(int(hidden_sizes[0]), int(hidden_sizes[1])),
                nn.GELU(),
                nn.Dropout(float(dropout[1])),
                nn.Linear(int(hidden_sizes[1]), 1),
            )

        def forward(self, values):
            return self.network(values).squeeze(-1)

    def normalization(values):
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale == 0] = 1.0
        return mean, scale

    def predict(model, values, mean, scale):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = (values[start : start + batch_size] - mean) / scale
                tensor = torch.from_numpy(batch).to(torch_device)
                outputs.append(model(tensor).detach().cpu().numpy())
        return np.concatenate(outputs)

    def train_epochs(model, loader, epochs):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for batch_features, batch_target in loader:
                batch_features = batch_features.to(torch_device)
                batch_target = batch_target.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_features), batch_target)
                loss.backward()
                optimizer.step()

    log_y = np.log1p(y).astype(np.float32)
    oof = np.zeros(len(y), dtype=np.float32)
    scores: list[float] = []
    selections: list[int] = []
    for fold_number, (train_index, valid_index) in enumerate(folds.split()):
        torch.manual_seed(random_state + fold_number)
        mean, scale = normalization(x[train_index])
        dataset = ArrayDataset(x[train_index], log_y[train_index], mean, scale)
        generator = torch.Generator().manual_seed(random_state + fold_number)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        model = PriceMLP(x.shape[1]).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()
        best_score = float("inf")
        best_epoch = 1
        best_state = None
        stale_epochs = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch_features, batch_target in loader:
                batch_features = batch_features.to(torch_device)
                batch_target = batch_target.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_features), batch_target)
                loss.backward()
                optimizer.step()
            validation = _positive_predictions(
                predict(model, x[valid_index], mean, scale), prediction_floor
            )
            score = smape(y[valid_index], validation)
            if score < best_score:
                best_score, best_epoch, stale_epochs = score, epoch, 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is None:
            raise RuntimeError("MLP training did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        predicted = _positive_predictions(
            predict(model, x[valid_index], mean, scale), prediction_floor
        )
        oof[valid_index] = predicted
        scores.append(smape(y[valid_index], predicted))
        selections.append(best_epoch)

    selected_epochs = _median_selection(selections)
    full_mean, full_scale = normalization(x)
    full_dataset = ArrayDataset(x, log_y, full_mean, full_scale)
    full_generator = torch.Generator().manual_seed(random_state)
    full_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, generator=full_generator
    )
    final_model = PriceMLP(x.shape[1]).to(torch_device)
    train_epochs(final_model, full_loader, selected_epochs)
    test_predictions = None
    if test is not None:
        test_predictions = _positive_predictions(
            predict(final_model, test, full_mean, full_scale), prediction_floor
        )
    bundle = {
        "state_dict": {
            name: value.detach().cpu() for name, value in final_model.state_dict().items()
        },
        "feature_mean": full_mean,
        "feature_scale": full_scale,
        "input_width": x.shape[1],
        "hidden_sizes": list(hidden_sizes),
    }
    parameters = {
        "hidden_sizes": list(hidden_sizes), "dropout": list(dropout),
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
        "selected_epochs": selected_epochs, "random_state": random_state,
        "device": str(torch_device),
    }
    result = ModelPredictions(
        "mlp", oof, test_predictions, scores, selections, parameters,
        feature_manifest_hash, folds.manifest_hash, bundle,
    )
    result.validate(
        expected_rows=len(y), expected_test_rows=None if test is None else test.shape[0]
    )
    return result


def optimize_smape_weights(
    predictions: np.ndarray,
    target: np.ndarray,
    *,
    minimum_weight: float = 0.01,
    model_order: Sequence[str] = MODEL_ORDER,
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
) -> np.ndarray:
    """Fit a non-negative, sum-to-one blend by direct multi-start SMAPE minimization."""
    matrix = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(y):
        raise ValueError("predictions must be shaped (n_samples, n_models)")
    if matrix.shape[1] != len(model_order) or tuple(model_order) != MODEL_ORDER:
        raise ValueError(f"Prediction columns must follow {MODEL_ORDER}")
    if not np.isfinite(matrix).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Ensemble predictions and targets must be finite, with positive targets")
    n_models = matrix.shape[1]
    if not 0 <= minimum_weight < 1.0 / n_models:
        raise ValueError("minimum_weight must be feasible for the number of models")

    def objective(weights):
        blended = np.maximum(matrix @ weights, prediction_floor)
        return smape(y, blended, percentage=False)

    equal = np.full(n_models, 1.0 / n_models)
    starts = [equal]
    remainder = (1.0 - (n_models - 1) * minimum_weight)
    for index in range(n_models):
        start = np.full(n_models, minimum_weight)
        start[index] = remainder
        starts.append(start)
    bounds = [(minimum_weight, 1.0) for _ in range(n_models)]
    constraint = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    optimized_candidates = []
    for start in starts:
        result = minimize(
            objective, start, method="SLSQP", bounds=bounds, constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and np.isfinite(result.fun) and np.isfinite(result.x).all():
            weights = np.asarray(result.x, dtype=np.float64)
            weights /= weights.sum()
            if (weights >= minimum_weight - 1e-8).all():
                optimized_candidates.append((objective(weights), weights))
    if not optimized_candidates:
        raise RuntimeError("SMAPE ensemble-weight optimization failed for every initialization")
    candidates = [
        (objective(np.asarray(start, dtype=np.float64)), np.asarray(start, dtype=np.float64))
        for start in starts
    ]
    candidates.extend(optimized_candidates)
    return min(candidates, key=lambda candidate: candidate[0])[1]


def fit_nonnegative_weights(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the final constrained four-model optimizer."""
    return optimize_smape_weights(predictions, target)


def blend_predictions(
    predictions: np.ndarray,
    weights: np.ndarray,
    *,
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=np.float64)
    blend_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(blend_weights):
        raise ValueError("One weight is required per prediction column")
    if not np.isfinite(matrix).all() or not np.isfinite(blend_weights).all():
        raise ValueError("Predictions and weights must be finite")
    if (blend_weights < 0).any() or not np.isclose(blend_weights.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to one")
    return np.maximum(matrix @ blend_weights, prediction_floor).astype(np.float32)


@dataclass
class EnsemblePredictions:
    model_results: list[ModelPredictions]
    model_order: tuple[str, ...]
    weights: np.ndarray
    oof_predictions: np.ndarray
    test_predictions: np.ndarray | None
    oof_smape: float
    feature_manifest_hash: str
    fold_manifest_hash: str
    optimization: dict[str, object]
    fold_assignments: FoldAssignments = field(repr=False)


def train_four_model_ensemble(
    features,
    target,
    *,
    sample_ids: Sequence[object],
    test_features=None,
    folds: FoldAssignments | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    feature_manifest_hash: str = "",
    prediction_floor: float = DEFAULT_PREDICTION_FLOOR,
    minimum_weight: float = 0.01,
    model_parameters: Mapping[str, Mapping[str, object]] | None = None,
) -> EnsemblePredictions:
    """Train the four final learners and fit their constrained OOF blend."""
    x = _feature_matrix(features)
    y = _target_vector(target, expected_rows=x.shape[0])
    if len(sample_ids) != len(y):
        raise ValueError("sample_ids and training features must have the same number of rows")
    if folds is None:
        folds = create_shared_folds(sample_ids, n_splits=n_splits, random_state=random_state)
    else:
        validate_shared_folds(folds, sample_ids)
        if folds.n_splits != n_splits or folds.random_state != random_state:
            raise ValueError("Configured fold settings do not match the supplied fold manifest")
    parameters = model_parameters or {}
    unknown_models = set(parameters) - set(MODEL_ORDER)
    if unknown_models:
        raise ValueError(f"Unknown model parameter groups: {sorted(unknown_models)}")
    trainers: dict[str, Callable[..., ModelPredictions]] = {
        "lightgbm": train_lightgbm_oof,
        "xgboost": train_xgboost_oof,
        "catboost": train_catboost_oof,
        "mlp": train_mlp_oof,
    }
    results = []
    for name in MODEL_ORDER:
        result = trainers[name](
            x, y, test_features=test_features, folds=folds,
            feature_manifest_hash=feature_manifest_hash,
            prediction_floor=prediction_floor, random_state=random_state,
            **dict(parameters.get(name, {})),
        )
        results.append(result)
    oof_matrix = np.column_stack([result.oof_predictions for result in results])
    weights = optimize_smape_weights(
        oof_matrix, y, minimum_weight=minimum_weight,
        model_order=MODEL_ORDER, prediction_floor=prediction_floor,
    )
    blended_oof = blend_predictions(oof_matrix, weights, prediction_floor=prediction_floor)
    blended_test = None
    if test_features is not None:
        test_matrix = np.column_stack([result.full_fit_test_predictions for result in results])
        blended_test = blend_predictions(test_matrix, weights, prediction_floor=prediction_floor)
    return EnsemblePredictions(
        results, MODEL_ORDER, weights, blended_oof, blended_test, smape(y, blended_oof),
        feature_manifest_hash, folds.manifest_hash,
        {"method": "SLSQP", "objective": "SMAPE", "minimum_weight": minimum_weight, "starts": 5},
        folds,
    )


def save_fold_assignments(
    assignments: FoldAssignments,
    path: str | Path,
    *,
    config_hash: str = "",
) -> None:
    """Persist ordered IDs and folds as a CSV with a hash-bearing sidecar."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        output,
        pd.DataFrame({"sample_id": assignments.sample_ids, "fold": assignments.folds}),
    )
    write_json(
        output.with_suffix(output.suffix + ".json"),
        {
            "rows": len(assignments.sample_ids),
            "n_splits": assignments.n_splits,
            "random_state": assignments.random_state,
            "manifest_hash": assignments.manifest_hash,
            "config_hash": config_hash,
        },
    )


def load_fold_assignments(
    path: str | Path,
    *,
    expected_ids: Sequence[object],
    config_hash: str = "",
) -> FoldAssignments:
    """Load and validate the exact fold plan created during feature assembly."""
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    frame = pd.read_csv(source, dtype={"sample_id": str})
    if frame.empty or list(frame.columns) != ["sample_id", "fold"]:
        raise ValueError("Fold CSV must contain exactly sample_id and fold")
    sample_ids = frame["sample_id"].to_numpy(dtype=str)
    try:
        numeric_folds = pd.to_numeric(frame["fold"], errors="raise")
        if not np.equal(numeric_folds, np.floor(numeric_folds)).all():
            raise ValueError("Fold values must be integers")
        folds = numeric_folds.to_numpy(dtype=np.int16)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fold values must be integers") from exc
    assignments = FoldAssignments(
        sample_ids=sample_ids,
        folds=folds,
        n_splits=int(metadata["n_splits"]),
        random_state=int(metadata["random_state"]),
        manifest_hash=str(metadata["manifest_hash"]),
    )
    if metadata.get("rows") != len(expected_ids):
        raise ValueError("Fold row count does not match the expected IDs")
    if metadata.get("config_hash", "") != config_hash:
        raise ValueError("Fold configuration hash does not match")
    validate_shared_folds(assignments, expected_ids)
    return assignments


def save_model_predictions(
    result: ModelPredictions,
    output_dir: str | Path,
    *,
    config_hash: str,
) -> None:
    """Persist one model's OOF/test predictions, metadata, and final fitted model."""
    destination = Path(output_dir) / result.model_name
    destination.mkdir(parents=True, exist_ok=True)
    oof_path = destination / "oof_predictions.npy"
    save_npy(oof_path, np.asarray(result.oof_predictions, dtype=np.float32))
    test_path: Path | None = None
    if result.full_fit_test_predictions is not None:
        test_path = destination / "test_predictions.npy"
        save_npy(
            test_path,
            np.asarray(result.full_fit_test_predictions, dtype=np.float32),
        )
    model_path: Path | None = None
    if result.final_model is not None:
        if result.model_name == "mlp":
            try:
                import torch
            except Exception as exc:
                raise RuntimeError("PyTorch is required to save the MLP model.") from exc
            model_path = destination / "model.pt"
            torch.save(result.final_model, model_path)
        else:
            try:
                import joblib
            except Exception as exc:
                raise RuntimeError("joblib is required to save tree models.") from exc
            model_path = destination / "model.joblib"
            joblib.dump(result.final_model, model_path)
    metadata = {
        "model_name": result.model_name,
        "fold_smape": result.fold_smape,
        "best_iterations_or_epochs": result.best_iterations_or_epochs,
        "parameters": result.parameters,
        "feature_manifest_hash": result.feature_manifest_hash,
        "fold_manifest_hash": result.fold_manifest_hash,
        "config_hash": config_hash,
        "artifacts": {
            "oof_predictions": {
                "file": oof_path.name,
                "sha256": file_sha256(oof_path),
            },
            "test_predictions": None
            if test_path is None
            else {"file": test_path.name, "sha256": file_sha256(test_path)},
            "model": None
            if model_path is None
            else {"file": model_path.name, "sha256": file_sha256(model_path)},
        },
    }
    write_json(destination / "manifest.json", metadata)


def save_ensemble_predictions(
    result: EnsemblePredictions,
    output_dir: str | Path,
    *,
    config_hash: str,
    submission_path: str | Path,
) -> None:
    """Persist final blend arrays and auditable ensemble metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_oof = np.column_stack(
        [item.oof_predictions for item in result.model_results]
    ).astype(np.float32)
    oof_path = destination / "oof_predictions.npy"
    blended_oof_path = destination / "blended_oof_predictions.npy"
    save_npy(oof_path, model_oof)
    save_npy(blended_oof_path, np.asarray(result.oof_predictions, dtype=np.float32))
    test_path: Path | None = None
    blended_test_path: Path | None = None
    if result.test_predictions is not None:
        model_test = np.column_stack(
            [item.full_fit_test_predictions for item in result.model_results]
        ).astype(np.float32)
        test_path = destination / "test_predictions.npy"
        blended_test_path = destination / "blended_test_predictions.npy"
        save_npy(test_path, model_test)
        save_npy(
            blended_test_path,
            np.asarray(result.test_predictions, dtype=np.float32),
        )
    write_json(
        destination / "model_columns.json",
        {
            "model_order": list(result.model_order),
            "config_hash": config_hash,
        },
    )
    weights = {
        "weights": dict(zip(result.model_order, map(float, result.weights), strict=True)),
        "model_order": list(result.model_order),
        "oof_smape": result.oof_smape,
        "feature_manifest_hash": result.feature_manifest_hash,
        "fold_manifest_hash": result.fold_manifest_hash,
        "optimization": result.optimization,
        "config_hash": config_hash,
        "artifacts": {
            "oof_predictions": file_sha256(oof_path),
            "blended_oof_predictions": file_sha256(blended_oof_path),
            "test_predictions": None if test_path is None else file_sha256(test_path),
            "blended_test_predictions": None
            if blended_test_path is None
            else file_sha256(blended_test_path),
            "submission": file_sha256(submission_path),
        },
    }
    write_json(destination / "weights.json", weights)
    metrics = {
        "config_hash": config_hash,
        "feature_manifest_hash": result.feature_manifest_hash,
        "fold_manifest_hash": result.fold_manifest_hash,
        "ensemble_oof_smape": result.oof_smape,
        "models": {
            item.model_name: {
                "fold_smape": item.fold_smape,
                "mean_smape": float(np.mean(item.fold_smape)),
            }
            for item in result.model_results
        },
    }
    write_json(destination / "metrics.json", metrics)
