"""Model training and evaluation. Nothing here runs unless you call it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccp.features import order_features
from ccp.utils import DataError


def encode_account_type(value: object, mapping: dict[str, int]) -> int:
    """Total over all inputs: unknown or missing values map to the 'Unknown' code (0)."""
    try:
        return int(mapping.get(str(value), mapping.get("Unknown", 0)))
    except (TypeError, ValueError):
        return 0


def coerce_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Coerce object columns to numeric (unparseable -> NaN), leaving `exclude` untouched."""
    exclude = exclude or set()
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def select_top_two(proba_row: np.ndarray) -> tuple[int, float, int, float]:
    order = np.argsort(proba_row)
    best, second = int(order[-1]), int(order[-2])
    return best, float(proba_row[best]), second, float(proba_row[second])


def check_class_counts(y: pd.Series, min_samples: int) -> None:
    counts = y.value_counts()
    thin = counts[counts < min_samples]
    if len(counts) < 2:
        raise DataError(f"Training needs at least 2 classes, found {len(counts)}")
    if not thin.empty:
        raise DataError(
            f"These classes have fewer than {min_samples} training rows: {thin.to_dict()}. "
            "Merge them into another class, collect more data, or lower model.min_class_samples."
        )


def baselines(y_true: np.ndarray, classes: list[str]) -> dict:
    """Reference scores that any headline metric must be read against."""
    n = len(classes)
    values, counts = np.unique(y_true, return_counts=True)
    majority_share = float(counts.max() / counts.sum()) if len(counts) else 0.0
    return {
        "uniform_baseline": 1.0 / n,
        "majority_class_baseline": majority_share,
        "majority_class": classes[int(values[counts.argmax()])] if len(values) else None,
    }


def train_model(features: pd.DataFrame, labels: pd.DataFrame, columns: list[str], cfg: dict) -> dict:
    """Train a multiclass LightGBM model. Returns model, metrics, label map and config."""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    m = cfg["model"]
    data = features.merge(labels[labels["dominant_category"] != "UNENGAGED"][["user_id", "dominant_category"]],
                          on="user_id", how="inner")
    if data.empty:
        raise DataError("No labelled users after joining features and labels. Check the windows.")
    check_class_counts(data["dominant_category"], m.get("min_class_samples", 10))

    X = order_features(data.drop(columns=["user_id", "dominant_category"]), columns).copy()
    enc = LabelEncoder()
    y = enc.fit_transform(data["dominant_category"])
    classes = list(enc.classes_)

    holdout = m["split"]["test_size"] + m["split"]["val_size"]
    seed = m["split"]["seed"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=holdout, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=m["split"]["test_size"] / holdout, random_state=seed, stratify=y_tmp
    )

    weights = None
    cw = m.get("class_weights") or {}
    if cw:
        unknown = set(cw) - set(classes)
        if unknown:
            raise DataError(f"model.class_weights names unknown classes: {sorted(unknown)}")
        w = {i: cw.get(c, 1.0) for i, c in enumerate(classes)}
        weights = np.array([w[i] for i in y_tr])

    params = dict(m["params"], num_class=len(classes))
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=weights)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=m["num_boost_round"], valid_sets=[val_set],
        callbacks=[lgb.early_stopping(m["early_stopping_rounds"]), lgb.log_evaluation(100)],
    )

    proba = booster.predict(X_te, num_iteration=booster.best_iteration)
    pred = proba.argmax(axis=1)
    k2, k3 = min(2, len(classes)), min(3, len(classes))
    metrics = {
        "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
        "best_iteration": int(booster.best_iteration),
        "accuracy": float(accuracy_score(y_te, pred)),
        "top2_accuracy": float(top_k_accuracy_score(y_te, proba, k=k2, labels=range(len(classes)))),
        "top3_accuracy": float(top_k_accuracy_score(y_te, proba, k=k3, labels=range(len(classes)))),
        "baselines": baselines(y_te, classes),
        "per_class": classification_report(y_te, pred, labels=range(len(classes)),
                                           target_names=classes, output_dict=True, zero_division=0),
    }
    importance = (pd.DataFrame({"feature": columns,
                                "gain": booster.feature_importance(importance_type="gain")})
                  .sort_values("gain", ascending=False))
    model_config = {
        "label_map": {int(i): c for i, c in enumerate(classes)},
        "best_iteration": int(booster.best_iteration),
        "feature_names": columns,          # lets scoring verify the column contract
    }
    return {"booster": booster, "metrics": metrics, "importance": importance, "model_config": model_config}


def save_model(result: dict, model_path: str, config_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    result["booster"].save_model(model_path)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(result["model_config"], fh, indent=2)
