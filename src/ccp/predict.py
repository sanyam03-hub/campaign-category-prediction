"""Batch scoring: load a trained model, score every user, write predictions atomically."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ccp.features import order_features
from ccp.train import coerce_numeric, select_top_two
from ccp.utils import ConfigError, DataError


def load_model(model_path: str, config_path: str):
    import lightgbm as lgb

    try:
        booster = lgb.Booster(model_file=model_path)
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, lgb.basic.LightGBMError) as exc:
        raise DataError(f"Could not load model from {model_path} / {config_path}: {exc}") from exc
    for key in ("label_map", "best_iteration", "feature_names"):
        if key not in config:
            raise ConfigError(f"Model config is missing '{key}'")
    config["label_map"] = {int(k): v for k, v in config["label_map"].items()}
    return booster, config


def predict_all(features: pd.DataFrame, booster, config: dict, now: datetime | None = None) -> pd.DataFrame:
    """Score every row. The feature order is checked against the order saved at training."""
    if features.empty:
        raise DataError("No users to score")
    X = coerce_numeric(features.drop(columns=["user_id"]))
    X = order_features(X, config["feature_names"])
    proba = booster.predict(X, num_iteration=config["best_iteration"], validate_features=True)
    label_map = config["label_map"]

    rows = [select_top_two(p) for p in proba]
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame({
        "user_id": features["user_id"].values,
        "predicted_category": [label_map[r[0]] for r in rows],
        "confidence": np.round([r[1] for r in rows], 4),
        "second_category": [label_map[r[2]] for r in rows],
        "second_confidence": np.round([r[3] for r in rows], 4),
        "initiated_at": stamp,
    })


def distribution_shift(new: pd.DataFrame, previous: pd.DataFrame | None, max_shift: float = 0.10) -> list[str]:
    """Compare category shares with the previous run. Returns the categories that moved more
    than `max_shift` (as a share of users). Use this as a gate before replacing live output."""
    if previous is None or previous.empty:
        return []
    a = new["predicted_category"].value_counts(normalize=True)
    b = previous["predicted_category"].value_counts(normalize=True)
    diff = (a.reindex(a.index.union(b.index), fill_value=0) - b.reindex(a.index.union(b.index), fill_value=0)).abs()
    return sorted(diff[diff > max_shift].index.tolist())
