"""Forward validation: do predictions made BEFORE the validation window match later clicks?

Reports the model next to the baselines that matter. A hit rate on its own is misleading
when one category dominates clicks (see docs/lessons-learned.md).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ccp.features import _to_date
from ccp.utils import DataError, require_columns


def forward_validation(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    predictions: pd.DataFrame,
    clicked_name: str,
    window_start: date,
    window_end: date,
) -> dict:
    require_columns(events, ["user_id", "campaign_name", "event_name", "event_date"], "notification events")
    require_columns(predictions, ["user_id", "predicted_category", "confidence"], "predictions")

    ev = events[events["event_name"] == clicked_name][["user_id", "campaign_name", "event_date"]].copy()
    ev["event_date"] = _to_date(ev["event_date"])
    ev = ev[(ev["event_date"] >= window_start) & (ev["event_date"] <= window_end)]
    ev = ev.merge(mapping[["campaign_name", "category"]], on="campaign_name", how="inner")
    ev = ev.rename(columns={"category": "actual_category"})
    joined = ev.merge(predictions[["user_id", "predicted_category", "confidence"]], on="user_id", how="inner")
    if joined.empty:
        raise DataError("No validation clicks could be matched to a prediction")

    joined["hit"] = (joined["actual_category"] == joined["predicted_category"]).astype(int)
    n = len(joined)
    n_cat = max(joined["actual_category"].nunique(), 1)
    shares = joined["actual_category"].value_counts(normalize=True)
    majority = shares.index[0]

    majority_hit = float(shares.iloc[0])
    non_majority = joined[joined["actual_category"] != majority]
    result = {
        "clicks": int(n),
        "users": int(joined["user_id"].nunique()),
        "hit_rate": float(joined["hit"].mean()),
        "uniform_baseline": 1.0 / n_cat,
        "majority_class": str(majority),
        "majority_class_baseline": majority_hit,
        "beats_majority_baseline": bool(joined["hit"].mean() > majority_hit),
        "hit_rate_excluding_majority_class": (
            float(non_majority["hit"].mean()) if len(non_majority) else None
        ),
        "by_actual_category": (
            joined.groupby("actual_category")["hit"].agg(clicks="size", correct="sum").assign(
                hit_rate=lambda d: d["correct"] / d["clicks"]).round(4).reset_index().to_dict("records")
        ),
        "by_confidence": _confidence_buckets(joined),
    }
    return result


def _confidence_buckets(joined: pd.DataFrame) -> list[dict]:
    bins = [0.0, 0.25, 0.4, 0.6, 1.0001]
    labels = ["<0.25", "0.25-0.4", "0.4-0.6", ">=0.6"]
    b = pd.cut(joined["confidence"], bins=bins, labels=labels, right=False)
    g = joined.groupby(b, observed=True)["hit"].agg(clicks="size", correct="sum")
    g["hit_rate"] = (g["correct"] / g["clicks"]).round(4)
    return g.reset_index().rename(columns={"confidence": "bucket"}).to_dict("records")
