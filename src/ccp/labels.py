"""Training labels: each user's most-clicked category in the LABEL window."""
from __future__ import annotations

from datetime import date

import pandas as pd

from ccp.features import _to_date
from ccp.utils import require_columns

UNENGAGED = "UNENGAGED"


def build_labels(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    categories: list[str],
    sent_name: str,
    clicked_name: str,
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """Return DataFrame[user_id, dominant_category, total_clicks].

    Dominant category = highest click count; ties are broken by click share of sends, then by
    the order of `categories`, so the result is fully deterministic (the original SQL was not).
    Only clicks on mapped campaigns count. Users with no mapped clicks are UNENGAGED."""
    require_columns(events, ["user_id", "campaign_name", "event_name", "event_date"], "notification events")
    ev = events[["user_id", "campaign_name", "event_name", "event_date"]].copy()
    ev["event_date"] = _to_date(ev["event_date"])
    ev = ev.dropna(subset=["user_id", "event_date"])
    ev = ev[(ev["event_date"] >= window_start) & (ev["event_date"] <= window_end)]
    ev = ev[ev["event_name"].isin([sent_name, clicked_name])]
    ev = ev.merge(mapping[["campaign_name", "category"]], on="campaign_name", how="inner")

    ev["sent"] = (ev["event_name"] == sent_name).astype(int)
    ev["clicked"] = (ev["event_name"] == clicked_name).astype(int)
    per = ev.groupby(["user_id", "category"], as_index=False)[["sent", "clicked"]].sum()
    per["share"] = per["clicked"] / per["sent"].where(per["sent"] > 0)
    per["share"] = per["share"].fillna(0.0)
    per["order"] = per["category"].map({c: i for i, c in enumerate(categories)})

    per = per.sort_values(["user_id", "clicked", "share", "order"], ascending=[True, False, False, True])
    top = per.groupby("user_id", as_index=False).first()
    total = per.groupby("user_id")["clicked"].sum().rename("total_clicks")
    top = top.merge(total, on="user_id")
    top["dominant_category"] = top["category"].where(top["total_clicks"] > 0, UNENGAGED)
    return top[["user_id", "dominant_category", "total_clicks"]].reset_index(drop=True)
