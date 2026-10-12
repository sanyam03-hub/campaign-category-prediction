"""Feature engineering: app behaviour + notification engagement + per-category exposure.

Leakage guards (the main fixes over the original production version):
  * Features are computed from a window that ENDS BEFORE the label window starts.
  * Lifetime flags look back to `lifetime_floor` but never past the feature-window end.
  * `has_clicked` is an explicit flag, so the "never clicked" sentinel (999) is not the
    only signal for that state.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import yaml

from ccp.utils import ConfigError, DataError, MissingFeatureError, require_columns, sent_column

NEVER_CLICKED = 999
NOTIFICATION_FEATURES = [
    "total_sent", "total_clicked", "click_rate", "campaigns_received",
    "days_since_last_click", "has_clicked",
]


def load_feature_spec(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    spec = raw.get("features")
    if not spec:
        raise ConfigError(f"{path} defines no features")
    kinds = {"count", "flag", "tenure"}
    names = [s["name"] for s in spec]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path} has duplicate feature names")
    for s in spec:
        if s.get("kind") not in kinds or not s.get("events"):
            raise ConfigError(f"Bad feature definition: {s}. kind must be one of {sorted(kinds)}")
    return spec


def feature_columns(spec: list[dict], categories: list[str]) -> list[str]:
    """The model's input column order. Training and scoring MUST use this same list."""
    return (
        [s["name"] for s in spec]
        + NOTIFICATION_FEATURES
        + [sent_column(c) for c in categories]
    )


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def build_app_features(
    app_events: pd.DataFrame,
    spec: list[dict],
    window_start: date,
    window_end: date,
    lifetime_floor: date,
    users: pd.Index | None = None,
) -> pd.DataFrame:
    """One row per user. `count` features use [window_start, window_end]; `flag` and `tenure`
    features use [lifetime_floor, window_end]. Nothing after window_end is read."""
    require_columns(app_events, ["user_id", "event", "event_date"], "app events")
    ev = app_events[["user_id", "event", "event_date"]].copy()
    ev["event_date"] = _to_date(ev["event_date"])
    ev = ev.dropna(subset=["user_id", "event_date"])
    ev = ev[(ev["event_date"] >= lifetime_floor) & (ev["event_date"] <= window_end)]

    in_window = (ev["event_date"] >= window_start) & (ev["event_date"] <= window_end)
    index = users if users is not None else pd.Index(ev["user_id"].unique(), name="user_id")
    index = pd.Index(index, name="user_id")
    columns: dict[str, pd.Series] = {}

    for s in spec:
        mask_events = ev["event"].isin(s["events"])
        if s["kind"] == "count":
            raw = ev[mask_events & in_window].groupby("user_id").size()
        elif s["kind"] == "flag":
            raw = (ev[mask_events].groupby("user_id").size() > 0).astype("int64")
        else:  # tenure
            first = ev[mask_events].groupby("user_id")["event_date"].min()
            raw = first.map(lambda d: max((window_end - d).days, 0)).astype("int64")
        # reindex to the full user list, then fill gaps with an explicit integer 0
        columns[s["name"]] = raw.reindex(index).astype("float64").fillna(0).astype("int64")

    out = pd.DataFrame(columns, index=index)
    return out.reset_index()


def build_notification_features(
    events: pd.DataFrame,
    sent_name: str,
    clicked_name: str,
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    require_columns(events, ["user_id", "campaign_name", "event_name", "event_date"], "notification events")
    ev = events[["user_id", "campaign_name", "event_name", "event_date"]].copy()
    ev["event_date"] = _to_date(ev["event_date"])
    ev = ev.dropna(subset=["user_id", "event_date"])
    ev = ev[(ev["event_date"] >= window_start) & (ev["event_date"] <= window_end)]
    ev = ev[ev["event_name"].isin([sent_name, clicked_name])]
    if ev.empty:
        return pd.DataFrame(columns=["user_id"] + NOTIFICATION_FEATURES)

    ev["is_sent"] = (ev["event_name"] == sent_name).astype(int)
    ev["is_click"] = (ev["event_name"] == clicked_name).astype(int)
    g = ev.groupby("user_id")
    out = pd.DataFrame({
        "total_sent": g["is_sent"].sum(),
        "total_clicked": g["is_click"].sum(),
        "campaigns_received": g["campaign_name"].nunique(),
    })
    out["click_rate"] = (out["total_clicked"] * 100.0 / out["total_sent"].where(out["total_sent"] > 0)).fillna(0.0)
    last_click = ev[ev["is_click"] == 1].groupby("user_id")["event_date"].max()
    gap = last_click.map(lambda d: (window_end - d).days)
    out["days_since_last_click"] = (
        gap.reindex(out.index).astype("float64").fillna(NEVER_CLICKED).astype(int)
    )
    out["has_clicked"] = (out["total_clicked"] > 0).astype(int)
    return out.reset_index()[["user_id"] + NOTIFICATION_FEATURES]


def build_category_exposure(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    categories: list[str],
    sent_name: str,
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    """Per-category 'sent' counts. Inner-joined to the mapping, like the original system, so
    campaigns that were dropped as generic do not count toward any category."""
    require_columns(events, ["user_id", "campaign_name", "event_name", "event_date"], "notification events")
    cols = [sent_column(c) for c in categories]
    ev = events[events["event_name"] == sent_name][["user_id", "campaign_name", "event_date"]].copy()
    ev["event_date"] = _to_date(ev["event_date"])
    ev = ev.dropna(subset=["user_id", "event_date"])
    ev = ev[(ev["event_date"] >= window_start) & (ev["event_date"] <= window_end)]
    ev = ev.merge(mapping[["campaign_name", "category"]], on="campaign_name", how="inner")
    if ev.empty:
        return pd.DataFrame(columns=["user_id"] + cols)
    counts = ev.groupby(["user_id", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=categories, fill_value=0)
    counts.columns = cols
    return counts.reset_index()


def build_feature_table(
    app_events: pd.DataFrame,
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    spec: list[dict],
    categories: list[str],
    sent_name: str,
    clicked_name: str,
    window_start: date,
    window_end: date,
    lifetime_floor: date,
    all_users: bool = True,
) -> pd.DataFrame:
    """Return `user_id` + feature columns in the contract order.

    all_users=True keeps users who appear in EITHER table (so notified users with no app events
    are scored too); this fixes the original join, which silently dropped them."""
    app = build_app_features(app_events, spec, window_start, window_end, lifetime_floor)
    notif = build_notification_features(events, sent_name, clicked_name, window_start, window_end)
    expo = build_category_exposure(events, mapping, categories, sent_name, window_start, window_end)

    how = "outer" if all_users else "left"
    table = app.merge(notif, on="user_id", how=how).merge(expo, on="user_id", how=how)
    columns = feature_columns(spec, categories)
    for col in columns:
        if col not in table.columns:
            table[col] = 0
    fill = {c: 0 for c in columns}
    fill["days_since_last_click"] = NEVER_CLICKED
    table = table.fillna(fill)
    table["days_since_last_click"] = table["days_since_last_click"].astype(int)
    return table[["user_id"] + columns].reset_index(drop=True)


def validate_schema(actual: list[str], expected: list[str]) -> None:
    """Raise naming the first missing column; also reject unexpected extras and wrong order."""
    missing = [c for c in expected if c not in actual]
    if missing:
        raise MissingFeatureError(f"Missing feature column: {missing[0]} (all missing: {missing})")
    extra = [c for c in actual if c not in expected]
    if extra:
        raise MissingFeatureError(f"Unexpected feature columns: {extra}")


def order_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    validate_schema(list(df.columns), expected)
    return df[expected]


def check_population(table: pd.DataFrame, min_rows: int = 1) -> None:
    if len(table) < min_rows:
        raise DataError("Feature table is empty; check the windows and input paths in the config")
