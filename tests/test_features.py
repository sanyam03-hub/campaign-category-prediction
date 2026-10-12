from datetime import date

import pandas as pd
import pytest

from ccp import features as F
from ccp.utils import MissingFeatureError

SPEC = [
    {"name": "tenure_days", "kind": "tenure", "events": ["account_created"]},
    {"name": "has_premium", "kind": "flag", "events": ["premium_activated"]},
    {"name": "app_logins", "kind": "count", "events": ["login"]},
]
CATS = ["Travel", "Referral"]
WS, WE, FLOOR = date(2024, 1, 1), date(2024, 1, 31), date(2023, 1, 1)


def app_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "event", "event_date"])


def test_count_feature_only_counts_inside_window():
    app = app_df([
        ("u1", "login", "2023-12-31"),   # before window
        ("u1", "login", "2024-01-10"),
        ("u1", "login", "2024-01-31"),
        ("u1", "login", "2024-02-01"),   # after window end
    ])
    out = F.build_app_features(app, SPEC, WS, WE, FLOOR).set_index("user_id")
    assert out.loc["u1", "app_logins"] == 2


def test_lifetime_flag_cannot_see_the_future():
    # premium activated AFTER the feature window must not leak in
    app = app_df([("u1", "premium_activated", "2024-03-01"), ("u2", "premium_activated", "2023-06-01")])
    # pass the user list explicitly so u1 appears even though its only event is filtered out
    out = F.build_app_features(app, SPEC, WS, WE, FLOOR, users=pd.Index(["u1", "u2"])).set_index("user_id")
    assert out.loc["u1", "has_premium"] == 0
    assert out.loc["u2", "has_premium"] == 1


def test_tenure_is_days_between_first_event_and_window_end():
    app = app_df([("u1", "account_created", "2024-01-01")])
    out = F.build_app_features(app, SPEC, WS, WE, FLOOR).set_index("user_id")
    assert out.loc["u1", "tenure_days"] == 30


def notif(rows):
    return pd.DataFrame(rows, columns=["user_id", "campaign_name", "event_name", "event_date"])


def test_notification_features_and_never_clicked_sentinel():
    ev = notif([
        ("u1", "A", "sent", "2024-01-05"), ("u1", "A", "sent", "2024-01-06"),
        ("u1", "B", "clicked", "2024-01-25"),
        ("u2", "A", "sent", "2024-01-05"),
    ])
    out = F.build_notification_features(ev, "sent", "clicked", WS, WE).set_index("user_id")
    assert out.loc["u1", "total_sent"] == 2
    assert out.loc["u1", "days_since_last_click"] == 6
    assert out.loc["u1", "has_clicked"] == 1
    assert out.loc["u2", "days_since_last_click"] == F.NEVER_CLICKED
    assert out.loc["u2", "has_clicked"] == 0
    assert out.loc["u2", "click_rate"] == 0


def test_category_exposure_ignores_unmapped_campaigns():
    ev = notif([("u1", "A", "sent", "2024-01-05"), ("u1", "GENERIC", "sent", "2024-01-05")])
    mapping = pd.DataFrame({"campaign_name": ["A"], "category": ["Travel"]})
    out = F.build_category_exposure(ev, mapping, CATS, "sent", WS, WE).set_index("user_id")
    assert out.loc["u1", "sent_travel"] == 1
    assert out.loc["u1", "sent_referral"] == 0


def test_feature_table_has_contract_order_and_includes_notified_only_users():
    app = app_df([("u1", "login", "2024-01-10")])
    ev = notif([("u2", "A", "sent", "2024-01-05")])   # u2 has no app events
    mapping = pd.DataFrame({"campaign_name": ["A"], "category": ["Travel"]})
    table = F.build_feature_table(app, ev, mapping, SPEC, CATS, "sent", "clicked", WS, WE, FLOOR)
    assert list(table.columns) == ["user_id"] + F.feature_columns(SPEC, CATS)
    assert set(table["user_id"]) == {"u1", "u2"}
    assert table.set_index("user_id").loc["u2", "days_since_last_click"] == F.NEVER_CLICKED


def test_validate_schema_names_missing_and_rejects_extras():
    with pytest.raises(MissingFeatureError, match="b"):
        F.validate_schema(["a"], ["a", "b"])
    with pytest.raises(MissingFeatureError, match="Unexpected"):
        F.validate_schema(["a", "b", "z"], ["a", "b"])


def test_order_features_reorders_to_contract():
    df = pd.DataFrame({"b": [1], "a": [2]})
    assert list(F.order_features(df, ["a", "b"]).columns) == ["a", "b"]


def test_duplicate_feature_names_are_rejected(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text("features:\n  - {name: x, kind: count, events: [e]}\n  - {name: x, kind: flag, events: [e]}\n")
    from ccp.utils import ConfigError
    with pytest.raises(ConfigError):
        F.load_feature_spec(str(p))
