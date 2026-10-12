from datetime import date

import numpy as np
import pandas as pd
import pytest

from ccp.labels import UNENGAGED, build_labels
from ccp.train import (baselines, check_class_counts, coerce_numeric, encode_account_type,
                       select_top_two)
from ccp.utils import DataError
from ccp.validate import forward_validation

CATS = ["Travel", "Referral"]
MAPPING = pd.DataFrame({"campaign_name": ["T1", "R1"], "category": ["Travel", "Referral"]})
LS, LE = date(2024, 2, 1), date(2024, 2, 28)


def ev(rows):
    return pd.DataFrame(rows, columns=["user_id", "campaign_name", "event_name", "event_date"])


def test_label_is_most_clicked_category():
    events = ev([("u1", "T1", "clicked", "2024-02-02"), ("u1", "T1", "clicked", "2024-02-03"),
                 ("u1", "R1", "clicked", "2024-02-04")])
    out = build_labels(events, MAPPING, CATS, "sent", "clicked", LS, LE).set_index("user_id")
    assert out.loc["u1", "dominant_category"] == "Travel"
    assert out.loc["u1", "total_clicks"] == 3


def test_label_tie_is_deterministic_by_share_then_category_order():
    events = ev([
        ("u1", "T1", "clicked", "2024-02-02"), ("u1", "R1", "clicked", "2024-02-02"),
        ("u1", "T1", "sent", "2024-02-01"), ("u1", "T1", "sent", "2024-02-01"),   # Travel share 0.5
        ("u1", "R1", "sent", "2024-02-01"),                                       # Referral share 1.0
    ])
    first = build_labels(events, MAPPING, CATS, "sent", "clicked", LS, LE)
    second = build_labels(events.sample(frac=1, random_state=1), MAPPING, CATS, "sent", "clicked", LS, LE)
    assert first.loc[0, "dominant_category"] == "Referral"
    assert first.equals(second)


def test_user_with_only_sends_is_unengaged():
    out = build_labels(ev([("u1", "T1", "sent", "2024-02-02")]), MAPPING, CATS, "sent", "clicked", LS, LE)
    assert out.loc[0, "dominant_category"] == UNENGAGED


def test_label_ignores_events_outside_window_and_unmapped_campaigns():
    events = ev([("u1", "T1", "clicked", "2024-03-15"), ("u1", "GENERIC", "clicked", "2024-02-02")])
    out = build_labels(events, MAPPING, CATS, "sent", "clicked", LS, LE)
    assert out.empty


def test_encode_account_type_is_total():
    m = {"Unknown": 0, "DCB": 1}
    assert encode_account_type("DCB", m) == 1
    assert encode_account_type("weird", m) == 0
    assert encode_account_type(None, m) == 0


def test_coerce_numeric_is_selective():
    df = pd.DataFrame({"a": ["1", "x"], "account_type": ["DCB", "SBM"]})
    out = coerce_numeric(df, exclude={"account_type"})
    assert out["a"].iloc[0] == 1 and np.isnan(out["a"].iloc[1])
    assert out["account_type"].tolist() == ["DCB", "SBM"]


def test_select_top_two():
    best, bc, second, sc = select_top_two(np.array([0.1, 0.6, 0.3]))
    assert (best, second) == (1, 2) and bc == 0.6 and sc == 0.3


def test_class_count_guard_blocks_thin_classes():
    y = pd.Series(["a"] * 20 + ["b"] * 3)
    with pytest.raises(DataError, match="fewer than 10"):
        check_class_counts(y, 10)
    check_class_counts(y, 3)


def test_majority_baseline_is_reported():
    out = baselines(np.array([0, 0, 0, 1]), ["A", "B"])
    assert out["majority_class"] == "A" and out["majority_class_baseline"] == 0.75


def test_forward_validation_flags_a_model_that_loses_to_majority():
    mapping = pd.DataFrame({"campaign_name": ["T1", "R1"], "category": ["Travel", "Referral"]})
    # 8 of 10 clicks are Travel, model always predicts Referral -> 20% hit rate vs 80% majority
    clicks = ev([(f"u{i}", "T1", "clicked", "2024-04-02") for i in range(8)]
                + [(f"u{i}", "R1", "clicked", "2024-04-02") for i in (8, 9)])
    preds = pd.DataFrame({"user_id": [f"u{i}" for i in range(10)],
                          "predicted_category": ["Referral"] * 10, "confidence": [0.7] * 10})
    res = forward_validation(clicks, mapping, preds, "clicked", date(2024, 4, 1), date(2024, 4, 30))
    assert res["hit_rate"] == pytest.approx(0.2)
    assert res["majority_class"] == "Travel"
    assert res["majority_class_baseline"] == pytest.approx(0.8)
    assert res["beats_majority_baseline"] is False
    assert res["hit_rate_excluding_majority_class"] == pytest.approx(1.0)


def test_forward_validation_errors_when_nothing_matches():
    mapping = pd.DataFrame({"campaign_name": ["T1"], "category": ["Travel"]})
    preds = pd.DataFrame({"user_id": ["zzz"], "predicted_category": ["Travel"], "confidence": [0.5]})
    with pytest.raises(DataError):
        forward_validation(ev([("u1", "T1", "clicked", "2024-04-02")]), mapping, preds, "clicked",
                           date(2024, 4, 1), date(2024, 4, 30))
