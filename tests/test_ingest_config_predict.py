import json
from datetime import datetime

import pandas as pd
import pytest

from ccp.config import get_windows, validate_config
from ccp.ingest import (check_suffix, freshness_gate, null_rate_gate, parse_lines,
                        write_day_idempotent)
from ccp.predict import distribution_shift
from ccp.utils import ConfigError, DataError, normalize_name, slug

FIELDS = {
    "event_name": ["event_name", "eventName"],
    "campaign_name": ["campaign_name", "eventProps.Campaign name"],
    "user_id": ["user_id", "profile.identity"],
    "ts": ["ts"],
}
FMT = "%Y%m%d%H%M%S"


def line(**kw):
    return json.dumps(kw)


def test_parse_handles_flat_and_nested_field_names():
    lines = [
        line(event_name="sent", campaign_name="A", user_id="u1", ts="20240105101500"),
        line(eventName="clicked", eventProps={"Campaign name": "B"}, profile={"identity": "u2"}, ts="20240106101500"),
    ]
    df = parse_lines(lines, FIELDS, FMT)
    assert df["user_id"].tolist() == ["u1", "u2"]
    assert df["campaign_name"].tolist() == ["A", "B"]
    assert df["timestamp"].iloc[0] == datetime(2024, 1, 5, 10, 15, 0)


def test_csv_content_in_a_json_reader_trips_the_null_rate_gate():
    # This is the original June incident: wrong-format lines parse to all-NULL rows.
    csv_lines = ["sent,Campaign A,user1,20240105101500", "clicked,Campaign B,user2,20240106101500"]
    df = parse_lines(csv_lines, FIELDS, FMT)
    assert df["user_id"].isna().all()
    with pytest.raises(DataError, match="NULL-rate gate"):
        null_rate_gate(df, 0.01)


def test_suffix_check_rejects_unexpected_file_types():
    from pathlib import Path
    check_suffix(Path("events.jsonl.gz"), [".jsonl", ".jsonl.gz"])
    with pytest.raises(DataError):
        check_suffix(Path("events.csv.gz"), [".jsonl", ".jsonl.gz"])


def test_rerunning_a_day_replaces_the_file_instead_of_doubling(tmp_path):
    df = pd.DataFrame({"user_id": ["u1"], "event_name": ["sent"]})
    write_day_idempotent(df, tmp_path, "2024-01-05")
    write_day_idempotent(df, tmp_path, "2024-01-05")
    assert len(pd.read_parquet(tmp_path / "events_2024-01-05.parquet")) == 1


def test_freshness_gate_blocks_scoring_on_missing_day(tmp_path):
    with pytest.raises(DataError, match="stale"):
        freshness_gate(tmp_path, "2024-01-05")


def base_cfg():
    return {
        "paths": {}, "classification": {}, "features": {}, "ingest": {}, "output": {},
        "notification_events": {"sent": "sent", "clicked": "clicked"},
        "model": {"split": {"test_size": 0.15, "val_size": 0.15}},
        "windows": {"feature_start": "2024-01-01", "feature_end": "2024-02-14",
                    "label_start": "2024-02-15", "label_end": "2024-03-30",
                    "validation_start": "2024-03-31", "validation_end": "2024-05-14",
                    "lifetime_floor": "2023-01-01", "rolling_days": 45},
    }


def test_valid_config_passes():
    validate_config(base_cfg())


def test_overlapping_feature_and_label_windows_are_rejected():
    cfg = base_cfg()
    cfg["windows"]["feature_end"] = "2024-02-20"      # runs into the label window
    with pytest.raises(ConfigError, match="BEFORE the label window"):
        get_windows(cfg)


def test_bad_date_is_reported_with_the_key_name():
    cfg = base_cfg()
    cfg["windows"]["label_end"] = "30/03/2024"
    with pytest.raises(ConfigError, match="label_end"):
        get_windows(cfg)


def test_sent_and_clicked_names_must_differ():
    cfg = base_cfg()
    cfg["notification_events"] = {"sent": "x", "clicked": "x"}
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_distribution_shift_gate():
    prev = pd.DataFrame({"predicted_category": ["A"] * 80 + ["B"] * 20})
    same = pd.DataFrame({"predicted_category": ["A"] * 78 + ["B"] * 22})
    moved = pd.DataFrame({"predicted_category": ["A"] * 50 + ["B"] * 50})
    assert distribution_shift(same, prev) == []
    assert distribution_shift(moved, prev) == ["A", "B"]
    assert distribution_shift(moved, None) == []


def test_helpers():
    assert slug("Forex & Remittance") == "forex_remittance"
    assert normalize_name("  A   b ") == "a b"
