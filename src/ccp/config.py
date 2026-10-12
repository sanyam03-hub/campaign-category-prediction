"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from ccp.utils import ConfigError

REQUIRED_TOP_LEVEL = [
    "paths", "notification_events", "windows", "classification",
    "features", "ingest", "model", "output",
]


@dataclass(frozen=True)
class Windows:
    feature_start: date
    feature_end: date
    label_start: date
    label_end: date
    validation_start: date
    validation_end: date
    lifetime_floor: date
    rolling_days: int


def _parse_date(value: object, key: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ConfigError(f"windows.{key} must be YYYY-MM-DD, got {value!r}") from exc


def get_windows(cfg: dict) -> Windows:
    raw = cfg["windows"]
    names = ["feature_start", "feature_end", "label_start", "label_end",
             "validation_start", "validation_end", "lifetime_floor"]
    missing = [n for n in names + ["rolling_days"] if n not in raw]
    if missing:
        raise ConfigError(f"windows is missing keys: {missing}")
    parsed = {n: _parse_date(raw[n], n) for n in names}
    rolling = raw["rolling_days"]
    if not isinstance(rolling, int) or rolling <= 0:
        raise ConfigError("windows.rolling_days must be a positive integer")
    w = Windows(rolling_days=rolling, **parsed)

    if not (w.feature_start <= w.feature_end):
        raise ConfigError("feature_start must be on or before feature_end")
    if not (w.label_start <= w.label_end):
        raise ConfigError("label_start must be on or before label_end")
    if not (w.validation_start <= w.validation_end):
        raise ConfigError("validation_start must be on or before validation_end")
    if not (w.feature_end < w.label_start):
        raise ConfigError(
            "The feature window must end BEFORE the label window starts "
            "(otherwise features can see the label period)."
        )
    if not (w.label_end < w.validation_start):
        raise ConfigError("The label window must end before the validation window starts.")
    if w.lifetime_floor > w.feature_start:
        raise ConfigError("lifetime_floor must be on or before feature_start")
    return w


def validate_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ConfigError("Config file must contain a YAML mapping")
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ConfigError(f"Config is missing top-level sections: {missing}")
    ne = cfg["notification_events"]
    if not {"sent", "clicked"} <= set(ne):
        raise ConfigError("notification_events must define 'sent' and 'clicked'")
    if ne["sent"] == ne["clicked"]:
        raise ConfigError("notification_events.sent and .clicked must differ")
    get_windows(cfg)  # raises on bad windows
    split = cfg["model"].get("split", {})
    if split.get("test_size", 0) + split.get("val_size", 0) >= 0.9:
        raise ConfigError("model.split test_size + val_size must be below 0.9")
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config not found: {path}. Copy config/pipeline.example.yaml to config/pipeline.yaml."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return validate_config(cfg)
