"""Raw event ingestion with guards against the failure that bit the original system:
a silent format change (JSON -> CSV) produced rows that parsed to all NULLs without any error."""
from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ccp.utils import DataError


def _open(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _get(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first(obj: dict, paths: list[str]):
    for p in paths:
        v = _get(obj, p)
        if v not in (None, ""):
            return v
    return None


def check_suffix(path: Path, allowed: list[str]) -> None:
    if not any(path.name.endswith(s) for s in allowed):
        raise DataError(f"{path.name}: unexpected file type. Allowed: {allowed}. "
                        "A silent format change would otherwise produce NULL rows.")


def parse_lines(lines, fields: dict, ts_format: str) -> pd.DataFrame:
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            obj = {}  # counted as a NULL row below, so the null-rate gate sees it
        ts = _first(obj, fields["ts"]) if obj else None
        parsed = None
        if ts is not None:
            try:
                parsed = datetime.strptime(str(ts), ts_format)
            except ValueError:
                parsed = None
        rows.append({
            "event_name": _first(obj, fields["event_name"]) if obj else None,
            "campaign_name": _first(obj, fields["campaign_name"]) if obj else None,
            "user_id": _first(obj, fields["user_id"]) if obj else None,
            "timestamp": parsed,
        })
    return pd.DataFrame(rows, columns=["event_name", "campaign_name", "user_id", "timestamp"])


def null_rate_gate(df: pd.DataFrame, max_null_rate: float) -> dict:
    """Fail loudly if too many rows are missing user_id or campaign_name."""
    if df.empty:
        raise DataError("Ingest produced zero rows")
    rates = {c: float(df[c].isna().mean()) for c in ("event_name", "campaign_name", "user_id", "timestamp")}
    bad = {c: r for c, r in rates.items() if r > max_null_rate}
    if bad:
        raise DataError(f"NULL-rate gate failed (limit {max_null_rate:.1%}): {bad}. "
                        "Check whether the export format changed.")
    return rates


def ingest_day(path: str | Path, cfg: dict, user_id_prefix: str = "") -> pd.DataFrame:
    path = Path(path)
    icfg = cfg["ingest"]
    check_suffix(path, icfg["allowed_suffixes"])
    with _open(path) as fh:
        df = parse_lines(fh, icfg["fields"], icfg["ts_format"])
    null_rate_gate(df, icfg["max_null_rate"])
    if user_id_prefix:
        df = df[df["user_id"].astype(str).str.startswith(user_id_prefix)]
    df = df.drop_duplicates()
    df["event_date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df.reset_index(drop=True)


def write_day_idempotent(df: pd.DataFrame, directory: str | Path, day: str) -> Path:
    """One file per day, replaced atomically. Re-running a day replaces it; it never doubles it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"events_{day}.parquet"
    tmp = final.with_name(final.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    return final


def freshness_gate(directory: str | Path, expected_day: str) -> None:
    """Fail scoring if yesterday's file is missing (the original exited 'successfully' instead)."""
    if not (Path(directory) / f"events_{expected_day}.parquet").exists():
        raise DataError(f"No events file for {expected_day}; refusing to score on stale data.")
