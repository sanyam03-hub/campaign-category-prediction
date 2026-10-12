"""Small shared helpers."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


class CCPError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(CCPError):
    pass


class DataError(CCPError):
    pass


class MissingFeatureError(CCPError):
    pass


def slug(name: str) -> str:
    """'Forex & Remittance' -> 'forex_remittance'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def sent_column(category: str) -> str:
    return f"sent_{slug(category)}"


def normalize_name(value: object) -> str:
    """Lowercase, trim, collapse whitespace. Used for rule and override matching."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{what} is missing required columns: {missing}. Found: {list(df.columns)}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise DataError(f"Unsupported file type: {path} (use .parquet or .csv)")


def read_dir(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"Directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix in (".parquet", ".csv"))
    if not files:
        raise DataError(f"No .parquet or .csv files in {directory}")
    return pd.concat([read_table(p) for p in files], ignore_index=True)


def write_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write to a temp file, then rename, so a failed run never leaves a half-written output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    elif path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        raise DataError(f"Unsupported output type: {path}")
    os.replace(tmp, path)
