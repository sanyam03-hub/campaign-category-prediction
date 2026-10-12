"""Campaign classification: generic filter -> keyword rules -> optional semantic fallback -> overrides.

Differences from the original production version (see docs/lessons-learned.md):
  * Generic campaign names are dropped BEFORE classification, not after.
  * Overrides match on normalised names and unmatched override keys are reported.
  * Low-confidence semantic matches go to a review file; they are not force-assigned.
  * One implementation is shared by training and daily scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import yaml

from ccp.utils import ConfigError, DataError, normalize_name, require_columns


@dataclass
class Rule:
    category: str
    keywords: list[str]
    exclude: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    categories: list[str]
    rules: list[Rule]
    generic_patterns: list[re.Pattern]
    descriptions: dict[str, str]


def load_taxonomy(path: str) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for key in ("categories", "rules"):
        if key not in raw:
            raise ConfigError(f"Taxonomy file {path} is missing '{key}'")
    categories = list(raw["categories"])
    rules = []
    for r in raw["rules"]:
        if r["category"] not in categories:
            raise ConfigError(f"Rule category {r['category']!r} is not in 'categories'")
        rules.append(Rule(r["category"], [normalize_name(k) for k in r["keywords"]],
                          [normalize_name(k) for k in r.get("exclude", [])]))
    generic = [re.compile(p) for p in raw.get("generic_name_patterns", [])]
    return Taxonomy(categories, rules, generic, raw.get("category_descriptions", {}))


def load_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {normalize_name(k): v for k, v in (raw.get("overrides") or {}).items()}


def is_generic(name: str, taxonomy: Taxonomy) -> bool:
    n = normalize_name(name)
    return any(p.search(n) for p in taxonomy.generic_patterns)


def keyword_classify(name: str, taxonomy: Taxonomy) -> str | None:
    """Return the first matching category, or None. Rule order is precedence."""
    n = normalize_name(name)
    for rule in taxonomy.rules:
        if any(k in n for k in rule.keywords):
            if rule.exclude and any(e in n for e in rule.exclude):
                continue  # excluded: fall through to later rules
            return rule.category
    return None


def semantic_classify(texts: list[str], taxonomy: Taxonomy, model_name: str):
    """Return (categories, similarity scores). Needs sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigError(
            "classification.use_semantic is true but sentence-transformers is not installed. "
            "Run: pip install -r requirements-optional.txt"
        ) from exc
    names = list(taxonomy.descriptions)
    if not names:
        raise ConfigError("Taxonomy has no category_descriptions; cannot run the semantic pass")
    model = SentenceTransformer(model_name)
    cat_emb = model.encode([taxonomy.descriptions[c] for c in names], convert_to_tensor=True)
    txt_emb = model.encode(texts, convert_to_tensor=True)
    sims = util.cos_sim(txt_emb, cat_emb)
    best = sims.argmax(dim=1).tolist()
    score = sims.max(dim=1).values.tolist()
    return [names[i] for i in best], score


def apply_overrides(df: pd.DataFrame, overrides: dict[str, str], categories: list[str]):
    """Apply overrides by normalised name. Returns (df, unmatched_keys)."""
    bad = {v for v in overrides.values() if v not in categories}
    if bad:
        raise ConfigError(f"Overrides use unknown categories: {sorted(bad)}")
    out = df.copy()
    norm = out["campaign_name"].map(normalize_name)
    matched: set[str] = set()
    for key, cat in overrides.items():
        mask = norm == key
        if mask.any():
            out.loc[mask, "category"] = cat
            out.loc[mask, "confidence"] = 0.95
            out.loc[mask, "source"] = "override"
            matched.add(key)
    return out, sorted(set(overrides) - matched)


def classify_campaigns(
    campaigns: pd.DataFrame,
    taxonomy: Taxonomy,
    overrides: dict[str, str] | None = None,
    use_semantic: bool = False,
    semantic_model: str = "all-MiniLM-L6-v2",
    min_confidence: float = 0.20,
) -> dict:
    """Classify campaigns.

    Returns a dict with:
      mapping      DataFrame[campaign_name, category, confidence, source]
      review       DataFrame of campaigns that need a human (unmatched or low confidence)
      dropped      DataFrame of generic campaigns removed before classification
      unmatched_overrides  list of override keys that matched nothing
    """
    require_columns(campaigns, ["campaign_name"], "campaigns table")
    df = campaigns.drop_duplicates(subset=["campaign_name"], keep="first").copy()
    df = df[df["campaign_name"].notna()]
    for col in ("title", "message"):
        if col not in df.columns:
            df[col] = ""

    generic_mask = df["campaign_name"].map(lambda n: is_generic(n, taxonomy))
    dropped = df[generic_mask][["campaign_name"]].copy()
    df = df[~generic_mask].copy()

    df["category"] = df["campaign_name"].map(lambda n: keyword_classify(n, taxonomy))
    df["confidence"] = df["category"].map(lambda c: 1.0 if c else float("nan"))
    df["source"] = df["category"].map(lambda c: "keyword" if c else "")

    pending = df["category"].isna()
    review_rows = []
    if pending.any():
        if use_semantic:
            texts = (df.loc[pending, "campaign_name"].fillna("") + " | "
                     + df.loc[pending, "title"].fillna("") + " | "
                     + df.loc[pending, "message"].fillna("")).tolist()
            cats, scores = semantic_classify(texts, taxonomy, semantic_model)
            df.loc[pending, "category"] = cats
            df.loc[pending, "confidence"] = scores
            df.loc[pending, "source"] = "semantic"
        else:
            df.loc[pending, "source"] = "unmatched"

    df, unmatched_overrides = apply_overrides(df, overrides or {}, taxonomy.categories)

    needs_review = df["category"].isna() | ((df["source"] == "semantic") & (df["confidence"] < min_confidence))
    review_rows = df[needs_review][["campaign_name", "category", "confidence", "source"]]
    mapping = df[~needs_review][["campaign_name", "category", "confidence", "source"]].reset_index(drop=True)

    return {
        "mapping": mapping,
        "review": review_rows.reset_index(drop=True),
        "dropped": dropped.reset_index(drop=True),
        "unmatched_overrides": unmatched_overrides,
    }


def classify_new_only(
    existing_mapping: pd.DataFrame,
    excluded_names: set[str],
    campaigns: pd.DataFrame,
    taxonomy: Taxonomy,
    **kwargs,
) -> dict:
    """Classify only campaign names that are neither mapped nor already excluded.

    `excluded_names` holds generic names dropped earlier, so daily scoring does not
    re-add them (the train/serve skew found in the original system).
    """
    known = set(existing_mapping["campaign_name"]) | excluded_names
    fresh = campaigns[~campaigns["campaign_name"].isin(known)]
    if fresh.empty:
        empty = pd.DataFrame(columns=["campaign_name", "category", "confidence", "source"])
        return {"mapping": empty, "review": empty, "dropped": pd.DataFrame(columns=["campaign_name"]),
                "unmatched_overrides": []}
    res = classify_campaigns(fresh, taxonomy, **kwargs)
    res["unmatched_overrides"] = []  # override keys are checked against the full list at training
    return res


def check_mapping(mapping: pd.DataFrame, categories: list[str]) -> None:
    require_columns(mapping, ["campaign_name", "category"], "campaign mapping")
    if mapping["campaign_name"].duplicated().any():
        dup = mapping.loc[mapping["campaign_name"].duplicated(), "campaign_name"].head(3).tolist()
        raise DataError(f"campaign mapping has duplicate campaign_name values, e.g. {dup}")
    bad = sorted(set(mapping["category"]) - set(categories))
    if bad:
        raise DataError(f"campaign mapping has unknown categories: {bad}")
