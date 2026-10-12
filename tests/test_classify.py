import pandas as pd
import pytest

from ccp import classify as C
from ccp.utils import ConfigError, DataError, normalize_name


def test_keyword_precedence_first_rule_wins(taxonomy):
    # contains both a Forex and a Travel keyword; Forex is listed first
    assert C.keyword_classify("Forex hotel combo", taxonomy) == "Forex & Remittance"


def test_keyword_is_case_and_whitespace_insensitive(taxonomy):
    assert C.keyword_classify("  HOTEL   Sale ", taxonomy) == "Travel"


def test_exclude_falls_through_to_later_rules(taxonomy):
    # 'dormant' matches Reactivation but 'active users' excludes it; 'hotel' then matches Travel
    assert C.keyword_classify("dormant active users hotel", taxonomy) == "Travel"
    assert C.keyword_classify("dormant active users", taxonomy) is None


def test_no_match_returns_none(taxonomy):
    assert C.keyword_classify("something unrelated", taxonomy) is None


def test_generic_names_are_detected(taxonomy):
    assert C.is_generic("PN1", taxonomy)
    assert C.is_generic(" email 2 ", taxonomy)
    assert not C.is_generic("PN Travel Sale", taxonomy)


def test_generics_dropped_before_classification(taxonomy):
    campaigns = pd.DataFrame({"campaign_name": ["PN1", "Newsletter May", "Mystery"]})
    res = C.classify_campaigns(campaigns, taxonomy)
    assert res["dropped"]["campaign_name"].tolist() == ["PN1"]
    assert res["mapping"]["campaign_name"].tolist() == ["Newsletter May"]
    assert res["review"]["campaign_name"].tolist() == ["Mystery"]


def test_each_campaign_gets_at_most_one_row(taxonomy):
    campaigns = pd.DataFrame({"campaign_name": ["Hotel A", "Hotel A", "Newsletter"]})
    res = C.classify_campaigns(campaigns, taxonomy)
    assert res["mapping"]["campaign_name"].is_unique


def test_override_matches_after_normalisation_and_reports_misses(taxonomy):
    campaigns = pd.DataFrame({"campaign_name": ["Mystery Offer!"]})
    overrides = {normalize_name("mystery   OFFER!"): "Travel", normalize_name("never seen"): "Referral"}
    res = C.classify_campaigns(campaigns, taxonomy, overrides)
    assert res["mapping"].loc[0, "category"] == "Travel"
    assert res["mapping"].loc[0, "source"] == "override"
    assert res["unmatched_overrides"] == ["never seen"]


def test_override_with_unknown_category_is_rejected(taxonomy):
    campaigns = pd.DataFrame({"campaign_name": ["x"]})
    with pytest.raises(ConfigError):
        C.classify_campaigns(campaigns, taxonomy, {"x": "Not A Category"})


def test_classify_new_only_skips_known_and_excluded_names(taxonomy):
    existing = pd.DataFrame({"campaign_name": ["Hotel A"], "category": ["Travel"]})
    campaigns = pd.DataFrame({"campaign_name": ["Hotel A", "PN1", "Newsletter B"]})
    res = C.classify_new_only(existing, {"PN1"}, campaigns, taxonomy)
    assert res["mapping"]["campaign_name"].tolist() == ["Newsletter B"]


def test_check_mapping_rejects_duplicates_and_unknown_categories():
    cats = ["Travel"]
    with pytest.raises(DataError):
        C.check_mapping(pd.DataFrame({"campaign_name": ["a", "a"], "category": ["Travel", "Travel"]}), cats)
    with pytest.raises(DataError):
        C.check_mapping(pd.DataFrame({"campaign_name": ["a"], "category": ["Nope"]}), cats)
