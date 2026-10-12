import re
import pytest

from ccp.classify import Rule, Taxonomy

CATS = ["Referral", "Travel", "Forex & Remittance", "Reactivation", "Educational Content"]


@pytest.fixture
def taxonomy():
    return Taxonomy(
        categories=CATS,
        rules=[
            Rule("Forex & Remittance", ["forex", "currency"]),
            Rule("Referral", ["referral"]),
            Rule("Reactivation", ["reactivation", "dormant"], exclude=["active users"]),
            Rule("Travel", ["hotel", "flight"]),
            Rule("Educational Content", ["newsletter"]),
        ],
        generic_patterns=[re.compile(r"^pn\s*\d*$"), re.compile(r"^email\s*\d*$")],
        descriptions={},
    )
