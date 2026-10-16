"""The label schema and the redaction policy.

Every source is mapped onto seven span types. A span is either one to redact (it carries a type) or a keep span:
an entity mention the source's annotators decided must stay readable. The policy, per source:

- OpenPII and the Faker documents: every annotated span is redacted, except a stand-alone honorific.
- TAB: DIRECT and QUASI mentions are redacted; NO_MASK mentions are keep spans.
- WNUT-17: person and location mentions are redacted; corporations, products, creative works and groups are not
  personal data and are left as ordinary text.
- WikiNEuRal: encyclopedic prose about public entities; every entity mention is a keep span (evaluation only).
- Wikipedia paragraphs: no spans at all; every token is outside a span (the "Wikipedia negatives").
"""

from __future__ import annotations

TYPES: tuple[str, ...] = ("NAME", "LOC", "ORG", "DATE", "CONTACT", "ID", "QUASI")
CORE: frozenset[str] = frozenset({"NAME", "LOC", "CONTACT", "ID"})

LABELS: tuple[str, ...] = ("O",) + tuple(f"{p}-{t}" for t in TYPES for p in ("B", "I"))
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}
IGNORE = -100

# OpenPII 1M labels. TITLE is merged into an adjacent name (as TAB's PERSON spans include "Mr" and "Ms") and is
# otherwise dropped; see sources.merge_adjacent.
OPENPII: dict[str, str | None] = {
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "TITLE": "TITLE",
    "CITY": "LOC",
    "STREET": "LOC",
    "BUILDINGNUM": "LOC",
    "ZIPCODE": "LOC",
    "EMAIL": "CONTACT",
    "TELEPHONENUM": "CONTACT",
    "DATE": "DATE",
    "AGE": "QUASI",
    "GENDER": "QUASI",
    "SEX": "QUASI",
    "IDCARDNUM": "ID",
    "CREDITCARDNUMBER": "ID",
    "DRIVERLICENSENUM": "ID",
    "TAXNUM": "ID",
    "SOCIALNUM": "ID",
    "PASSPORTNUM": "ID",
}

TAB: dict[str, str] = {
    "PERSON": "NAME",
    "CODE": "ID",
    "LOC": "LOC",
    "ORG": "ORG",
    "DATETIME": "DATE",
    "DEM": "QUASI",
    "QUANTITY": "QUASI",
    "MISC": "QUASI",
}
TAB_REDACT = frozenset({"DIRECT", "QUASI"})
TAB_KEEP = frozenset({"NO_MASK"})

WNUT_TAGS = (
    "O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work", "B-group", "I-group",
    "B-location", "I-location", "B-person", "I-person", "B-product", "I-product",
)
WNUT: dict[str, str | None] = {
    "person": "NAME",
    "location": "LOC",
    "corporation": None,
    "creative-work": None,
    "group": None,
    "product": None,
}

WIKINEURAL_TAGS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")

# The off-the-shelf small model's labels (Presidio-style) mapped onto ours, for the zero-shot baseline only.
OFFSHELF: dict[str, str | None] = {
    "PERSON": "NAME",
    "HONORIFIC": "NAME",
    "LOCATION": "LOC",
    "COORDINATE": "LOC",
    "ORGANIZATION": "ORG",
    "DATE_TIME": "DATE",
    "EMAIL_ADDRESS": "CONTACT",
    "PHONE_NUMBER": "CONTACT",
    "URL": "CONTACT",
    "IP_ADDRESS": "ID",
    "MAC_ADDRESS": "ID",
    "IMEI": "ID",
    "CREDIT_CARD": "ID",
    "IBAN_CODE": "ID",
    "FINANCIAL": "ID",
    "PASSWORD": "ID",
    "US_BANK_NUMBER": "ID",
    "US_DRIVER_LICENSE": "ID",
    "US_ITIN": "ID",
    "US_LICENSE_PLATE": "ID",
    "US_PASSPORT": "ID",
    "US_SSN": "ID",
    "AGE": "QUASI",
    "NRP": "QUASI",
    "TITLE": "QUASI",
}
