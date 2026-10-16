import numpy as np
import pytest

from keepspan.encode import decode, encode_doc, merge_window_logits, window_starts, windows
from keepspan.labels import ID2LABEL, IGNORE, LABEL2ID
from keepspan.metrics import doc_counts, rates, summarize

DOC = {
    "id": "d1",
    "source": "tab",
    "text": "The applicant, Anna Berg, sued the Ministry of Justice on 3 May 2004.",
    "spans": [[15, 24, "NAME"], [58, 68, "DATE"]],
    "keep": [[35, 54, "ORG"]],
}


@pytest.fixture(scope="module")
def tok():
    from keepspan.smoke import tiny_tokenizer

    return tiny_tokenizer([DOC, {"text": "padding words for the vocabulary"}])


def test_labels_and_keep_mask(tok):
    de = encode_doc(tok, DOC)
    words = [DOC["text"][s:e] for s, e in de.offsets]
    lab = {w: ID2LABEL.get(x, x) for w, x in zip(words, de.labels)}
    assert lab["Anna"] == "B-NAME" and lab["Berg"] == "I-NAME"
    assert lab["3"] == "B-DATE" and lab["2004"] == "I-DATE"
    assert lab["Ministry"] == "O" and lab["The"] == "O"
    keep_words = {w for w, k in zip(words, de.keep_mask) if k}
    assert keep_words == {"Ministry", "of", "Justice"}


def test_keep_negatives_switch(tok):
    de = encode_doc(tok, DOC)
    with_k = windows(de, 64, 8, tok.cls_token_id, tok.sep_token_id, keep_negatives=True)[0]["labels"]
    without = windows(de, 64, 8, tok.cls_token_id, tok.sep_token_id, keep_negatives=False)[0]["labels"]
    diff = [i for i, (a, b) in enumerate(zip(with_k, without)) if a != b]
    assert len(diff) == 3
    assert all(with_k[i] == LABEL2ID["O"] and without[i] == IGNORE for i in diff)
    assert with_k[0] == IGNORE and with_k[-1] == IGNORE


def test_decode_round_trip(tok):
    de = encode_doc(tok, DOC)
    gold = [x if x != IGNORE else 0 for x in de.labels]
    spans = decode(DOC["text"], de.offsets, gold)
    assert [DOC["text"][s:e] for s, e, _ in spans] == ["Anna Berg", "3 May 2004"]


def test_windows_cover_every_token():
    for n in (1, 50, 510, 511, 1000, 2049):
        starts = window_starts(n, 510, 128)
        covered = set()
        for s in starts:
            covered |= set(range(s, min(n, s + 510)))
        assert covered == set(range(n))
    lg = merge_window_logits(4, [(0, np.ones((3, 2))), (1, 3 * np.ones((3, 2)))])
    assert lg[:, 0].tolist() == [1.0, 2.0, 2.0, 3.0]


def test_metrics_definitions():
    text = DOC["text"]
    # Perfect: strict and lenient 1, everything removed, no keep span touched.
    c = doc_counts(text, DOC["spans"], DOC["keep"], DOC["spans"])
    r = rates(c)
    assert r["f1_strict"] == 1.0 and r["removal_all"] == 1.0 and r["keep_violation"] == 0.0
    # Partial name, a wrong type on the date, and a redaction inside the keep span.
    pred = [[15, 19, "NAME"], [58, 68, "ID"], [35, 43, "ORG"]]
    r = rates(doc_counts(text, DOC["spans"], DOC["keep"], pred))
    assert r["precision_strict"] == 0.0
    assert r["recall_lenient"] == 1.0 and r["precision_lenient"] == pytest.approx(2 / 3)
    assert r["removal_all"] == 0.5          # "Berg" stays readable
    assert r["removal_core"] == 0.0         # the name is the only core span
    assert r["keep_violation"] == 1.0


def test_summarize_pools_and_macro_averages():
    docs = [DOC, {"id": "w", "source": "wikineural", "text": "Paris is in France.", "spans": [],
                  "keep": [[0, 5, "LOC"], [12, 18, "LOC"]]}]
    s = summarize(docs, {"d1": DOC["spans"], "w": [[0, 5, "LOC"]]})
    assert s["by_source"]["wikineural"]["keep_violation"] == 0.5
    assert s["headline"]["tab_keep_violation"] == 0.0
    assert s["macro"]["f1_strict"] == 1.0
