from keepspan.labels import CORE, LABEL2ID, LABELS, TYPES
from keepspan.sources import bio_to_spans, detokenize, merge_adjacent, trim
from keepspan.split import TEST, TRAIN, VALIDATION, doc_hash, split_of
from keepspan.synth import TEMPLATES, generate


def test_label_schema():
    assert LABELS[0] == "O"
    assert len(LABELS) == 1 + 2 * len(TYPES) == 15
    assert CORE <= set(TYPES)
    assert LABEL2ID["B-NAME"] != LABEL2ID["I-NAME"]


def test_detokenize_offsets_point_at_tokens():
    tokens = ["The", "Court", "(", "Grand", "Chamber", ")", "said", ":", '"', "no", '"', ",", "it", "'s", "late", "."]
    text, offs = detokenize(tokens)
    assert text == 'The Court (Grand Chamber) said: "no", it\'s late.'
    for tok, (s, e) in zip(tokens, offs):
        assert text[s:e] == tok


def test_bio_to_spans_starts_a_span_on_a_stray_inside_tag():
    tags = ["O", "B-PER", "I-PER", "O", "I-LOC", "I-LOC", "B-LOC", "O"]
    assert bio_to_spans(tags) == [(1, 3, "PER"), (4, 6, "LOC"), (6, 7, "LOC")]


def test_merge_adjacent_joins_names_and_titles():
    text = "Dear Mr John Smith, see 12 Main Street, Springfield."
    spans = [[5, 7, "TITLE"], [8, 12, "NAME"], [13, 18, "NAME"], [24, 26, "LOC"], [27, 38, "LOC"], [40, 51, "LOC"]]
    out = merge_adjacent(text, spans)
    assert [text[s:e] for s, e, _ in out] == ["Mr John Smith", "12 Main Street", "Springfield"]
    assert merge_adjacent("Dr. Who?", [[0, 3, "TITLE"]]) == []


def test_trim_strips_edges():
    text = ' ("Jane Doe"), '
    assert text[slice(*trim(text, 0, len(text)))] == "Jane Doe"


def test_split_is_by_content():
    assert doc_hash("Hello  World") == doc_hash("hello world")
    assert split_of(doc_hash("a")) == split_of(doc_hash("A"))
    counts = {TRAIN: 0, VALIDATION: 0, TEST: 0}
    for i in range(4000):
        counts[split_of(doc_hash(f"document {i}"))] += 1
    assert 0.76 < counts[TRAIN] / 4000 < 0.84
    assert 0.07 < counts[TEST] / 4000 < 0.13


def test_synth_spans_cover_the_filled_values():
    docs = generate(200, seed=1)
    assert docs == generate(200, seed=1)
    for d in docs:
        for s, e, t in d["spans"]:
            assert 0 <= s < e <= len(d["text"])
            assert d["text"][s:e].strip() == d["text"][s:e]
            assert t in TYPES
    assert len({d["text"] for d in docs}) > 150
    assert len(TEMPLATES) >= 30
