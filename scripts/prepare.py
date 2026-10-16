"""Build the study's data from the raw downloads.

Reads data/raw/ (see scripts/download.py), maps every source onto the label schema (keepspan.labels), splits by
document hash (keepspan.split), applies the per-source caps below in hash order, and writes:

    data/processed/{train,validation,test}.jsonl.gz    texts and spans (git-ignored: the texts are the sources')
    data/splits/{train,validation,test}.tsv             id, source, sha256 of every document used (committed)
    data/splits/stats.json                              counts per source and split (committed)

    python scripts/prepare.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import sources as S  # noqa: E402
from keepspan.split import TEST, TRAIN, VALIDATION, doc_hash, split_of  # noqa: E402
from keepspan.synth import generate  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

# Per-source caps (documents), applied after the hash split, keeping the documents with the smallest hashes.
# None: no cap. 0: the source is not used in that split.
CAPS = {
    "openpii": {TRAIN: 16000, VALIDATION: 800, TEST: 1000},
    "faker": {TRAIN: None, VALIDATION: None, TEST: None},
    "wnut17": {TRAIN: None, VALIDATION: None, TEST: None},
    "tab": {TRAIN: None, VALIDATION: None, TEST: None},
    "wikineural": {TRAIN: 0, VALIDATION: 800, TEST: 2000},      # evaluation only (CC BY-NC-SA)
    "wikipedia": {TRAIN: 6000, VALIDATION: 400, TEST: 1000},    # the Wikipedia negatives
}
N_FAKER = 6000
WIKI_CANDIDATES = 60000  # paragraphs read from the shard before splitting


def validate(rec: dict) -> None:
    n = len(rec["text"])
    for s, e, _ in rec["spans"] + rec["keep"]:
        assert 0 <= s < e <= n, (rec["id"], s, e, n)


def main() -> int:
    recs: list[dict] = []
    recs += list(S.load_openpii(RAW / "openpii" / "data" / "validation.jsonl"))
    recs += generate(N_FAKER)
    recs += list(S.load_wnut(sorted((RAW / "wnut17" / "wnut_17").glob("*/0000.parquet"))))
    recs += list(S.load_tab(sorted((RAW / "tab").glob("echr_*.json"))))
    wn = list(S.load_wikineural(sorted((RAW / "wikineural" / "data").glob("*_en-*.parquet"))))
    recs += wn
    wn_shingles: set = set()
    for r in wn:
        wn_shingles |= S.shingles(r["text"])
    wiki = []
    for r in S.load_wikipedia(RAW / "wikipedia" / "20231101.en" / "train-00007-of-00041.parquet", wn_shingles):
        wiki.append(r)
        if len(wiki) >= WIKI_CANDIDATES:
            break
    recs += wiki

    # Hash, deduplicate exact (normalised) duplicates, split.
    seen: dict[str, str] = {}
    dupes = Counter()
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        validate(r)
        h = doc_hash(r["text"])
        if h in seen:
            dupes[r["source"]] += 1
            continue
        seen[h] = r["id"]
        r["sha256"] = h
        r["split"] = split_of(h)
        buckets[(r["source"], r["split"])].append(r)

    out: dict[str, list[dict]] = {TRAIN: [], VALIDATION: [], TEST: []}
    stats: dict = {"duplicates_dropped": dict(dupes), "sources": {}}
    for (src, split), rs in sorted(buckets.items()):
        cap = CAPS[src][split]
        rs = sorted(rs, key=lambda r: r["sha256"])
        if cap is not None:
            rs = rs[:cap]
        out[split] += rs
        st = stats["sources"].setdefault(src, {})
        st[split] = {
            "docs": len(rs),
            "chars": sum(len(r["text"]) for r in rs),
            "spans": sum(len(r["spans"]) for r in rs),
            "keep": sum(len(r["keep"]) for r in rs),
            "span_types": dict(Counter(s[2] for r in rs for s in r["spans"])),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for split, rs in out.items():
        rs.sort(key=lambda r: (r["source"], r["sha256"]))
        with gzip.open(OUT / f"{split}.jsonl.gz", "wt") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (SPLITS / f"{split}.tsv").open("w") as f:
            f.write("id\tsource\tsha256\n")
            for r in rs:
                f.write(f"{r['id']}\t{r['source']}\t{r['sha256']}\n")
    (SPLITS / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for src, st in stats["sources"].items():
        print(src, {k: (v["docs"], v["spans"], v["keep"]) for k, v in st.items()})
    print("duplicates dropped:", dict(dupes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
