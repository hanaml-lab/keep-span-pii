"""Download every public source at a pinned revision and record what was fetched.

Each file is fetched from a revision-pinned URL, so a later run reads the same bytes. The script writes
data/sources/manifest.json with the URL, revision, licence, byte size and SHA-256 of every file. Raw files go to
data/raw/ (git-ignored). Run from the repository root:

    python scripts/download.py            # all sources
    python scripts/download.py --only tab wnut17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "sources" / "manifest.json"

HF = "https://huggingface.co"

# Every source, at the revision this study used. `files` are paths inside the repository at that revision.
SOURCES: dict[str, dict] = {
    "openpii": {
        "name": "Ai4Privacy OpenPII 1M (English records only)",
        "repo": "ai4privacy/pii-masking-openpii-1m",
        "kind": "dataset",
        "revision": "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0",
        "licence": "CC BY 4.0",
        "page": "https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m",
        # The release's validation file holds about 180k records in 23 languages; its English records are more
        # than this study needs, so the 3.7 GB train file is not downloaded.
        "files": ["data/validation.jsonl", "README.md"],
    },
    "tab": {
        "name": "Text Anonymization Benchmark (TAB), ECHR corpus",
        "repo": "ildpil/text-anonymization-benchmark",
        "kind": "dataset",
        "revision": "1f22ce098996054689b5b331fb20b3a74a78e1c5",
        "licence": "MIT",
        "page": "https://github.com/NorskRegnesentral/text-anonymization-benchmark",
        "files": ["echr_train.zip", "echr_dev.zip", "echr_test.zip", "LICENSE.txt"],
    },
    "wnut17": {
        "name": "WNUT-17 emerging and rare entities",
        "repo": "leondz/wnut_17",
        "kind": "dataset",
        # The dataset is a loading script; the Hub's own Parquet conversion of it is pinned here.
        "revision": "refs%2Fconvert%2Fparquet",
        "revision_note": "Parquet conversion of revision be8fbd57399293da6f9aad56a793bce5bf1fd30e",
        "licence": "CC BY 4.0",
        "page": "https://noisy-text.github.io/2017/emerging-rare-entities.html",
        "files": ["wnut_17/train/0000.parquet", "wnut_17/validation/0000.parquet", "wnut_17/test/0000.parquet"],
    },
    "wikineural": {
        "name": "WikiNEuRal (English)",
        "repo": "Babelscape/wikineural",
        "kind": "dataset",
        "revision": "74c9b9dca034bb1606a6769457983904bd976803",
        "licence": "CC BY-NC-SA 4.0 (used for evaluation only)",
        "page": "https://huggingface.co/datasets/Babelscape/wikineural",
        "files": [
            "data/train_en-00000-of-00001.parquet",
            "data/val_en-00000-of-00001.parquet",
            "data/test_en-00000-of-00001.parquet",
        ],
    },
    "wikipedia": {
        "name": "Wikipedia (English, 2023-11-01 dump), one shard",
        "repo": "wikimedia/wikipedia",
        "kind": "dataset",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "licence": "CC BY-SA 3.0 and GFDL",
        "page": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "files": ["20231101.en/train-00007-of-00041.parquet"],
    },
}


def url_for(src: dict, path: str) -> str:
    prefix = "datasets/" if src["kind"] == "dataset" else ""
    return f"{HF}/{prefix}{src['repo']}/resolve/{src['revision']}/{path}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "keep-span-pii/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for key, src in SOURCES.items():
        if args.only and key not in args.only:
            continue
        files = []
        for path in src["files"]:
            dest = RAW / key / path
            url = url_for(src, path)
            if not dest.exists():
                t0 = time.time()
                print(f"[{key}] {url}", flush=True)
                fetch(url, dest)
                print(f"[{key}]   {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
            files.append({"path": path, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)})
        manifest[key] = {k: v for k, v in src.items() if k != "files"} | {"files": files}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
