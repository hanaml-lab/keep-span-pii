"""Compute every number the report quotes from the committed run outputs.

Reads results/runs/*/ (config, env, metrics, predictions as offsets) and the test split, and writes
results/summary.json:

  configs   per model and condition: mean, standard deviation and seeds of each headline metric (fp32 and int8)
  effects   per model: the effect of K (K - base), of W (W - base) and of both (KW - base), as the mean over the
            seeds both conditions have, with a 95% interval from a document-level bootstrap (1,000 resamples,
            stratified by source, the same resample for both conditions and every seed)
  by_source per model and condition: per-source metrics, averaged over seeds
  tab_types TAB keep-span violation and removal by TAB entity type, pooled over seeds
  int8      per run and per model: int8 minus fp32 for each headline metric
  compute   GPU seconds per run and in total
  offshelf  the off-the-shelf model

    python scripts/aggregate.py
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import metrics as M  # noqa: E402
from keepspan.sources import trim  # noqa: E402
from keepspan.train import load_split  # noqa: E402

RUNS = ROOT / "results" / "runs"
MODELS = ("small", "modernbert", "deberta")
CONDS = ("base", "K", "W", "KW")
KEYS = ("f1_strict_macro", "f1_lenient_macro", "removal_core_pooled", "removal_all_pooled", "tab_removal_all",
        "tab_keep_violation", "wikineural_keep_violation", "wikipedia_pred_per_1k_words")
SOURCES = ("openpii", "faker", "wnut17", "tab", "wikineural", "wikipedia")
SRC_KEYS = ("f1_strict", "f1_lenient", "precision_lenient", "recall_lenient", "removal_core", "removal_all",
            "keep_violation", "pred_per_1k_words", "char_precision")
B = 1000


def load_preds(path: Path) -> dict[str, list]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["spans"]
    return out


def doc_matrix(docs, preds) -> np.ndarray:
    """Per-document count vectors (docs x COUNT_KEYS)."""
    return np.array([[M.doc_counts(d["text"], d["spans"], d["keep"], preds.get(d["id"], []))[k]
                      for k in M.COUNT_KEYS] for d in docs], dtype=np.float64)


def headline_from_counts(by_src: dict[str, np.ndarray]) -> dict:
    """The headline metrics from summed counts per source (same definitions as metrics.summarize)."""
    def as_dict(v):
        return dict(zip(M.COUNT_KEYS, v.tolist()))

    rates = {s: M.rates(as_dict(v)) for s, v in by_src.items()}
    pooled = M.rates(as_dict(sum(by_src[s] for s in M.REDACT_SOURCES)))

    def macro(k):
        vals = [rates[s][k] for s in M.REDACT_SOURCES if rates[s][k] is not None]
        return sum(vals) / len(vals)

    return {
        "f1_strict_macro": macro("f1_strict"), "f1_lenient_macro": macro("f1_lenient"),
        "removal_core_pooled": pooled["removal_core"], "removal_all_pooled": pooled["removal_all"],
        "tab_removal_all": rates["tab"]["removal_all"], "tab_keep_violation": rates["tab"]["keep_violation"],
        "wikineural_keep_violation": rates["wikineural"]["keep_violation"],
        "wikipedia_pred_per_1k_words": rates["wikipedia"]["pred_per_1k_words"],
    }


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None, "n": len(xs),
            "values": xs}


def tab_types(docs, preds_list) -> dict:
    """Keep-span violation and removal per TAB entity type, pooled over the given runs' predictions."""
    keep = defaultdict(lambda: [0, 0])
    red = defaultdict(lambda: [0, 0])
    for preds in preds_list:
        for d in docs:
            if d["source"] != "tab":
                continue
            text = d["text"]
            cov = bytearray(len(text))
            for s, e, _ in preds.get(d["id"], []):
                s, e = trim(text, s, e)
                cov[s:e] = b"\x01" * max(0, e - s)
            for s, e, t in d["keep"]:
                s, e = trim(text, s, e)
                if e <= s:
                    continue
                keep[t][0] += 1
                keep[t][1] += any(cov[i] and text[i].isalnum() for i in range(s, e))
            for s, e, t in d["spans"]:
                s, e = trim(text, s, e)
                al = [i for i in range(s, e) if text[i].isalnum()]
                if not al:
                    continue
                red[t][0] += 1
                red[t][1] += all(cov[i] for i in al)
    return {"keep": {t: {"n": n, "violation": v / n} for t, (n, v) in sorted(keep.items())},
            "redact": {t: {"n": n, "removal": v / n} for t, (n, v) in sorted(red.items())}}


def main() -> int:
    docs = load_split("test")
    src_idx = {s: np.array([i for i, d in enumerate(docs) if d["source"] == s]) for s in SOURCES}
    runs = {}
    mats = {}
    for d in sorted(RUNS.iterdir()):
        if not (d / "metrics_test.json").exists():
            continue
        rec = {"metrics": json.loads((d / "metrics_test.json").read_text())}
        for name in ("config", "env", "onnx"):
            p = d / f"{name}.json"
            if p.exists():
                rec[name] = json.loads(p.read_text())
        for name in ("metrics_test_int8", "metrics_test_onnx_fp32", "metrics_validation"):
            p = d / f"{name}.json"
            if p.exists():
                rec[name] = json.loads(p.read_text())
        runs[d.name] = rec
        mats[d.name] = doc_matrix(docs, load_preds(d / "pred_test.jsonl.gz"))
        print("loaded", d.name, flush=True)

    def parse(name):
        parts = name.split("-")
        return parts[0], parts[1], int(parts[2][1:])

    trained = {n: parse(n) for n in runs if n.split("-")[0] in MODELS}
    summary: dict = {"configs": {}, "effects": {}, "by_source": {}, "tab_types": {}, "int8": {"runs": {}},
                     "compute": {}, "offshelf": None, "test_docs": {s: int(len(v)) for s, v in src_idx.items()}}

    # Configurations: mean and sd over seeds.
    for m in MODELS:
        for c in CONDS:
            names = sorted(n for n, (mm, cc, _) in trained.items() if mm == m and cc == c)
            if not names:
                continue
            key = f"{m}-{c}"
            entry = {"runs": names, "seeds": [trained[n][2] for n in names]}
            for k in KEYS:
                entry[k] = mean_sd([runs[n]["metrics"]["headline"][k] for n in names])
                i8 = [runs[n]["metrics_test_int8"]["headline"][k] for n in names if "metrics_test_int8" in runs[n]]
                entry[f"{k}_int8"] = mean_sd(i8)
            summary["configs"][key] = entry
            summary["by_source"][key] = {
                s: {k: (mean_sd([runs[n]["metrics"]["by_source"][s][k] for n in names]) or {}).get("mean")
                    for k in SRC_KEYS} for s in SOURCES}
            summary["tab_types"][key] = tab_types(docs, [load_preds(RUNS / n / "pred_test.jsonl.gz") for n in names])

    # Effects with a stratified document bootstrap.
    rng = np.random.default_rng(20261016)
    boots = [{s: rng.integers(0, len(ix), len(ix)) for s, ix in src_idx.items()} for _ in range(B)]

    def head(mat, sample=None):
        by = {}
        for s, ix in src_idx.items():
            rows = ix if sample is None else ix[sample[s]]
            by[s] = mat[rows].sum(axis=0)
        return headline_from_counts(by)

    for m in MODELS:
        summary["effects"][m] = {}
        for c in ("K", "W", "KW"):
            seeds = sorted({trained[n][2] for n in trained if trained[n][:2] == (m, c)}
                           & {trained[n][2] for n in trained if trained[n][:2] == (m, "base")})
            if not seeds:
                continue
            pairs = [(f"{m}-base-s{s}", f"{m}-{c}-s{s}") for s in seeds]
            point = {k: statistics.fmean(head(mats[b])[k] - head(mats[a])[k] for a, b in pairs) for k in KEYS}
            reps = {k: [] for k in KEYS}
            for smp in boots:
                diffs = {k: 0.0 for k in KEYS}
                for a, b in pairs:
                    ha, hb = head(mats[a], smp), head(mats[b], smp)
                    for k in KEYS:
                        diffs[k] += (hb[k] - ha[k]) / len(pairs)
                for k in KEYS:
                    reps[k].append(diffs[k])
            summary["effects"][m][c] = {
                "seeds": seeds,
                **{k: {"diff": point[k], "lo": float(np.percentile(reps[k], 2.5)),
                       "hi": float(np.percentile(reps[k], 97.5))} for k in KEYS}}
            print("effect", m, c, seeds, {k: round(point[k], 3) for k in ("tab_keep_violation", "tab_removal_all")},
                  flush=True)

    # int8 minus fp32, per run and per model.
    per_model = defaultdict(lambda: defaultdict(list))
    for n, rec in runs.items():
        if "metrics_test_int8" not in rec:
            continue
        d = {k: rec["metrics_test_int8"]["headline"][k] - rec["metrics"]["headline"][k] for k in KEYS}
        summary["int8"]["runs"][n] = {"delta": d, **{k: rec["onnx"][k] for k in ("fp32_onnx_bytes", "int8_onnx_bytes",
                                                                                 "int8_eval_s") if k in rec.get("onnx", {})}}
        if "metrics_test_onnx_fp32" in rec:
            summary["int8"]["runs"][n]["onnx_fp32_minus_torch"] = {
                k: rec["metrics_test_onnx_fp32"]["headline"][k] - rec["metrics"]["headline"][k] for k in KEYS}
        m = n.split("-")[0]
        for k in KEYS:
            per_model[m][k].append(d[k])
    summary["int8"]["models"] = {m: {k: mean_sd(v) for k, v in ks.items()} for m, ks in per_model.items()}

    # Compute.
    tot = defaultdict(float)
    for n, rec in runs.items():
        if "env" in rec:
            s = rec["env"]["train_seconds"] + rec["env"]["eval_seconds"]
            summary["compute"][n] = {"train_s": rec["env"]["train_seconds"], "eval_s": rec["env"]["eval_seconds"],
                                     "max_mem_allocated_gb": rec["env"]["max_mem_allocated_gb"],
                                     "steps": rec["config"]["steps"], "params": rec["config"]["params"]}
            tot[n.split("-")[0]] += s
    summary["compute_total_gpu_hours"] = {m: v / 3600 for m, v in tot.items()} | {"all": sum(tot.values()) / 3600}

    if "offshelf-small" in runs:
        r = runs["offshelf-small"]["metrics"]
        summary["offshelf"] = {"headline": r["headline"],
                               "by_source": {s: {k: r["by_source"][s][k] for k in SRC_KEYS} for s in SOURCES},
                               "tab_types": tab_types(docs, [load_preds(RUNS / "offshelf-small" / "pred_test.jsonl.gz")])}

    out = ROOT / "results" / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
