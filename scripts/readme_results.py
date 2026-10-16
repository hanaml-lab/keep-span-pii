"""Write the measured results into README.md, between the results markers, from results/summary.json.

    python scripts/readme_results.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
MODELS = (("small", "bert-small (28.5M)"), ("modernbert", "ModernBERT-base (150M)"), ("deberta", "DeBERTa-v3-base (184M)"))
CONDS = (("base", "none"), ("K", "K"), ("W", "W"), ("KW", "K+W"))
COLS = (("f1_strict_macro", "F1 strict"), ("removal_core_pooled", "core removed"), ("tab_removal_all", "TAB removed"),
        ("tab_keep_violation", "TAB keep redacted ↓"), ("wikineural_keep_violation", "WikiNEuRal redacted ↓"))


def pct(v, signed=False):
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


def block() -> str:
    out = ["| Model | Negatives | " + " | ".join(n for _, n in COLS) + " |",
           "|---|---|" + "---:|" * len(COLS)]
    if S.get("offshelf"):
        h = S["offshelf"]["headline"]
        out.append("| bert-small, off the shelf | none | " + " | ".join(pct(h[k]) for k, _ in COLS) + " |")
    for m, name in MODELS:
        for c, cname in CONDS:
            cfg = S["configs"].get(f"{m}-{c}")
            if not cfg:
                continue
            cells = []
            for k, _ in COLS:
                e = cfg[k]
                cells.append(f"{pct(e['mean'])} ± {pct(e['sd'])}" if e["sd"] is not None else pct(e["mean"]))
            out.append(f"| {name} | {cname} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Percent on the test split, mean ± standard deviation over seeds (seeds per row: "
               + ", ".join(sorted({str(len(c['seeds'])) for c in S['configs'].values()})) + "). F1 is the macro mean "
               "over OpenPII, Faker, WNUT-17 and TAB; core removal (names, places, contacts, identifiers fully "
               "removed) is pooled over them. K: keep-span negatives; W: Wikipedia negatives.")
    out.append("")
    out.append("Effect of keep-span negatives (K minus none, percentage points, 95% bootstrap interval):")
    out.append("")
    out.append("| Model | TAB keep redacted | TAB removed | core removed | F1 strict |")
    out.append("|---|---:|---:|---:|---:|")
    for m, name in MODELS:
        e = S["effects"].get(m, {}).get("K")
        if not e:
            continue
        out.append(f"| {name} | " + " | ".join(
            f"{pct(e[k]['diff'], True)} [{pct(e[k]['lo'], True)}, {pct(e[k]['hi'], True)}]"
            for k in ("tab_keep_violation", "tab_removal_all", "removal_core_pooled", "f1_strict_macro")) + " |")
    if S["int8"].get("models"):
        out.append("")
        out.append("int8 (onnxruntime dynamic quantisation) minus fp32, percentage points, mean over the runs evaluated:")
        out.append("")
        out.append("| Model | runs | F1 strict | core removed | TAB keep redacted |")
        out.append("|---|---:|---:|---:|---:|")
        for m, name in MODELS:
            d = S["int8"]["models"].get(m)
            if d:
                out.append(f"| {name} | {d['f1_strict_macro']['n']} | " + " | ".join(
                    f"{100 * d[k]['mean']:+.2f}" for k in ("f1_strict_macro", "removal_core_pooled",
                                                            "tab_keep_violation")) + " |")
    mix = S.get("int8_recipe", {}).get("fp32-mlp-out", {})
    if mix:
        out.append("")
        out.append("The base models with their MLP output projections kept in fp32 (int8 elsewhere), minus fp32:")
        out.append("")
        out.append("| Model | runs | size (MB) | F1 strict | core removed | TAB keep redacted |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for m, name in MODELS:
            rs = [r for n, r in mix.items() if n.startswith(m + "-")]
            if rs:
                out.append(f"| {name} | {len(rs)} | {rs[0]['int8_onnx_bytes'] / 1e6:.0f} | " + " | ".join(
                    f"{statistics.fmean(100 * r['delta'][k] for r in rs):+.2f}"
                    for k in ("f1_strict_macro", "removal_core_pooled", "tab_keep_violation")) + " |")
    lat = ROOT / "results" / "latency_browser.json"
    if lat.exists():
        W = json.loads(lat.read_text())
        out.append("")
        out.append(f"Browser timing ({W['browser']}, onnxruntime-web {W['onnxruntime_web']}, WebAssembly; median ms per "
                   "forward pass, batch 1):")
        out.append("")
        out.append("| Model | threads | 32 tokens | 128 tokens | 512 tokens | session ready (s) |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for r in W["results"]:
            if r.get("error"):
                continue
            ls = {x["length"]: x["median_ms"] for x in r["lengths"]}
            out.append(f"| {r['model']} | {r['threads']} | " + " | ".join(f"{ls.get(n, float('nan')):.1f}"
                                                                         for n in (32, 128, 512))
                       + f" | {r['create_ms'] / 1000:.1f} |")
    gh = S["compute_total_gpu_hours"]
    out.append("")
    out.append(f"Compute: {len([n for n in S['compute'] if not n.startswith('offshelf')])} fine-tuning runs, "
               f"{gh['all']:.1f} GPU-hours on one RTX 4070 Ti SUPER, including each run's evaluation "
               f"(mean minutes per run: " + ", ".join(
                   f"{name.split(' (')[0]} {statistics.fmean(v['train_s'] for n, v in S['compute'].items() if n.startswith(m + '-')) / 60:.0f}"
                   for m, name in MODELS if any(n.startswith(m + '-') for n in S['compute'])) + ").")
    return "\n".join(out)


def main() -> int:
    p = ROOT / "README.md"
    text = p.read_text()
    new = re.sub(r"(<!-- results:start -->\n).*?(<!-- results:end -->)",
                 lambda m: m.group(1) + block() + "\n" + m.group(2), text, flags=re.S)
    p.write_text(new)
    print(block())
    return 0


if __name__ == "__main__":
    sys.exit(main())
