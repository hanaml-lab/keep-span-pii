"""Check the committed run outputs against results/summary.json, with no data download.

- every trained run has its config, training log, environment, metrics and test predictions;
- each run's config matches its name (model, negatives, seed) and records the base model's revision;
- every configuration mean and standard deviation in results/summary.json equals the one recomputed from the runs'
  metrics files, and every int8 difference equals int8 minus fp32 from those files.

Recomputing the metrics themselves from the predictions needs the test split (scripts/download.py and
scripts/prepare.py, then scripts/aggregate.py).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
NEEDED = ("config.json", "train_log.jsonl", "env.json", "metrics_test.json", "metrics_validation.json",
          "pred_test.jsonl.gz")


def main() -> int:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    errors = []
    metrics = {}
    for d in sorted(RUNS.iterdir()):
        if d.name.startswith("offshelf"):
            continue
        for f in NEEDED:
            if not (d / f).exists():
                errors.append(f"{d.name}: missing {f}")
        cfg = json.loads((d / "config.json").read_text())
        model, cond, seed = d.name.split("-")
        if (cfg["model"], cfg["seed"]) != (model, int(seed[1:])):
            errors.append(f"{d.name}: config names {cfg['model']} seed {cfg['seed']}")
        if (cfg["keep_negatives"], cfg["wikipedia_negatives"]) != ("K" in cond, "W" in cond):
            errors.append(f"{d.name}: config conditions do not match the name")
        if not cfg.get("base_revision"):
            errors.append(f"{d.name}: no base revision")
        metrics[d.name] = json.loads((d / "metrics_test.json").read_text())["headline"]
        p8 = d / "metrics_test_int8.json"
        if p8.exists():
            h8 = json.loads(p8.read_text())["headline"]
            want = summary["int8"]["runs"].get(d.name, {}).get("delta", {})
            for k, v in want.items():
                if abs(v - (h8[k] - metrics[d.name][k])) > 1e-9:
                    errors.append(f"{d.name}: int8 delta {k} differs")
    for key, entry in summary["configs"].items():
        for k, v in entry.items():
            if not isinstance(v, dict) or "values" not in v or k.endswith("_int8"):
                continue
            vals = [metrics[r][k] for r in entry["runs"]]
            if abs(statistics.fmean(vals) - v["mean"]) > 1e-9:
                errors.append(f"{key}: mean of {k} differs from the runs")
            if len(vals) > 1 and abs(statistics.stdev(vals) - v["sd"]) > 1e-9:
                errors.append(f"{key}: sd of {k} differs from the runs")
    for e in errors:
        print("ERROR", e)
    print(f"checked {len(metrics)} runs and {len(summary['configs'])} configurations: "
          f"{'ok' if not errors else f'{len(errors)} errors'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
