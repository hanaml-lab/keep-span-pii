"""Print one line of headline test metrics per finished run (a quick look while the sweep runs)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in sorted((ROOT / "results" / "runs").iterdir()):
    p = d / "metrics_test.json"
    if not p.exists() or (sys.argv[1:] and not any(a in d.name for a in sys.argv[1:])):
        continue
    m = json.loads(p.read_text())
    h, b = m["headline"], m["by_source"]
    env = d / "env.json"
    t = json.loads(env.read_text())["train_seconds"] if env.exists() else 0
    print("%-20s F1s %.3f F1l %.3f core %.3f all %.3f | TAB rem %.3f keepV %.3f | WN %.3f | wiki %5.1f | wnut %.3f"
          " | %4.0fs" % (d.name, h["f1_strict_macro"], h["f1_lenient_macro"], h["removal_core_pooled"],
                         h["removal_all_pooled"], b["tab"]["removal_all"], b["tab"]["keep_violation"],
                         b["wikineural"]["keep_violation"], b["wikipedia"]["pred_per_1k_words"],
                         b["wnut17"]["f1_lenient"], t))
