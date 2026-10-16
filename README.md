# keep-span-pii: learning what to keep in on-device PII span detection

[![ci](https://github.com/hanaml-lab/keep-span-pii/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/keep-span-pii/actions/workflows/ci.yml)

Code, configurations, logs and predictions for *Learning What to Keep: Keep-Span Negatives and Over-Redaction in On-Device PII Span Detection* (Agarwal, hanaML technical report, 2026). The report is at [hanaml.com/research/papers/keep-span-pii](https://hanaml.com/research/papers/keep-span-pii).

A model that redacts personal data on a device before text reaches a cloud model can fail in two directions: it can leave personal data readable, or it can remove the courts, places and dates the cloud model needs. Most PII training data mark only what to remove. The Text Anonymization Benchmark (TAB) also marks the entity mentions its annotators decided to keep. This repository asks, on public data only, whether training on those keep spans as negatives reduces over-redaction without losing recall, across three model sizes, and what int8 quantisation for the browser costs.

**Every number here is measured**: fine-tuning runs on one NVIDIA RTX 4070 Ti SUPER, three seeds per configuration, with each run's configuration, training log, metrics, test predictions (as character offsets), run time and memory committed under `results/runs/`. Nothing is simulated.

## Results

<!-- results:start -->
| Model | Negatives | F1 strict | core removed | TAB removed | TAB keep redacted ↓ | WikiNEuRal redacted ↓ |
|---|---|---:|---:|---:|---:|---:|
| bert-small, off the shelf | none | 57.0 | 80.9 | 71.9 | 58.7 | 67.9 |
| bert-small (28.5M) | none | 78.4 ± 0.4 | 97.6 ± 0.1 | 92.4 ± 0.2 | 70.1 ± 1.4 | 65.1 ± 0.9 |
| bert-small (28.5M) | K | 79.6 ± 0.3 | 96.7 ± 0.0 | 86.3 ± 0.4 | 28.5 ± 1.6 | 62.2 ± 1.0 |
| bert-small (28.5M) | W | 78.7 ± 0.2 | 97.6 ± 0.1 | 92.0 ± 0.1 | 69.6 ± 1.2 | 24.4 ± 0.7 |
| bert-small (28.5M) | K+W | 80.2 ± 0.3 | 96.6 ± 0.2 | 85.6 ± 0.5 | 27.9 ± 0.9 | 23.5 ± 1.1 |
| ModernBERT-base (150M) | none | 80.0 ± 0.3 | 97.4 ± 0.2 | 91.7 ± 0.4 | 69.1 ± 3.0 | 57.6 ± 2.2 |
| ModernBERT-base (150M) | K | 80.4 ± 0.4 | 95.5 ± 0.1 | 83.3 ± 0.1 | 23.6 ± 0.4 | 53.6 ± 1.5 |
| ModernBERT-base (150M) | W | 78.9 ± 0.5 | 97.5 ± 0.1 | 91.8 ± 0.2 | 68.5 ± 0.9 | 22.8 ± 2.1 |
| ModernBERT-base (150M) | K+W | 80.0 ± 0.1 | 95.7 ± 0.2 | 83.9 ± 0.2 | 24.3 ± 0.6 | 23.1 ± 2.6 |
| DeBERTa-v3-base (184M) | none | 82.5 ± 0.5 | 98.7 ± 0.1 | 93.8 ± 0.2 | 71.2 ± 0.7 | 58.4 ± 2.2 |
| DeBERTa-v3-base (184M) | K | 83.7 ± 0.6 | 97.5 ± 0.1 | 87.3 ± 0.6 | 24.5 ± 1.3 | 54.0 ± 2.5 |
| DeBERTa-v3-base (184M) | W | 82.4 ± 0.5 | 98.6 ± 0.0 | 93.9 ± 0.2 | 72.0 ± 0.9 | 7.7 ± 1.1 |
| DeBERTa-v3-base (184M) | K+W | 83.6 ± 0.2 | 97.3 ± 0.1 | 87.2 ± 0.3 | 24.3 ± 1.5 | 6.8 ± 0.4 |

Percent on the test split, mean ± standard deviation over seeds (seeds per row: 3). F1 is the macro mean over OpenPII, Faker, WNUT-17 and TAB; core removal (names, places, contacts, identifiers fully removed) is pooled over them. K: keep-span negatives; W: Wikipedia negatives.

Effect of keep-span negatives (K minus none, percentage points, 95% bootstrap interval):

| Model | TAB keep redacted | TAB removed | core removed | F1 strict |
|---|---:|---:|---:|---:|
| bert-small (28.5M) | -41.6 [-45.3, -38.1] | -6.1 [-7.4, -5.0] | -0.9 [-1.4, -0.6] | +1.2 [+0.7, +1.6] |
| ModernBERT-base (150M) | -45.5 [-49.3, -41.7] | -8.4 [-9.9, -7.1] | -1.9 [-2.6, -1.3] | +0.4 [-0.5, +1.1] |
| DeBERTa-v3-base (184M) | -46.7 [-50.9, -42.8] | -6.4 [-7.7, -5.2] | -1.1 [-1.6, -0.8] | +1.3 [+0.8, +1.8] |

int8 (onnxruntime dynamic quantisation) minus fp32, percentage points, mean over the runs evaluated:

| Model | runs | F1 strict | core removed | TAB keep redacted |
|---|---:|---:|---:|---:|
| bert-small (28.5M) | 12 | -0.22 | -0.02 | -0.17 |
| ModernBERT-base (150M) | 12 | -39.30 | -32.94 | -22.83 |
| DeBERTa-v3-base (184M) | 12 | -82.99 | -87.72 | -46.64 |

The base models with their MLP output projections kept in fp32 (int8 elsewhere), minus fp32:

| Model | runs | size (MB) | F1 strict | core removed | TAB keep redacted |
|---|---:|---:|---:|---:|---:|
| ModernBERT-base (150M) | 4 | 209 | -1.02 | -0.09 | +0.40 |
| DeBERTa-v3-base (184M) | 4 | 328 | -0.94 | -0.32 | +0.63 |

Browser timing (chromium 149.0.7827.55, onnxruntime-web 1.30.0, WebAssembly; median ms per forward pass, batch 1):

| Model | threads | 32 tokens | 128 tokens | 512 tokens | session ready (s) |
|---|---:|---:|---:|---:|---:|
| small-int8 | 1 | 39.3 | 158.9 | 821.6 | 0.9 |
| small-int8 | 4 | 22.4 | 72.7 | 205.2 | 0.8 |
| small-fp32 | 1 | 67.8 | 267.3 | 1209.5 | 1.2 |
| small-fp32 | 4 | 32.1 | 69.6 | 304.2 | 1.1 |
| modernbert-int8-mix | 1 | 362.8 | 1426.2 | 7010.9 | 2.5 |
| modernbert-int8-mix | 4 | 114.6 | 390.2 | 1829.9 | 2.6 |
| deberta-int8-mix | 1 | 373.2 | 1423.8 | 6742.9 | 4.3 |
| deberta-int8-mix | 4 | 114.6 | 384.0 | 1848.4 | 4.3 |

Compute: 36 fine-tuning runs, 7.8 GPU-hours on one RTX 4070 Ti SUPER, including each run's evaluation (mean minutes per run: bert-small 2, ModernBERT-base 14, DeBERTa-v3-base 22).
<!-- results:end -->

`results/summary.json` holds every number above and in the report, computed by `scripts/aggregate.py` from the per-run files.

## What is in the study

- **Data** (`scripts/download.py`, `scripts/prepare.py`, `src/keepspan/sources.py`): five public sources at pinned Hugging Face revisions, with each file's URL, size and SHA-256 recorded in `data/sources/manifest.json`, plus synthetic documents from hand-written templates filled by Faker (`src/keepspan/synth.py`).

  | Source | Licence | Role |
  |---|---|---|
  | [OpenPII 1M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m), English records | CC BY 4.0 | every span redacted |
  | [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | MIT | direct and quasi-identifiers redacted; `NO_MASK` mentions are keep spans |
  | [WNUT-17](https://noisy-text.github.io/2017/emerging-rare-entities.html) | CC BY 4.0 | persons and locations redacted |
  | [WikiNEuRal](https://huggingface.co/datasets/Babelscape/wikineural), English | CC BY-NC-SA 4.0 | evaluation only: every entity is a keep span |
  | [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), one 2023-11-01 shard | CC BY-SA 3.0 | the Wikipedia negatives: no spans |
  | Faker documents (ours) | MIT tooling | every filled value redacted |

  The split is fixed by a SHA-256 hash of each document's normalised text (80/10/10), so identical documents always share a split; the document list of each split is committed in `data/splits/`.
- **Labels** (`src/keepspan/labels.py`): seven span types (name, location, organisation, date, contact, identifier, other quasi-identifier) as fifteen BIO labels; name, location, contact and identifier are the core types.
- **Models** (`src/keepspan/models.py`): [bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) (28.5M parameters, Apache-2.0), [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (Apache-2.0) and [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) (MIT), each with a fresh token-classification head.
- **Conditions** (`src/keepspan/train.py`): keep-span negatives (`--keep 1`: keep-span tokens labelled outside a span; `--keep 0`: the same tokens left out of the loss) crossed with Wikipedia negatives (`--wiki 1`: 6,000 paragraphs with every token outside a span; `--wiki 0`: none). Nothing else differs.
- **Metrics** (`src/keepspan/metrics.py`): strict and lenient precision, recall and F1; removal (every alphanumeric character of a span covered by a prediction); keep-span violation (any alphanumeric character of a keep span covered); predicted spans per 1,000 words of Wikipedia.
- **Operating curves** (`scripts/curves.py`): a bias on the logit of the outside-a-span label, swept, to tell a better model from a stricter threshold.
- **Export and int8** (`src/keepspan/export.py`, `scripts/onnx_queue.py`): ONNX export, checked against PyTorch on the full test split, and onnxruntime's dynamic int8 quantisation, evaluated on the full test split. With the default recipe the two base models collapse; `scripts/int8_variants.py` finds which weights cause it (the MLP output projections), `scripts/activation_ranges.py` measures the activation outliers at their inputs, and `scripts/eval_int8_recipe.py` evaluates the mixed recipe that keeps those projections in fp32.
- **Timing** (`scripts/latency_cpu.py`, `bench/browser/`): onnxruntime on CPU, and onnxruntime-web on WebAssembly in headless Chromium driven by Playwright.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch                      # a CUDA build for training; the CPU build is enough for the smoke run
pip install -e ".[dev,onnx,paper]"

python scripts/download.py             # ~1.2 GB into data/raw/, checked against data/sources/manifest.json
python scripts/prepare.py              # data/processed/ and data/splits/
python scripts/encode_cache.py         # tokenise the training split once per model

# One run (about 3 minutes for the small model, 15 to 25 for the base models on an RTX 4070 Ti SUPER):
python -m keepspan.train --model deberta --keep 1 --wiki 0 --seed 0
# The study: 3 models x {none, K, W, K+W} x seeds 0-2, one job at a time, resumable.
python scripts/sweep.py

python scripts/eval_offshelf.py        # the off-the-shelf model, with its own head
python scripts/onnx_queue.py           # ONNX export, int8, evaluation on CPU
python scripts/int8_variants.py modernbert-base-s0 --per-source 60    # which weights int8 breaks
python scripts/activation_ranges.py modernbert-base-s0 small-base-s0 deberta-base-s0
python scripts/eval_int8_recipe.py --recipe fp32-mlp-out modernbert-K-s0 deberta-K-s0   # and the other seed-0 runs
python scripts/curves.py --all         # operating curves
(cd bench/browser && npm ci) && scripts/latency_all.sh    # CPU and browser timing, pinned to cores 0-7
python scripts/aggregate.py            # results/summary.json
python scripts/check_results.py        # summary.json against the per-run files
python scripts/readme_results.py       # the tables above
```

The training runs were capped at 3.3 GB of GPU memory because the GPU was shared: the input word-embedding matrix is frozen in every model and the two base models use gradient checkpointing (both recorded in each run's `config.json`).

## Quick check on CPU (no downloads, under a minute)

```bash
pytest -q
python -m keepspan.smoke     # a tiny random BERT trained with and without keep-span negatives on generated text
```

The smoke run checks that every stage runs; its numbers are not the study's.

## Weights

The trained weights are not yet published. Every number above can be recomputed from the committed predictions with the test split (`scripts/download.py`, `scripts/prepare.py`, `scripts/aggregate.py`).

## Citation

See `CITATION.cff`.

## Licence

Apache-2.0 for the code. The data keep their own licences (table above); WikiNEuRal's non-commercial licence is why it is used for evaluation only.
