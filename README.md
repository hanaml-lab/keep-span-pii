# keep-span-pii: learning what to keep in on-device PII span detection

[![ci](https://github.com/hanaml-lab/keep-span-pii/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/keep-span-pii/actions/workflows/ci.yml)

Code, configurations, logs and predictions for *Learning What to Keep: Keep-Span Negatives and Over-Redaction in On-Device PII Span Detection* (Agarwal, hanaML technical report, 2026). The report is at [hanaml.com/research/papers/keep-span-pii](https://hanaml.com/research/papers/keep-span-pii).

A model that redacts personal data on a device before text reaches a cloud model can fail in two directions: it can leave personal data readable, or it can remove the courts, places and dates the cloud model needs. Most PII training data mark only what to remove. The Text Anonymization Benchmark (TAB) also marks the entity mentions its annotators decided to keep. This repository asks, on public data only, whether training on those keep spans as negatives reduces over-redaction without losing recall, across three model sizes, and what int8 quantisation for the browser costs.

**Every number here is measured**: fine-tuning runs on one NVIDIA RTX 4070 Ti SUPER, three seeds per configuration, with each run's configuration, training log, metrics, test predictions (as character offsets), run time and memory committed under `results/runs/`. Nothing is simulated.

## Results

<!-- results:start -->
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
- **Export and int8** (`src/keepspan/export.py`, `scripts/onnx_queue.py`): ONNX export, checked against PyTorch on the full test split, and onnxruntime's dynamic int8 quantisation, evaluated on the full test split.
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
python scripts/curves.py --all         # operating curves
taskset -c 0-7 python scripts/latency_cpu.py --runs small-K-s0 modernbert-K-s0 deberta-K-s0
(cd bench/browser && npm ci && node bench.mjs --models small-int8=../../checkpoints/small-K-s0/onnx/model_int8.onnx \
    --vocab small-int8=30522 --out ../../results/latency_browser.json)
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
