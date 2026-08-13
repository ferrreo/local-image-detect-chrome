# Model compare report (OpenRouter + hardcases)

Generated `2026-08-13T22:00:31.060Z` · **130** images · **6** hardcases · **48** Lexica AI

Two requested models lacked usable detector ONNX weights; substitutes noted below. Soylent hardcase is a local proxy. We do not fetch private Proofmark quants — train truepixel-accurate-v1 with npm run distill:accurate.

**Leakage warning:** `truepixel-accurate-v1` below was trained on this same OpenRouter+Lexica tree. Full-corpus 99% BA is not a promotion signal. Use `models/truepixel-accurate-v1/manifest.json` → `onnxHoldoutAt065` (and grow the corpus) before swapping FORENSICS_MODEL.

## Unavailable requested models

- **ai-human-generated-image-detection-ONNX** — Hugging Face repo has no weights (only .gitattributes). Substituted by zonn-ai/ai-image-detection-ONNX.
- **deepguard-ai** — Repo ships EfficientNet .pth + inswapper face-swap ONNX, not a classifier ONNX. Substituted by current distilled.
- **proofmark-webwild-v3** — Proofmark/proofmark-webwild-v3 is private on HF; we do not fetch or ship third-party bundled quants. Distill our own accurate head instead.

## Ranking @ 65% threshold (bounty)

| Rank | Model | BA | TPR | TNR | Lexica TPR | Avg ms/img | Total s | Hardcase FP | Notes |
|-----:|-------|---:|----:|----:|-----------:|-----------:|--------:|------------:|-------|
| 1 | `truepixel-accurate-v1` | 99.4% | 98.8% | 100.0% | 97.9% (47/48) | 83.0 | 10.8 | 0/6 | our distill |
| 2 | `Deep-Fake-Detector-v2-Model-ONNX` | 65.6% | 74.4% | 56.8% | 75.0% (36/48) | 110.3 | 14.4 | 4/6 | requested |
| 3 | `detectra-v1` | 64.0% | 30.2% | 97.7% | 33.3% (16/48) | 101.6 | 13.3 | 1/6 | requested |
| 4 | `ai-image-detect-distilled` | 52.4% | 7.0% | 97.7% | 0.0% (0/48) | 38.9 | 5.1 | 0/6 | sub for deepguard-ai |

## Product threshold @ 69.51%

| Model | BA | Avg ms/img | Hardcase FP |
|-------|---:|-----------:|------------:|
| `truepixel-accurate-v1` | 99.4% | 83.0 | 0/6 |
| `Deep-Fake-Detector-v2-Model-ONNX` | 56.1% | 110.3 | 2/6 |
| `detectra-v1` | 63.4% | 101.6 | 1/6 |
| `ai-image-detect-distilled` | 50.6% | 38.9 | 0/6 |

## Soylent hardcase proxy

| Model | Confidence | @65% | @69.51% | ms |
|-------|-----------:|------|---------|---:|
| `truepixel-accurate-v1` | 0.1% | real/other | real/other | 73.4 |
| `Deep-Fake-Detector-v2-Model-ONNX` | 72.2% | AI | AI | 96.3 |
| `detectra-v1` | 0.0% | real/other | real/other | 91.6 |
| `ai-image-detect-distilled` | 23.5% | real/other | real/other | 30.7 |

## Verdict

- **Best BA @65%:** `truepixel-accurate-v1` at **99.4%** (avg **83.0 ms**/image, **10.8 s** total).
- **Fastest:** `ai-image-detect-distilled` at **38.9 ms**/image (**5.1 s** total).
- **Best hardcase FP control:** `truepixel-accurate-v1` with **0/6** hardcase false positives @65%.
