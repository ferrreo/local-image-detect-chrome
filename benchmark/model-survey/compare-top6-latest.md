# Model compare report (OpenRouter + hardcases)

Generated `2026-08-13T21:45:27.007Z` · **130** images · **6** hardcases · **48** Lexica AI

Two requested models lacked usable detector ONNX weights; substitutes noted below. Soylent hardcase is a local proxy (original screenshot bytes were not available to the agent). proofmark-webwild-v3 is the Q8 ONNX shipped by Dyno-man/Dino-ImageGen-Ext.

## Unavailable requested models

- **ai-human-generated-image-detection-ONNX** — Hugging Face repo has no weights (only .gitattributes). Substituted by zonn-ai/ai-image-detection-ONNX.
- **deepguard-ai** — Repo ships EfficientNet .pth + inswapper face-swap ONNX, not a classifier ONNX. Substituted by current distilled.

## Ranking @ 65% threshold (bounty)

| Rank | Model | BA | TPR | TNR | Lexica TPR | Avg ms/img | Total s | Hardcase FP | Notes |
|-----:|-------|---:|----:|----:|-----------:|-----------:|--------:|------------:|-------|
| 1 | `proofmark-webwild-v3` | 76.8% | 55.8% | 97.7% | 60.4% (29/48) | 89.7 | 11.7 | 0/6 | Proofmark vendor Q8 |
| 2 | `Deep-Fake-Detector-v2-Model-ONNX` | 65.6% | 74.4% | 56.8% | 75.0% (36/48) | 108.7 | 14.2 | 4/6 | requested |
| 3 | `detectra-v1` | 64.0% | 30.2% | 97.7% | 33.3% (16/48) | 113.6 | 14.8 | 1/6 | requested |
| 4 | `ai-image-detect-distilled` | 52.4% | 7.0% | 97.7% | 0.0% (0/48) | 30.0 | 4.0 | 0/6 | sub for deepguard-ai |
| 5 | `zonn-ai-image-detection-ONNX` | 50.0% | 2.3% | 97.7% | 4.2% (2/48) | 89.5 | 11.7 | 0/6 | sub for ai-human-generated-image-detection-ONNX |
| 6 | `ai-image-detection-ONNX` | 49.2% | 80.2% | 18.2% | 79.2% (38/48) | 111.0 | 14.5 | 5/6 | requested |
| 7 | `sdxl-detector` | 40.1% | 32.6% | 47.7% | 22.9% (11/48) | 113.9 | 14.8 | 3/6 | requested |

## Product threshold @ 69.51%

| Model | BA | Avg ms/img | Hardcase FP |
|-------|---:|-----------:|------------:|
| `proofmark-webwild-v3` | 76.2% | 89.7 | 0/6 |
| `Deep-Fake-Detector-v2-Model-ONNX` | 56.1% | 108.7 | 2/6 |
| `detectra-v1` | 63.4% | 113.6 | 1/6 |
| `ai-image-detect-distilled` | 50.6% | 30.0 | 0/6 |
| `zonn-ai-image-detection-ONNX` | 50.0% | 89.5 | 0/6 |
| `ai-image-detection-ONNX` | 45.7% | 111.0 | 5/6 |
| `sdxl-detector` | 40.1% | 113.9 | 3/6 |

## Soylent hardcase proxy

| Model | Confidence | @65% | @69.51% | ms |
|-------|-----------:|------|---------|---:|
| `proofmark-webwild-v3` | 0.0% | real/other | real/other | 74.7 |
| `Deep-Fake-Detector-v2-Model-ONNX` | 72.2% | AI | AI | 97.1 |
| `detectra-v1` | 0.0% | real/other | real/other | 105.7 |
| `ai-image-detect-distilled` | 23.5% | real/other | real/other | 21.1 |
| `zonn-ai-image-detection-ONNX` | 16.3% | real/other | real/other | 63.0 |
| `ai-image-detection-ONNX` | 91.9% | AI | AI | 96.3 |
| `sdxl-detector` | 0.0% | real/other | real/other | 100.5 |

## Verdict

- **Best BA @65%:** `proofmark-webwild-v3` at **76.8%** (avg **89.7 ms**/image, **11.7 s** total).
- **Fastest:** `ai-image-detect-distilled` at **30.0 ms**/image (**4.0 s** total).
- **Best hardcase FP control:** `proofmark-webwild-v3` with **0/6** hardcase false positives @65%.
