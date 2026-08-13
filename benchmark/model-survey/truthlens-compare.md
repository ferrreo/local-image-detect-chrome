# TruthLens-related image detector compare

Generated `2026-08-13T22:06:41Z` · **130** images · **6** hardcases · **48** Lexica AI

TruthLens name is overloaded on HF. Image-capable public weights evaluated here: umm-maybe (used by Pikachu771 TruthLens server) and Medsa's TruthLens-branded TorchScript detector. Most other TruthLens repos are text/fake-news classifiers.

## Ranking @ 65%

| Rank | Model | BA | TPR | TNR | Lexica TPR | Avg ms | Hardcase FP | Role |
|-----:|-------|---:|----:|----:|-----------:|-------:|------------:|------|
| 1 | `umm-maybe/AI-image-detector` | 58.1% | 16.3% | 100.0% | 29.2% (14/48) | 88.7 | 0/6 | TruthLens Pikachu771 backend (Swin) |
| 2 | `Medsa/ai-image-authenticity-detector` | 50.0% | 52.3% | 47.7% | 52.1% (25/48) | 9.8 | 3/6 | TruthLens-branded TorchScript CNN (32×32 CIFAKE) |

## Product threshold @ 69.51%

| Model | BA | Avg ms | Hardcase FP |
|-------|---:|-------:|------------:|
| `umm-maybe/AI-image-detector` | 56.4% | 88.7 | 0/6 |
| `Medsa/ai-image-authenticity-detector` | 48.8% | 9.8 | 3/6 |

## Soylent hardcase proxy

| Model | Confidence | @65% | @69.51% | ms |
|-------|-----------:|------|---------|---:|
| `umm-maybe/AI-image-detector` | 4.8% | real/other | real/other | 81.3 |
| `Medsa/ai-image-authenticity-detector` | 24.4% | real/other | real/other | 6.1 |

