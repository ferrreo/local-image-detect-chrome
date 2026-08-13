# TruePixel accurate-v1 Q8 — honest holdout

Artifact: `models/truepixel-accurate-v1/model_quantized.onnx` (~24 MB, ImageNet short440-center384, logit+sigmoid).
Backbone: public `OwensLab/commfor-model-384`. Head: TruePixel-trained 384→32→1.

## Contaminated full-corpus compare (do not promote on this)

When Lexica is **in** the train set, `npm run eval:compare` reported ~99% BA / ~98% Lexica TPR on the same 130 images. That is train leakage.

## Domain holdout (train without Lexica)

Protocol: train on 82 non-Lexica images (with 1 augment view), export Q8, score never-seen Lexica + hardcases.

| Split | n | BA @65% | TPR | TNR |
|-------|--:|--------:|----:|----:|
| Train (non-Lexica) | 82 | 100% | 100% | 100% |
| Lexica feed (held out) | 48 | — (all AI) | **12.5–19%** | — |
| Hardcases (held out reals) | 6 | — | — | **100%** (0 FP) |
| Lexica + hardcases | 54 | ~57% | ~15% | 100% |

Lexica score distribution (held-out model): mean ~0.21, median ~0.014. Most tiles sit near zero; a few clear 0.65.

## Verdict

- Our quant is real and ours (not Proofmark’s).
- It is **not** ready to replace Community Forensics in the extension.
- It memorizes the tiny OpenRouter+Lexica tree; it does **not** generalize to Lexica when Lexica is held out.
- Next: `npm run fetch:corpus50k` (~50k train + sealed Lexica holdout), then `npm run distill:loop` until Lexica holdout TPR, hardcase TNR, gate BA, and latency clear the bar vs Community Forensics.
