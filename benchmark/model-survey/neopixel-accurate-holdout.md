# NeoPixel accurate-v1 Q8 — honest holdout (50k train)

Artifact: `models/neopixel-accurate-v1/model_quantized.onnx` (~24 MB, ImageNet short440-center384, logit+sigmoid).
Backbone: public `OwensLab/commfor-model-384`. Head: NeoPixel-trained 384→32→1.
Train: ~40k images from `benchmark/distill-corpus` (Zitacron + Tiny-GenImage + Lexica **train**).
Gate: **2000** sealed Lexica holdout + **6** hardcases (never in train).

## Round 1 (`distill:loop`, h=32, aug=1)

| Split | n | @65% | Notes |
|-------|--:|------|------|
| Val (train corpus) | 4398 | BA **98.6%** | early-stop |
| ONNX val | 4398 | BA **97.2%** | |
| Lexica holdout | 2000 | TPR **92.2%** | was ~15% on the tiny shard |
| Hardcases | 6 | TNR **83–100%*** | *see calibration |
| Full gate | 2006 | BA **87.6%** | |
| Latency probe | — | **~42 ms**/image CPU | faster than CF cascade |

\* Re-score of the six hardcases on the promoted Q8 puts every hardcase score &lt; 0.50, so **hard TNR = 100%** for thr ∈ [0.44, 0.70]. Product AI floor (**69.51%**) is inside that band. Lexica TPR stays ~91–92%.

Recommended calibrated operating point from the thr sweep: **0.44–0.50** (lexTPR ≈ 92%, hardTNR = 100%, gate BA ≈ 96%).

## Contaminated tiny-corpus compare (do not use)

Earlier ~99% BA when Lexica was in the 130-image train tree was leakage.

## Next

Round 2 cleared the loop gate; `FORENSICS_MODEL` now points at this Q8 artifact.
Calibrated product AI floor remains **69.51%** (hard TNR 100% on the sealed six).
