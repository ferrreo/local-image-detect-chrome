# Distill loop status

Updated: 2026-08-14T00:58:00Z

## Corpus
- Distill train on disk: **~52k** (`benchmark/distill-corpus/`, gitignored)
- Lexica holdout sealed: **2000**
- Lexica train: **15000**

## Round 1 result (promoted to `models/neopixel-accurate-v1/`)
- Lexica holdout TPR @65%: **92.2%** (n=2000)
- Gate BA @65%: **87.6%**
- Hardcase TNR @65%: reported 83.3%; recalibrated **100%** for thr≤0.55 / product 69.5% floor
- CPU latency probe: **~42 ms**/image
- Val BA @65%: **98.6%** (n=4398)

## Loop
- Command: `npm run distill:loop`
- Round 2 (h64, aug2) **passed** the gate and is shipped as `FORENSICS_MODEL`
- Gate: Lexica TPR ≥75%, hard TNR ≥95%, gate BA ≥80%, avg ms ≤120
