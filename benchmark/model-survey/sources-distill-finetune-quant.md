# Model sources: distill / finetune / quant

Inventory of **public** weights and datasets we can legally use to beat Proofmark-class accuracy without shipping their private Q8.

Proofmark’s recipe (for comparison only): freeze `OwensLab/commfor-model-384`, train 384→32→1 head on ~4.6k images (Zitacron + Tiny-GenImage), MatMul-only Q8 (~23 MB), untouched 1.7k test → ~94.6% BA @65%.

---

## 1. Backbones / teachers (finetune or feature distill)

| Source | Format | Size | License | Role | Notes |
|--------|--------|-----:|---------|------|-------|
| **OwensLab/commfor-model-384** | safetensors | 87 MB | MIT | **Primary backbone** | Same ViT-S/16@384 Proofmark froze. Best starting point. |
| OwensLab/commfor-model-224 | safetensors | 87 MB | MIT | Faster backbone experiment | Smaller input; may win latency, lose Lexica. |
| buildborderless/CommunityForensics-DeepfakeDet-ViT | safetensors | 87–144 MB | MIT | Alt CF head | Upstream of our current cascade ONNX. |
| onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX | ONNX q4 | ~24 MB | MIT | **Shipped accurate head today** | Stretch/0.5/logits2. |
| Vontra/detectra-v1 | ONNX + ST | 44 / 87 MB | MIT | Teacher / rival | Same short440-center384 + ImageNet as Proofmark family. 64% BA on our 130. |
| jacoballessio / onnx-community **ai-image-detect-distilled** | ST / ONNX fp16 | 58 / 29 MB | MIT | **Shipped realtime head** | Fast (~40 ms). 0% Lexica TPR alone. |
| onnx-community/Deep-Fake-Detector-v2-Model-ONNX | ONNX fp16 | ~ | Apache-2.0 | Teacher | Best public ONNX on our 130 (~66% BA) but 4/6 hardcase FP. |
| umm-maybe/AI-image-detector | pytorch bin | 348 MB | CC BY-4.0 | Heavy teacher | TruthLens server backend. 58% BA; too big for MV3. |
| wkaandemir/ai-image-detector | safetensors | 343 MB | MIT | Heavy teacher | Convert/quantize if it wins teacher eval. |
| dima806 / capcheck ViTs | ST | ~343 MB | Apache-2.0 | Weak / outdated | CIFAKE-era; poor on modern gens. |
| Organika/sdxl-detector | ST+ONNX | 348 MB | **CC BY-NC-3.0** | Avoid shipping | NC license. |
| onnx-community/SMOGY-…-ONNX | ONNX | — | **CC BY-NC-4.0** | Avoid shipping | NC license. |
| Red-had1911/deepfake-detector-onnx | ONNX | 87–378 MB | MIT | Probe | `generative_detector.onnx` (~87 MB) worth a compare pass. |
| vijayakumarn85/commfor-model-224-onnx | ONNX | 88 MB | ? | Fast CF-224 | Thin provenance; verify before ship. |
| Proofmark/proofmark-webwild-v3 | ONNX Q8 | 23 MB | private HF | **Do not fetch** | Their fine-tune quant. |

### Recommended use

| Goal | Source |
|------|--------|
| Finetune / head-distill backbone | `OwensLab/commfor-model-384` |
| Speed student (realtime) | Keep or re-distill into jacoballessio-sized ViT / CF-224 |
| Soft-label teachers (ensemble KD) | Detectra-v1 + Deep-Fake-v2 + CF-q4 logits on train images |
| Ship today | Distilled fp16 realtime + CF-q4 accurate |

---

## 2. Datasets (train / calib / holdout)

| Dataset | License | Use | Notes |
|---------|---------|-----|-------|
| **Zitacron/real-vs-ai-corpus** | CC BY-4.0 | **Primary train** | What Proofmark sampled (~flux, synthetic-characters, laion-aesthetic, …). |
| **TheKernel01/Tiny-GenImage** | CC BY-NC-SA-4.0 | Train (NC) | Generator-tagged; Proofmark used. Images not redistributed. |
| OwensLab/CommunityForensics-Small | CC BY-NC-SA-4.0 | Train / domain mix | Large; parquet shards. |
| OwensLab/CommunityForensics-Eval | CC BY-NC-SA-4.0 | Held-out eval only | Do not train on this if we want a clean CF-family test. |
| dragonintelligence/CIFAKE / yanbax CIFAKE | MIT / varies | Optional warm-up | Too easy / outdated for Lexica. |
| Our `benchmark/openrouter` | mixed fetch | Train (non-Lexica) | Modern OpenRouter gens. |
| Our `benchmark/openrouter/ai/lexica__feed` | scraped | **Frozen test** | Never train; promotion gate. |
| Our hardcases | local | Frozen FP gate | |

Target scale to match Proofmark: **≥4k train images**, separate calib (~1k), Lexica+hardcases untouched.

---

## 3. Quantization paths

| Method | Typical size | Latency | Browser notes |
|--------|-------------:|---------|---------------|
| **Dynamic Q8 MatMul-only** (Proofmark) | ~23–24 MB | ~80–140 ms CPU | ORT Web friendly; leave Conv fp32. |
| Dynamic Q4 (current CF) | ~24 MB | similar | Already shipped; slightly different graph. |
| FP16 ONNX | ~29–45 MB | fast on WebGPU | Distilled path today. |
| FP32 | ~58–87 MB | slow | Dev only. |
| Knowledge-distill → tiny student then Q8 | &lt;15 MB goal | **speed win** | Second phase after head quality clears Lexica. |

Ship rule: only MIT / Apache-2.0 weights in `dist/`. NC datasets OK for local training if we do not redistribute images.

---

## 4. Attack plan (beat Proofmark on quality + speed)

1. **Data** — Pull Zitacron stratified sample + Tiny-GenImage validation/train slices into `benchmark/distill-corpus/{ai,real}/…`. Keep Lexica + hardcases out.
2. **Finetune head** — Freeze CommFor-384; train 384→32→1 (and 16/64 sweep) with JPEG/recompress/crop aug + domain weights; calibrate threshold on held-out calib for ≥90% TNR.
3. **Optional KD** — Mix hard labels with teacher soft labels from Detectra + DF-v2.
4. **Quant** — Export MatMul-only Q8; also try Q4 and a smaller CF-224 student for speed.
5. **Gate** — Promote only if: Lexica TPR @65% ≫ current (~15% domain-holdout), hardcase FP = 0, BA on clean holdout ≥ Proofmark’s public claim band, and avg ms ≤ distilled+CF cascade wall time.
6. **Speed** — Realtime stays distilled; accurate head Q8 with viewport gate; later distill a single fused student if quality allows dropping the second session.

---

## 5. Already measured on our 130-image tree (@65%)

| Model | BA | Lexica TPR | Hard FP | Avg ms |
|-------|---:|-----------:|--------:|-------:|
| truepixel-accurate (train∋Lexica) | ~99% | ~98% | 0 | ~83 | contaminated |
| truepixel-accurate (Lexica-out) | — | **~13–19%** | 0 | ~83 | honest |
| Deep-Fake-Detector-v2 ONNX | 65.6% | 75% | 4/6 | 110 |
| detectra-v1 | 64.0% | 33% | 1/6 | 102 |
| umm-maybe (TruthLens backend) | 58.1% | 29% | 0/6 | 89 |
| distilled (ours realtime) | 52.4% | 0% | 0/6 | **40** |
| Medsa TruthLens CNN | 50.0% | 52% | 3/6 | 10 |

Bottom line: **data scale + frozen CommFor-384 + proper holdout** is the Proofmark gap; quant trick is secondary. Next work is corpus fetch + upgraded `distill:accurate`, not more HF name shopping.
