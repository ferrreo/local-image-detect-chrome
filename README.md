# TruePixel

Chrome Manifest V3 extension that detects AI-generated images entirely inside the browser.

No cloud inference. No local Python/Node server. After a one-time download of public model weights, scanning stays offline on-device via WebGPU (preferred) or WASM.

## What it does

Install the extension, open the popup, download models once, then browse. TruePixel automatically analyzes visible images and overlays a confidence badge (`AI 82%`, `Real 71%`, or uncertain).

Detection pipeline:

1. **Provenance** — EXIF/XMP/C2PA/AIGC/string fingerprints (scans file head + tail)
2. **Spectral forensics** — FFT / noise / chroma features in plain TypeScript
3. **Visual classifiers** — fast distilled ViT, then Proofmark webwild-v3 (accurate) via `onnxruntime-web`
4. **Fusion** — calibrated ensemble (product AI label ≥69.51%; bounty eval often uses 65%)

## Requirements

- Google Chrome 121+ (WebGPU capable preferred)
- Node.js 20+ to build from source

## Build from source

```bash
npm ci
node scripts/generate-icons.mjs
node scripts/generate-fixtures.mjs
npm run build
```

Optional: prefetch model weights into `./models` (also downloaded in-browser on first setup):

```bash
npm run setup:models
```

Proofmark’s accurate head (`proofmark-webwild-v3` Q8) has a private HF repo (`Proofmark/proofmark-webwild-v3`, 401). Setup/extension download the matching public ONNX from [Dyno-man/Dino-ImageGen-Ext](https://github.com/Dyno-man/Dino-ImageGen-Ext) (backbone [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384)).

Load unpacked:

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select the `dist/` directory
4. Click the TruePixel toolbar icon → **Download models**

## Tests

Unit tests cover each integrated part (provenance, spectral/FFT, fusion, image decode, model cache, visual stub, pipeline):

```bash
npm run test:unit
```

Integration tests launch a real Chromium profile (Playwright's Chromium / Chrome for Testing), load the unpacked extension, and exercise popup/options/content-script overlays.

Branded Google Chrome 137+ removed `--load-extension`, so automated tests use Playwright Chromium. Manual installs still use normal Chrome via Load unpacked.

```bash
npm run build
npx playwright install chromium
# headed Chromium is required for --load-extension
xvfb-run --auto-servernum npm run test:integration   # CI / headless machines
npm run test:integration                             # local desktop
```

Full suite:

```bash
npm run test:all
```

### GitHub Actions / `gh`

CI is defined in `.github/workflows/ci.yml` (unit, Playwright Chromium integration, production build artifact).

```bash
gh workflow list
gh workflow run ci.yml
gh run watch
gh run view --log
```

## Evaluation threshold

Balanced accuracy is measured at a **65% confidence threshold**, matching the bounty brief. Configure the threshold on the options page if you need a different operating point for personal use.

Local eval prefers the stored OpenRouter multi-model corpus under `benchmark/openrouter/` (real ONNX by default; `TRUEPIXEL_STUB=1` for the heuristic stub):

```bash
npm run setup:ort && npm run build:zig   # once (Zig 0.16 + libwebp)
npm run eval:local                       # prefers Zig+ORT host when built
# TRUEPIXEL_BACKEND=node npm run eval:local
# TRUEPIXEL_STUB=1 npm run eval:local
```

OpenRouter corpus (76 images, threshold 65%), same machine:

| Config | BA @65% | tp | tn | fp | fn | avg ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Node CPU · distilled only | 93.4% | 33 | 38 | 0 | 5 | ~47 |
| Zig CPU · distilled only | 92.1% | 32 | 38 | 0 | 6 | ~46 |
| Node CPU · dual (always CF) | **100%** | 38 | 38 | 0 | 0 | ~130 |
| Zig CPU · dual (always CF) | **100%** | 38 | 38 | 0 | 0 | ~158 |
| Zig CPU · cascade (default eval) | **100%** | 38 | 38 | 0 | 0 | **~78** (CF on 14/76) |

Distilled-alone misses hard Krea/Riverflow and Lexica feed cases. The browser cascade now recovers with **Proofmark webwild-v3** (Q8 accurate head; ~77–78% BA @65% on the 130-image OpenRouter+Lexica corpus, 0 hardcase FPs). `npm run eval:compare` ranks heads; see `benchmark/model-survey/compare-top6-latest.md`. The Zig host still tries **WebGPU** (Dawn→Vulkan) → CUDA → XNNPACK → CPU; browser accurate path prefers ort-web while Proofmark needs ImageNet center-crop preprocess.

### Full offline suite (CPU + GPU modes)

Runs the **real unpacked extension** in Playwright Chromium (ORT WebGPU / WASM) plus host Node/Zig matrices, then writes `benchmark/eval-suite/index.html`.

```bash
npm ci                             # required once (esbuild, playwright, …)
npm run setup:models && npm run setup:ort && npm run build:zig
npm run setup:ort-wasm && npm run build:zig-wasm   # once: link ORT into extension WASM
npm run eval:suite                 # local PC: host + browser zig-ort-wasm / webgpu cascade
npm run eval:suite:ci              # CI-sized subset (wasm + key host modes)
# EVAL_SUITE_LIMIT=16 EVAL_SUITE_BROWSER_PROVIDERS=webgpu,wasm npm run eval:suite
```

Browser path: **realtime distilled**, then **accurate Proofmark refine** when the first paint is below the AI threshold (Lexica-class near-zero distilled scores included). Silent stub fallback is disabled — if offscreen ORT fails, the result is an error, not a fake score.

Live page inside the extension (after Load unpacked `dist/`):

`chrome-extension://<id>/eval.html?corpus=http://127.0.0.1:<port>&provider=webgpu&autorun=1`

Serve the corpus with any static server rooted at `benchmark/openrouter` (the Playwright harness starts one automatically).

Refresh that corpus incrementally when OpenRouter adds models (needs `OPENROUTER_API_KEY` in `.env`):

```bash
npm run fetch:openrouter   # skips models already in benchmark/openrouter/registry.json
npm run fetch:real
```

## Privacy

- Image bytes are fetched by the extension and processed in an offscreen document / service worker
- No analytics, accounts, or remote inference APIs
- Model weights are public Hugging Face artifacts downloaded once, then cached with SHA-256 verification

## License

MIT. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Reproducibility

| Item | Value |
| --- | --- |
| Extension | Manifest V3, TypeScript → esbuild bundle in `dist/` |
| Visual models | `ai-image-detect-distilled` fp16 + `CommunityForensics-DeepfakeDet-ViT` q4 |
| Model license | MIT |
| Runtime | `onnxruntime-web` WebGPU → WASM fallback |
| Threshold default | 0.65 |
