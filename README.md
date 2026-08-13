# TruePixel

Chrome Manifest V3 extension that detects AI-generated images entirely inside the browser.

No cloud inference. No local Python/Node server. After a one-time download of public model weights, scanning stays offline on-device via WebGPU (preferred) or WASM.

## What it does

Install the extension, open the popup, download models once, then browse. TruePixel automatically analyzes visible images and overlays a confidence badge (`AI 82%`, `Real 71%`, or uncertain).

Detection pipeline:

1. **Provenance** — EXIF/XMP/C2PA/string fingerprints for known generators
2. **Spectral forensics** — FFT / noise / chroma features in plain TypeScript
3. **Visual classifier** — distilled ViT ONNX model through `onnxruntime-web`
4. **Fusion** — calibrated score with a default **65%** AI threshold (bounty evaluation point)

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

Local eval prefers the stored OpenRouter multi-model corpus under `benchmark/openrouter/` (stub visual path by default):

```bash
npm run eval:local
```

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
| Visual model | `onnx-community/ai-image-detect-distilled-ONNX` (`model_fp16.onnx`) |
| Model license | MIT |
| Runtime | `onnxruntime-web` WebGPU → WASM fallback |
| Threshold default | 0.65 |
