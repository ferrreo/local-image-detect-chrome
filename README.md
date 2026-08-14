# NeoPixel

Chrome MV3 extension that scores whether an image looks AI-generated. Inference stays on-device (WebGPU or WASM). No page text, no cloud classifier.

## Install (unpacked)

```bash
npm ci
npm run build
```

1. `chrome://extensions` → Developer mode
2. **Load unpacked** → `dist/`
3. Toolbar icon → **Download models** (one-time)

While browsing, NeoPixel blurs images until scored. **AI** tiles stay blurred (with an `AI N%` badge). Real / uncertain stay clear; those badges only show if you enable **Debug** in options.

## How it works

Pixels only — no surrounding captions or page context.

1. **Watermarks / provenance** — visible marks, SynthID soft-binding when present, EXIF/C2PA-ish fingerprints on the byte stream
2. **Spectral features** — FFT / noise / chroma / edge geometry in TypeScript (`spectral.ts`)
3. **Visual heads** — fast distilled ViT, then the NeoPixel accurate head (Q8 ONNX) via `onnxruntime-web`
4. **Fusion** — combines scores with holds for UI screenshots, flat graphics, and photo macros; promotes neon/CGI stock that the heads undershoot

Default labels: AI if P(AI) ≥ **69.51%**, Real if ≤ **40.99%**, else uncertain. Bounty-style eval often scores at 65%.

Google Images: prefers full `imgurl` over mushy `encrypted-tbn` thumbs when the page exposes it.

## Build

```bash
npm ci
npm run build          # icons + fixtures + esbuild → dist/
npm run typecheck
npm run test:unit
```

Integration (Playwright Chromium — branded Chrome dropped `--load-extension`):

```bash
npx playwright install chromium
npm run build
xvfb-run --auto-servernum npm run test:integration   # CI / headless
# or: npm run test:integration
```

Optional model prefetch into `./models` (otherwise the popup downloads them):

```bash
npm run setup:models
```

## Options worth knowing

| Setting | Effect |
| --- | --- |
| Hide AI images | `blur` / `blank` / off |
| Debug mode | Also show Real and `?` badges |
| Thresholds | AI floor / Real ceiling |

## Eval (optional)

Committed under `benchmark/openrouter/ai/`: OpenRouter API samples we generated for this project.

Real photos are optional and gitignored:

```bash
# .env: OPENROUTER_API_KEY=...
npm run fetch:openrouter   # refresh AI samples
npm run fetch:real         # Unsplash/Picsum into real/ (local only)
npm run eval:local
npm run eval:suite
```

Synthetic fixtures under `tests/fixtures/images/` cover unit/CI smoke without a full corpus.

Accurate-head training (Python) is optional and documented under `benchmark/model-survey/`. Shipped forensics weights live in `models/neopixel-accurate-v1/`.

## Privacy

Image bytes are fetched by the extension and scored in an offscreen document / service worker. No analytics, accounts, or remote inference. Model weights are public HF artifacts, SHA-256 checked, then cached.

## License

MIT — see [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
