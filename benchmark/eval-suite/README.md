# TruePixel offline eval suite

Produces CPU + GPU numbers for:

| Mode | Runtime |
| --- | --- |
| `js-node-cpu-distilled` / `js-node-cpu-dual` | Node `onnxruntime-node` |
| `zig-{webgpu,vulkan,xnnpack,cpu}-{distilled,dual,cascade}` | Zig host + ORT (prefers EP, falls back) |
| `js-ext-wasm-dual` / `js-ext-webgpu-dual` | Unpacked MV3 extension in Playwright Chromium |

## Local PC

```bash
npm run setup:models && npm run setup:ort && npm run build:zig
npm run build
npm run eval:suite
open benchmark/eval-suite/index.html
```

## CI

`npm run eval:suite:ci` — limited corpus, host CPU modes, browser WASM under xvfb.

## Extension page

`dist/eval.html` runs the full provenance → spectral → visual → fusion path through the offscreen document. Playwright loads the unpacked `dist/` extension and autoruns it against a local corpus HTTP server.
