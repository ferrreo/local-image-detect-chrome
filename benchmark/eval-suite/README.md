# TruePixel offline eval suite

Produces CPU + GPU numbers for:

| Mode | Runtime |
| --- | --- |
| `js-node-cpu-distilled` / `js-node-cpu-dual` | Node `onnxruntime-node` |
| `zig-{webgpu,vulkan,xnnpack,cpu}-{distilled,dual,cascade}` | Zig host + ORT (prefers EP, falls back). `vulkan` aliases native WebGPU/Dawn→Vulkan |
| `js-ext-wasm-cascade` / `js-ext-webgpu-cascade` | Unpacked MV3 extension (ORT-web or Zig+ORT WASM): distilled → CF cascade |

## Local PC

```bash
npm ci                             # once — installs esbuild / playwright / …
npm run setup:models && npm run setup:ort && npm run build:zig
npm run eval:suite                 # builds extension once, then host + browser
open benchmark/eval-suite/index.html
```

If you see `Cannot find package 'esbuild'`, you’re missing `node_modules` — run `npm ci` from the repo root (the suite also tries this automatically once).

## CI

`npm run eval:suite:ci` — limited corpus, host CPU modes, browser WASM (Chromium `--headless=new`).

Playwright defaults to headless. For a visible window: `EVAL_SUITE_HEADED=1 npm run eval:suite:browser` (or `PW_HEADED=1` for integration tests).

## Extension page

`dist/eval.html` runs the full provenance → spectral → visual → fusion path through the offscreen document. Playwright loads the unpacked `dist/` extension and autoruns it against a local corpus HTTP server.
