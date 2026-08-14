# Offline eval suite

Host (Node ORT) + browser (unpacked extension via Playwright).

| Mode | Stack |
| --- | --- |
| `js-node-cpu-distilled` / `js-node-cpu-dual` | onnxruntime-node |
| `js-ext-{wasm,webgpu}-cascade` | Extension `onnxruntime-web` |

```bash
npm run setup:models
npm run eval:suite
npm run eval:suite:ci
```

If `benchmark/openrouter` has no images on disk, the harness uses `tests/fixtures/images/` instead.

Reports land in `benchmark/eval-suite/` when you run the suite (not committed).
