# truepixel-infer (Zig 0.16 + ONNX Runtime)

Host-side inference for local eval / tooling. Tries execution providers in order:

1. **WebGPU** — native plugin EP (`libonnxruntime_providers_webgpu.so`), Dawn → **Vulkan** on Linux
2. **CUDA** — only with the ORT `*-gpu*` package
3. **XNNPACK** — when the linked ORT build exposes it
4. **CPU** — ORT MLAS, built for `x86_64_v3` (AVX2+)

There is no separate ORT “Vulkan” EP. Preferring `Vulkan` in the JSONL protocol is an alias for WebGPU.

`npm run setup:ort` downloads ORT **≥ 1.24.4** (default **1.29.0**) plus the WebGPU plugin from NuGet. Shared libs land next to the binary (`zig-out/lib`, RPATH `$ORIGIN/../lib`).

## Build

Requires [Zig 0.16](https://ziglang.org/download/), `libwebp`, and ORT:

```bash
npm run setup:ort
# optional NVIDIA CUDA package:
# TRUEPIXEL_ORT_VARIANT=gpu npm run setup:ort
npm run build:zig
```

## Protocol (JSONL on stdin/stdout)

```json
{"cmd":"ping"}
{"cmd":"warm","preferEp":"WebGPU"}
{"cmd":"infer","path":"benchmark/openrouter/real/picsum_10.jpg","models":["distilled"]}
{"cmd":"quit"}
```

EOF on stdin exits immediately (no hang). Use `quit` for a clean bye.

## WASM (Zig + ORT static lib)

Links `libonnxruntime_webassembly.a` into the extension module (same cascade dual as the host binary, WASM EP):

```bash
npm run setup:ort-wasm   # builds ORT wasm static lib via emscripten (slow, cached)
npm run build:zig-wasm   # zig → .o, emcc link → zig-out/wasm/truepixel_infer.{js,wasm}
npm run build            # copies into dist/wasm/
```

Layout:
- `src/wasm_api.zig` — SIMD preprocess (`tp_rgb_to_nchw_half`, …)
- `src/wasm_ort_bridge.c` — ORT C API sessions (`tp_session_create` / `tp_session_run`)
- Linked with `libonnxruntime_webassembly.a` → `tp_has_ort_session() === 1`

The offscreen document prefers this engine when `dist/wasm/` is present; otherwise falls back to `onnxruntime-web`.
