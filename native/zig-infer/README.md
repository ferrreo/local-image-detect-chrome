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

## WASM spike

```bash
zig build -Dwasm=true -Doptimize=ReleaseFast
```

Exports SIMD preprocess helpers only (`tp_rgb_to_nchw_half`, …). Full in-browser ORT stays on `onnxruntime-web` until a static ORT wasm library is linked.
