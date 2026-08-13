# truepixel-infer (Zig 0.16 + ONNX Runtime)

Host-side inference for local eval / tooling. Tries execution providers in order:

1. **WebGPU** (when the ORT package includes it)
2. **Vulkan** (same)
3. **XNNPACK** (same)
4. **CPU** — ORT MLAS, built for `x86_64_v3` (AVX2+)

Stock `onnxruntime-linux-x64` ships CPU only; GPU EPs fall through immediately. ORT shared libs are installed next to the binary (`zig-out/lib`, RPATH `$ORIGIN/../lib`).

## Build

Requires [Zig 0.16](https://ziglang.org/download/), `libwebp`, and ORT:

```bash
npm run setup:ort
# PATH must include zig 0.16
npm run build:zig
```

## Protocol (JSONL on stdin/stdout)

```json
{"cmd":"ping"}
{"cmd":"warm"}
{"cmd":"infer","path":"benchmark/openrouter/real/picsum_10.jpg","models":["distilled"]}
{"cmd":"quit"}
```

EOF on stdin exits immediately (no hang). Use `quit` for a clean bye.

## WASM spike

```bash
zig build -Dwasm=true -Doptimize=ReleaseFast
```

Exports SIMD preprocess helpers only (`tp_rgb_to_nchw_half`, …). Full in-browser ORT stays on `onnxruntime-web` until a static ORT wasm library is linked.
