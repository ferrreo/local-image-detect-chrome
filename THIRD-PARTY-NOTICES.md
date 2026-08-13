# Third-party notices

## onnxruntime-web

- License: MIT
- https://github.com/microsoft/onnxruntime

## ai-image-detect-distilled (ONNX)

- Upstream: `jacoballessio/ai-image-detect-distilled`
- ONNX packaging: `onnx-community/ai-image-detect-distilled-ONNX`
- License: MIT
- https://huggingface.co/jacoballessio/ai-image-detect-distilled

TruePixel downloads these weights at setup time (or via `npm run setup:models`) and verifies SHA-256 before caching.

## ai-image-detection / CapCheck (ONNX, optional eval)

- Upstream: CapCheck / `onnx-community/ai-image-detection-ONNX`
- License: Apache-2.0
- https://huggingface.co/onnx-community/ai-image-detection-ONNX

Downloaded by `npm run setup:models` for local evaluation experiments. Not required by the extension runtime unless an ensemble path is enabled.
