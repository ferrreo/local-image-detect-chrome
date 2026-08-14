# Third-party notices

## onnxruntime-web

- License: MIT
- https://github.com/microsoft/onnxruntime

## ai-image-detect-distilled (ONNX)

- Upstream: `jacoballessio/ai-image-detect-distilled`
- ONNX packaging: `onnx-community/ai-image-detect-distilled-ONNX`
- License: MIT
- https://huggingface.co/jacoballessio/ai-image-detect-distilled

Downloaded at setup (or `npm run setup:models`) and verified with SHA-256 before caching.

## neopixel-accurate-v1 (ONNX)

- Backbone: OwensLab / Community Forensics ViT (`OwensLab/commfor-model-384`)
- Fine-tune / Q8 export: this repo (`npm run distill:accurate`)
- License: MIT (upstream backbone)
- Paper: https://arxiv.org/abs/2411.04125

Shipped under `models/neopixel-accurate-v1/` as the secondary visual (“accurate”) head.

## OpenSynthID detect (optional watermark pass)

- Quantized ONNX under `models/opensynthid-detect/`
- See that folder’s `manifest.json` for upstream attribution
