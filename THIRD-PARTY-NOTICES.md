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

## CommunityForensics-DeepfakeDet-ViT (ONNX)

- Upstream: Jeongsoo Park & Andrew Owens / Community Forensics
- HF packaging: `onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX`
- License: MIT
- Paper: https://arxiv.org/abs/2411.04125
- https://huggingface.co/onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX

Downloaded during one-time setup alongside the distilled detector and used as a secondary visual signal in fusion.
