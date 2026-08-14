#!/usr/bin/env python3
"""Convert fyxme/opensynthid-detect-0.1 (.pt) → ONNX (+ dynamic Q8) for NeoPixel.

Apache-2.0 surrogate for Google DeepMind SynthID image watermarks (also used
by OpenAI). Not an official Google detector.

Usage:
  curl -L -o models/opensynthid-detect/model.pt \\
    https://huggingface.co/fyxme/opensynthid-detect-0.1/resolve/main/model.pt
  python3 scripts/convert-opensynthid.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "opensynthid-detect"
CKPT = OUT_DIR / "model.pt"
ONNX_PATH = OUT_DIR / "model.onnx"
Q8_PATH = OUT_DIR / "model_quantized.onnx"
MANIFEST = OUT_DIR / "manifest.json"


class DualStreamWatermarkNet(nn.Module):
    def __init__(
        self,
        spatial_in: int = 4,
        freq_in: int = 2,
        hidden_dim: int = 256,
        backbone: str = "resnet34",
    ):
        super().__init__()
        if backbone == "resnet34":
            self.spatial = models.resnet34(weights=None)
        else:
            self.spatial = models.resnet18(weights=None)
        self.spatial.conv1 = nn.Conv2d(
            spatial_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.spatial.fc = nn.Identity()
        self.freq = nn.Sequential(
            nn.Conv2d(freq_in, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        fusion_dim = 512 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, :4, :, :]
        freq = x[:, 4:, :, :]
        s_feat = self.spatial(spatial)
        f_feat = self.freq(freq).flatten(1)
        fused = torch.cat([s_feat, f_feat], dim=1)
        logit = self.classifier(fused).squeeze(1)
        return torch.sigmoid(logit)


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(
            f"Missing {CKPT}. Download:\n"
            "  curl -L -o models/opensynthid-detect/model.pt "
            "https://huggingface.co/fyxme/opensynthid-detect-0.1/resolve/main/model.pt"
        )

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    backbone = ckpt.get("args", {}).get("backbone", "resnet34")
    channels = int(ckpt.get("channels", 6))
    freq_in = max(1, channels - 4)
    net = DualStreamWatermarkNet(freq_in=freq_in, backbone=backbone)
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, channels, 512, 512)
    torch.onnx.export(
        net,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["probability"],
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(ONNX_PATH), str(Q8_PATH), weight_type=QuantType.QUInt8)

    digest = hashlib.sha256(Q8_PATH.read_bytes()).hexdigest()
    size = Q8_PATH.stat().st_size
    meta = {
        "id": "opensynthid-detect",
        "sha256": digest,
        "bytes": size,
        "inputSize": 512,
        "channels": channels,
        "license": "Apache-2.0",
        "source": "https://huggingface.co/fyxme/opensynthid-detect-0.1",
        "artifact": "model_quantized.onnx",
        "note": "Community SynthID surrogate (Q8) — not Google DeepMind official.",
    }
    MANIFEST.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {Q8_PATH} ({size} bytes)")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
