#!/usr/bin/env python3
"""
Train TruePixel's accurate secondary head on a frozen public backbone.

Backbone: OwensLab/commfor-model-384 (MIT) — ViT-S/16 @ 384.
We do NOT download or redistribute third-party fine-tune quants
(e.g. private Proofmark/proofmark-webwild-v3).

Pipeline:
  1. Download the official safetensors checkpoint from Hugging Face
  2. Freeze the ViT, train a small 384→H→1 head on TruePixel's corpus
  3. Export Q8 ONNX to models/truepixel-accurate-v1/model_quantized.onnx
  4. Write models/truepixel-accurate-v1/manifest.json (sha256, metrics)

Promote into the extension by pointing FORENSICS_MODEL at this artifact
once held-out BA clears the bounty bar.

Usage:
  python3 -m pip install -r scripts/requirements-distill.txt
  npm run distill:accurate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmark" / "openrouter"
DEFAULT_OUT = ROOT / "models" / "truepixel-accurate-v1"
DEFAULT_CACHE = ROOT / ".truepixel-cache" / "distill-accurate"
HF_REPO = "OwensLab/commfor-model-384"
HF_FILE = "model.safetensors"
EXPECTED_BACKBONE_SHA256 = (
    "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--augment-views", type=int, default=1)
    p.add_argument("--skip-export", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_samples(corpus: Path) -> list[tuple[Path, float, str]]:
    samples: list[tuple[Path, float, str]] = []
    for label, y in (("ai", 1.0), ("real", 0.0)):
        root = corpus / label
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = path.relative_to(root)
            domain = rel.parts[0] if rel.parts else "unknown"
            domain = re.sub(r"[^a-z0-9._-]+", "-", domain.lower())
            samples.append((path, y, domain))
    if not samples:
        raise SystemExit(f"No images under {corpus}/{{ai,real}}")
    return samples


def split_train_val(
    samples: list[tuple[Path, float, str]], val_frac: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[float, list] = {0.0: [], 1.0: []}
    for item in samples:
        by_label[item[1]].append(item)
    train: list = []
    val: list = []
    for bucket in by_label.values():
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_frac))) if len(bucket) > 4 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def ensure_backbone(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / HF_FILE
    if dest.exists() and sha256_file(dest) == EXPECTED_BACKBONE_SHA256:
        print(f"backbone cached {dest}")
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install distill deps: python3 -m pip install -r scripts/requirements-distill.txt"
        ) from exc
    print(f"download {HF_REPO}/{HF_FILE}")
    downloaded = Path(
        hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=str(cache))
    )
    if downloaded.resolve() != dest.resolve():
        dest.write_bytes(downloaded.read_bytes())
    digest = sha256_file(dest)
    if digest != EXPECTED_BACKBONE_SHA256:
        raise SystemExit(
            f"Backbone checksum mismatch: expected {EXPECTED_BACKBONE_SHA256}, got {digest}"
        )
    return dest


def build_transforms():
    import torchvision.transforms as T

    exact = T.Compose(
        [
            T.Resize(440, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    aug = T.Compose(
        [
            T.RandomResizedCrop(
                384,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.12),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.12, 0.12, 0.1, 0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    return exact, aug


def load_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_vit_state(checkpoint: Path) -> dict:
    from safetensors.torch import load_file

    raw = load_file(str(checkpoint))
    # Accept either bare timm keys or vit.* prefixed CommFor dumps.
    if any(k.startswith("vit.") for k in raw):
        state = {k[4:]: v for k, v in raw.items() if k.startswith("vit.")}
        if "head.weight" in raw:
            state["head.weight"] = raw["head.weight"]
        if "head.bias" in raw:
            state["head.bias"] = raw["head.bias"]
        return state
    return raw


def make_feature_backbone(checkpoint: Path, torch, nn, timm):
    class FeatureBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(
                "vit_small_patch16_384.augreg_in21k_ft_in1k",
                pretrained=False,
                num_classes=0,
            )
            state = load_vit_state(checkpoint)
            feature_state = {
                k: v for k, v in state.items() if not k.startswith("head.")
            }
            missing, unexpected = self.vit.load_state_dict(feature_state, strict=False)
            if missing:
                print(f"backbone missing keys: {len(missing)} (ok if head-only)")
            if unexpected:
                print(f"backbone unexpected keys: {unexpected[:8]}")
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        @torch.inference_mode()
        def forward(self, x):
            return self.vit(x)

    return FeatureBackbone()


def make_head(hidden: int, nn):
    return nn.Sequential(nn.Linear(384, hidden), nn.GELU(), nn.Linear(hidden, 1))


def extract_features(backbone, samples, exact, aug, augment_views, device, torch):
    xs: list[np.ndarray] = []
    ys: list[float] = []
    domains: list[str] = []
    backbone.eval()
    for i, (path, y, domain) in enumerate(samples):
        images = [exact(load_rgb(path))]
        for _ in range(augment_views):
            images.append(aug(load_rgb(path)))
        batch = torch.stack(images).to(device)
        feats = backbone(batch).detach().cpu().numpy()
        for row in feats:
            xs.append(row.astype(np.float32))
            ys.append(y)
            domains.append(domain)
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            print(f"\rfeatures {i + 1}/{len(samples)} ({len(xs)} rows)", end="", flush=True)
    print()
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(domains)


def eval_logits(logits: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    scores = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    pred = scores >= threshold
    ai = y >= 0.5
    real = ~ai
    tpr = float(pred[ai].mean()) if ai.any() else 0.0
    tnr = float((~pred[real]).mean()) if real.any() else 0.0
    return {
        "threshold": threshold,
        "balancedAccuracy": (tpr + tnr) / 2,
        "tpr": tpr,
        "tnr": tnr,
        "n": int(len(y)),
    }


def train_head(head, train_x, train_y, val_x, val_y, args, device, torch, nn):
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    best_state = None
    best_ba = -1.0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(x), args.batch_size):
            idx = perm[start : start + args.batch_size]
            logits = head(x[idx]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            steps += 1
        head.eval()
        with torch.no_grad():
            eval_x = val_x if len(val_x) else train_x
            eval_y = val_y if len(val_y) else train_y
            logits = (
                head(torch.from_numpy(eval_x).to(device)).squeeze(-1).cpu().numpy()
            )
        metrics = eval_logits(logits, eval_y, 0.65)
        if metrics["balancedAccuracy"] > best_ba:
            best_ba = metrics["balancedAccuracy"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:3d}  loss={total_loss / max(1, steps):.4f}  "
                f"valBA@0.65={metrics['balancedAccuracy'] * 100:.1f}%  "
                f"TPR={metrics['tpr'] * 100:.1f}% TNR={metrics['tnr'] * 100:.1f}%"
            )
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_ba


def eval_onnx_holdout(onnx_path: Path, samples, exact, threshold: float) -> dict:
    import onnxruntime as ort

    if not samples:
        return eval_logits(np.zeros((0,), np.float32), np.zeros((0,), np.float32), threshold)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    logits = []
    ys = []
    for path, y, _domain in samples:
        tensor = exact(load_rgb(path)).numpy()[None, ...]
        out = session.run(None, {input_name: tensor.astype(np.float32)})[0]
        logits.append(float(np.asarray(out).reshape(-1)[0]))
        ys.append(y)
    return eval_logits(np.asarray(logits, dtype=np.float32), np.asarray(ys, dtype=np.float32), threshold)


def export_onnx(checkpoint: Path, head, out_dir: Path, torch, nn, timm):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class ExportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(
                "vit_small_patch16_384.augreg_in21k_ft_in1k",
                pretrained=False,
            )
            state = {
                k: v
                for k, v in load_vit_state(checkpoint).items()
                if not k.startswith("head.")
            }
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            # head.* missing is expected — we install our trained head next.
            other_missing = [k for k in missing if not k.startswith("head.")]
            if other_missing:
                print(f"export missing: {other_missing}")
            if unexpected:
                print(f"export unexpected: {unexpected[:8]}")
            self.vit.head = head

        def forward(self, pixel_values):
            return self.vit(pixel_values)

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ExportModel().eval()
    fp32 = out_dir / "model_fp32.onnx"
    q8 = out_dir / "model_quantized.onnx"
    dummy = torch.zeros(1, 3, 384, 384)
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    quantize_dynamic(
        str(fp32),
        str(q8),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    fp32.unlink(missing_ok=True)
    return q8


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
        import timm
    except ImportError as exc:
        raise SystemExit(
            "Missing distill deps. Run:\n"
            "  python3 -m pip install -r scripts/requirements-distill.txt"
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = collect_samples(args.corpus)
    excluded = [
        s
        for s in samples
        if any(sub.lower() in str(s[0]).lower() for sub in args.exclude_substr)
    ]
    eligible = [s for s in samples if s not in excluded] if excluded else samples
    train_s, val_s = split_train_val(eligible, args.val_frac, args.seed)
    if excluded:
        # Domain holdout: evaluate excluded paths after export (not used in train).
        val_s = excluded
        print(
            f"corpus {len(samples)}  train-eligible {len(eligible)}  "
            f"excluded-holdout {len(excluded)}  train {len(train_s)}"
        )
    else:
        print(f"corpus {len(samples)}  train {len(train_s)}  val {len(val_s)}")

    backbone_path = ensure_backbone(args.cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    backbone = make_feature_backbone(backbone_path, torch, nn, timm).to(device)
    exact, aug = build_transforms()

    train_x, train_y, _ = extract_features(
        backbone, train_s, exact, aug, args.augment_views, device, torch
    )
    if val_s:
        val_x, val_y, _ = extract_features(
            backbone, val_s, exact, aug, 0, device, torch
        )
    else:
        val_x = np.zeros((0, 384), np.float32)
        val_y = np.zeros((0,), np.float32)

    head = make_head(args.hidden, nn).to(device)
    best_ba = train_head(
        head, train_x, train_y, val_x, val_y, args, device, torch, nn
    )
    head.eval()
    with torch.no_grad():
        eval_x = val_x if len(val_x) else train_x
        eval_y = val_y if len(val_y) else train_y
        val_logits = head(torch.from_numpy(eval_x).to(device)).squeeze(-1).cpu().numpy()
    report = {
        "backbone": HF_REPO,
        "backboneSha256": EXPECTED_BACKBONE_SHA256,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "corpusImages": len(samples),
        "trainRows": int(len(train_y)),
        "valRows": int(len(val_y)),
        "valAt065": eval_logits(val_logits, eval_y, 0.65),
        "bestValBaDuringTrain": best_ba,
        "preprocess": "short440-center384",
        "norm": {"mean": list(MEAN), "std": list(STD)},
        "outputKind": "logit",
    }
    print(
        f"held-out BA@0.65={(report['valAt065']['balancedAccuracy'] * 100):.1f}% "
        f"TPR={(report['valAt065']['tpr'] * 100):.1f}% "
        f"TNR={(report['valAt065']['tnr'] * 100):.1f}%"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")

    holdout = {
        "files": [
            str(path.relative_to(args.corpus)).replace("\\", "/")
            for path, _, _ in val_s
        ],
        "note": "Images held out of head training (still used for early-stopping).",
    }
    (args.output / "holdout.json").write_text(json.dumps(holdout, indent=2) + "\n")

    if args.skip_export:
        print("skip export")
        return

    q8 = export_onnx(backbone_path, head.cpu(), args.output, torch, nn, timm)
    digest = sha256_file(q8)

    # Honest check: ORT on the held-out files only (not the train rows).
    holdout_metrics = eval_onnx_holdout(q8, val_s, exact, 0.65)
    report["onnxHoldoutAt065"] = holdout_metrics
    print(
        f"ONNX holdout BA@0.65={holdout_metrics['balancedAccuracy'] * 100:.1f}% "
        f"TPR={holdout_metrics['tpr'] * 100:.1f}% "
        f"TNR={holdout_metrics['tnr'] * 100:.1f}% "
        f"n={holdout_metrics['n']}"
    )
    print(
        "Warning: npm run eval:compare on the full OpenRouter tree includes "
        "train images — use holdout.json / onnxHoldoutAt065 for promotion decisions."
    )

    manifest = {
        "id": "truepixel-accurate-v1",
        "path": "model_quantized.onnx",
        "sha256": digest,
        "bytes": q8.stat().st_size,
        "license": "MIT (OwensLab backbone + TruePixel-trained head)",
        "report": report,
        "holdout": holdout,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {q8} ({manifest['bytes']} bytes)")
    print(f"sha256 {digest}")
    print(
        "Next: point FORENSICS_MODEL at this artifact in "
        "extension/src/lib/model-manifest.ts only after holdout BA clears the bar "
        "on a larger shard."
    )


if __name__ == "__main__":
    main()
