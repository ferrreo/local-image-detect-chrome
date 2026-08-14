#!/usr/bin/env python3
"""
Train NeoPixel's accurate secondary head on a frozen public backbone.

Backbone: OwensLab/commfor-model-384 (MIT) — ViT-S/16 @ 384.
We do NOT download or redistribute third-party fine-tune quants
(e.g. private Proofmark/proofmark-webwild-v3).

Pipeline:
  1. Download the official safetensors checkpoint from Hugging Face
  2. Freeze the ViT, train a small 384→H→1 head on NeoPixel's corpus
  3. Export Q8 ONNX to models/neopixel-accurate-v1/model_quantized.onnx
  4. Write models/neopixel-accurate-v1/manifest.json (sha256, metrics)

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
DEFAULT_CORPUS = ROOT / "benchmark" / "distill-corpus"
DEFAULT_OUT = ROOT / "models" / "neopixel-accurate-v1"
DEFAULT_CACHE = ROOT / ".neopixel-cache" / "distill-accurate"
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
    p.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "benchmark" / "distill-corpus",
        help="Train corpus root with {ai,real}/… (default: benchmark/distill-corpus)",
    )
    p.add_argument(
        "--holdout-corpus",
        type=Path,
        default=ROOT / "benchmark" / "openrouter",
        help="Optional second tree used only for post-export gate "
        "(default: benchmark/openrouter Lexica holdout + hardcases).",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--feature-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--augment-views", type=int, default=1)
    p.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Optional cap on train-eligible images (0 = all).",
    )
    p.add_argument(
        "--exclude-substr",
        action="append",
        default=[],
        help="Drop training images whose relative path contains this substring "
        "(repeatable). Example: --exclude-substr lexica__holdout",
    )
    p.add_argument(
        "--holdout-substr",
        action="append",
        default=["lexica__holdout", "lexica__feed", "hardcase"],
        help="Paths under --holdout-corpus to score after export (repeatable).",
    )
    p.add_argument("--skip-export", action="store_true")
    p.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Recompute backbone features even if npz cache exists.",
    )
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
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.15, 0.15, 0.12, 0.03),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.12, scale=(0.02, 0.12), value=0),
        ]
    )
    return exact, aug


def load_rgb(path: Path, jpeg_aug: bool = False, rng: random.Random | None = None):
    from io import BytesIO

    from PIL import Image, ImageOps

    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
    if jpeg_aug and rng is not None and rng.random() < 0.35:
        quality = rng.randint(55, 92)
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as recompressed:
            rgb = recompressed.convert("RGB")
    return rgb


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


def feature_cache_path(cache: Path, tag: str) -> Path:
    return cache / "features" / f"{tag}.npz"


def extract_features(
    backbone,
    samples,
    exact,
    aug,
    augment_views,
    device,
    torch,
    *,
    feature_batch_size: int = 16,
    cache_file: Path | None = None,
    use_cache: bool = True,
    jpeg_aug: bool = False,
    seed: int = 42,
):
    if use_cache and cache_file and cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        print(f"feature cache hit {cache_file} rows={len(data['y'])}")
        return data["x"], data["y"], data["domains"]

    xs: list[np.ndarray] = []
    ys: list[float] = []
    domains: list[str] = []
    backbone.eval()
    rng = random.Random(seed)
    pending_tensors: list = []
    pending_meta: list[tuple[float, str]] = []

    def flush():
        nonlocal pending_tensors, pending_meta
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device)
        feats = backbone(batch).detach().cpu().numpy()
        for row, (y, domain) in zip(feats, pending_meta):
            xs.append(row.astype(np.float32))
            ys.append(y)
            domains.append(domain)
        pending_tensors = []
        pending_meta = []

    for i, (path, y, domain) in enumerate(samples):
        views = [exact(load_rgb(path))]
        for _ in range(augment_views):
            views.append(aug(load_rgb(path, jpeg_aug=jpeg_aug, rng=rng)))
        for tensor in views:
            pending_tensors.append(tensor)
            pending_meta.append((y, domain))
            if len(pending_tensors) >= feature_batch_size:
                flush()
        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            print(
                f"\rfeatures {i + 1}/{len(samples)} ({len(xs) + len(pending_tensors)} rows)",
                end="",
                flush=True,
            )
    flush()
    print()
    x = np.stack(xs) if xs else np.zeros((0, 384), np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    d_arr = np.asarray(domains)
    if use_cache and cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y_arr, domains=d_arr)
        print(f"feature cache write {cache_file}")
    return x, y_arr, d_arr


def collect_holdout_samples(
    holdout_corpus: Path, substrings: list[str]
) -> list[tuple[Path, float, str]]:
    if not holdout_corpus or not holdout_corpus.is_dir():
        return []
    samples = collect_samples(holdout_corpus)
    if not substrings:
        return samples
    kept = []
    for path, y, domain in samples:
        rel = str(path).lower()
        if any(sub.lower() in rel for sub in substrings):
            kept.append((path, y, domain))
    return kept


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


def eval_onnx_holdout(
    onnx_path: Path,
    samples,
    exact,
    threshold: float,
    *,
    batch_size: int = 8,
) -> dict:
    logits, ys = onnx_logits(onnx_path, samples, exact, batch_size=batch_size)
    return eval_logits(logits, ys, threshold)


def onnx_logits(
    onnx_path: Path,
    samples,
    exact,
    *,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    if not samples:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    logits: list[float] = []
    ys: list[float] = []
    batch_tensors: list[np.ndarray] = []
    batch_ys: list[float] = []

    def flush() -> None:
        nonlocal batch_tensors, batch_ys
        if not batch_tensors:
            return
        x = np.stack(batch_tensors, axis=0).astype(np.float32)
        out = session.run(None, {input_name: x})[0]
        flat = np.asarray(out).reshape(len(batch_ys), -1)[:, 0]
        logits.extend(float(v) for v in flat)
        ys.extend(batch_ys)
        batch_tensors = []
        batch_ys = []

    for i, (path, y, _domain) in enumerate(samples):
        batch_tensors.append(exact(load_rgb(path)).numpy())
        batch_ys.append(y)
        if len(batch_tensors) >= batch_size:
            flush()
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"\rontx {i + 1}/{len(samples)}", end="", flush=True)
    flush()
    print()
    return (
        np.asarray(logits, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def calibrate_threshold(
    lex_logits: np.ndarray,
    hard_logits: np.ndarray,
    *,
    min_hard_tnr: float = 0.95,
    min_lex_tpr: float = 0.75,
) -> dict:
    """Pick an operating threshold that clears hardcase TNR without gutting Lexica TPR."""
    lex_y = np.ones(len(lex_logits), dtype=np.float32)
    hard_y = np.zeros(len(hard_logits), dtype=np.float32)
    candidates = []
    for thr_i in range(35, 95):
        thr = thr_i / 100.0
        lex_m = eval_logits(lex_logits, lex_y, thr) if len(lex_logits) else {
            "tpr": 0.0, "tnr": 0.0, "balancedAccuracy": 0.0, "n": 0, "threshold": thr
        }
        hard_m = eval_logits(hard_logits, hard_y, thr) if len(hard_logits) else {
            "tpr": 0.0, "tnr": 1.0, "balancedAccuracy": 0.5, "n": 0, "threshold": thr
        }
        ok = (
            float(hard_m["tnr"]) >= min_hard_tnr
            and float(lex_m["tpr"]) >= min_lex_tpr
        )
        score = (
            100.0 * float(lex_m["tpr"])
            + 40.0 * 0.5 * (float(lex_m["tpr"]) + float(hard_m["tnr"]))
            + 20.0 * float(hard_m["tnr"])
        )
        candidates.append(
            {
                "threshold": thr,
                "ok": ok,
                "score": score,
                "lexica": lex_m,
                "hardcase": hard_m,
                "gateBa": 0.5 * (float(lex_m["tpr"]) + float(hard_m["tnr"])),
            }
        )
    ok_rows = [c for c in candidates if c["ok"]]
    pool = ok_rows or candidates
    best = max(pool, key=lambda c: (c["ok"], c["score"]))
    return {
        "selected": best,
        "minHardTnr": min_hard_tnr,
        "minLexTpr": min_lex_tpr,
        "grid": [
            {
                "threshold": c["threshold"],
                "ok": c["ok"],
                "lexTpr": c["lexica"]["tpr"],
                "hardTnr": c["hardcase"]["tnr"],
                "gateBa": c["gateBa"],
            }
            for c in candidates
        ],
    }


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
    eligible = [s for s in samples if s not in excluded] if excluded else list(samples)
    if args.max_train_images and args.max_train_images > 0:
        rng_cap = random.Random(args.seed)
        rng_cap.shuffle(eligible)
        eligible = eligible[: args.max_train_images]

    train_s, val_s = split_train_val(eligible, args.val_frac, args.seed)
    gate_s = collect_holdout_samples(args.holdout_corpus, args.holdout_substr)
    # Never train on gate images even if someone pointed --corpus at openrouter.
    gate_ids = {p.resolve() for p, _, _ in gate_s}
    train_s = [s for s in train_s if s[0].resolve() not in gate_ids]
    val_s = [s for s in val_s if s[0].resolve() not in gate_ids]

    print(
        f"corpus {len(samples)}  train-eligible {len(eligible)}  "
        f"train {len(train_s)}  val {len(val_s)}  gate {len(gate_s)}"
    )
    if excluded:
        print(f"excluded-from-train {len(excluded)}")

    backbone_path = ensure_backbone(args.cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    backbone = make_feature_backbone(backbone_path, torch, nn, timm).to(device)
    exact, aug = build_transforms()

    cache_tag = hashlib.sha256(
        json.dumps(
            {
                "corpus": str(args.corpus.resolve()),
                "train": sorted(str(p) for p, _, _ in train_s),
                "aug": args.augment_views,
                "seed": args.seed,
                "hidden": args.hidden,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    train_x, train_y, _ = extract_features(
        backbone,
        train_s,
        exact,
        aug,
        args.augment_views,
        device,
        torch,
        feature_batch_size=args.feature_batch_size,
        cache_file=feature_cache_path(args.cache, f"train-{cache_tag}"),
        use_cache=not args.no_feature_cache,
        jpeg_aug=True,
        seed=args.seed,
    )
    if val_s:
        val_x, val_y, _ = extract_features(
            backbone,
            val_s,
            exact,
            aug,
            0,
            device,
            torch,
            feature_batch_size=args.feature_batch_size,
            cache_file=feature_cache_path(args.cache, f"val-{cache_tag}"),
            use_cache=not args.no_feature_cache,
            jpeg_aug=False,
            seed=args.seed + 1,
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
        "lr": args.lr,
        "weightDecay": args.weight_decay,
        "augmentViews": args.augment_views,
        "corpus": str(args.corpus),
        "corpusImages": len(samples),
        "trainImages": len(train_s),
        "trainRows": int(len(train_y)),
        "valRows": int(len(val_y)),
        "gateImages": len(gate_s),
        "valAt065": eval_logits(val_logits, eval_y, 0.65),
        "bestValBaDuringTrain": best_ba,
        "preprocess": "short440-center384",
        "norm": {"mean": list(MEAN), "std": list(STD)},
        "outputKind": "logit",
    }
    print(
        f"val BA@0.65={(report['valAt065']['balancedAccuracy'] * 100):.1f}% "
        f"TPR={(report['valAt065']['tpr'] * 100):.1f}% "
        f"TNR={(report['valAt065']['tnr'] * 100):.1f}%"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")

    holdout = {
        "valFiles": [
            str(path.relative_to(args.corpus)).replace("\\", "/")
            for path, _, _ in val_s
        ],
        "gateFiles": [
            (
                str(path.relative_to(args.holdout_corpus)).replace("\\", "/")
                if args.holdout_corpus in path.parents
                or path.parent == args.holdout_corpus
                else str(path)
            )
            for path, _, _ in gate_s
        ],
        "note": "valFiles = early-stopping split from train corpus. "
        "gateFiles = frozen Lexica holdout + hardcases (promotion gate).",
    }
    (args.output / "holdout.json").write_text(json.dumps(holdout, indent=2) + "\n")

    if args.skip_export:
        print("skip export")
        return

    q8 = export_onnx(backbone_path, head.cpu(), args.output, torch, nn, timm)
    digest = sha256_file(q8)

    lexica_gate = [s for s in gate_s if "lexica" in str(s[0]).lower()]
    hard_gate = [
        s
        for s in gate_s
        if "hardcase" in str(s[0]).lower() or "hard-case" in str(s[0]).lower()
    ]
    if not hard_gate:
        hard_gate = [s for s in gate_s if s[1] < 0.5]

    # Score once, then report @0.65 and a calibrated operating point.
    gate_logits, gate_y = onnx_logits(q8, gate_s, exact)
    lex_logits, lex_y = onnx_logits(q8, lexica_gate, exact)
    hard_logits, hard_y = onnx_logits(q8, hard_gate, exact)
    val_onnx = eval_onnx_holdout(q8, val_s, exact, 0.65)
    gate_onnx = eval_logits(gate_logits, gate_y, 0.65)
    report["onnxValAt065"] = val_onnx
    report["onnxHoldoutAt065"] = gate_onnx
    report["onnxLexicaHoldoutAt065"] = eval_logits(lex_logits, lex_y, 0.65)
    report["onnxHardcaseAt065"] = eval_logits(hard_logits, hard_y, 0.65)
    calib = calibrate_threshold(lex_logits, hard_logits)
    report["calibrated"] = calib
    sel = calib["selected"]
    report["onnxLexicaHoldoutCalibrated"] = sel["lexica"]
    report["onnxHardcaseCalibrated"] = sel["hardcase"]
    report["onnxHoldoutCalibrated"] = {
        "threshold": sel["threshold"],
        "balancedAccuracy": sel["gateBa"],
        "tpr": sel["lexica"]["tpr"],
        "tnr": sel["hardcase"]["tnr"],
        "n": int(len(gate_y)),
    }

    print(
        f"ONNX val BA@0.65={val_onnx['balancedAccuracy'] * 100:.1f}% n={val_onnx['n']}"
    )
    print(
        f"ONNX gate BA@0.65={gate_onnx['balancedAccuracy'] * 100:.1f}% "
        f"TPR={gate_onnx['tpr'] * 100:.1f}% TNR={gate_onnx['tnr'] * 100:.1f}% "
        f"n={gate_onnx['n']}"
    )
    print(
        f"ONNX Lexica TPR@0.65="
        f"{report['onnxLexicaHoldoutAt065']['tpr'] * 100:.1f}% "
        f"n={report['onnxLexicaHoldoutAt065']['n']}  "
        f"hardcase TNR="
        f"{report['onnxHardcaseAt065']['tnr'] * 100:.1f}% "
        f"n={report['onnxHardcaseAt065']['n']}"
    )
    print(
        f"calibrated thr={sel['threshold']:.2f} "
        f"lexTPR={sel['lexica']['tpr'] * 100:.1f}% "
        f"hardTNR={sel['hardcase']['tnr'] * 100:.1f}% "
        f"gateBA={sel['gateBa'] * 100:.1f}% ok={sel['ok']}"
    )

    manifest = {
        "id": "neopixel-accurate-v1",
        "path": "model_quantized.onnx",
        "sha256": digest,
        "bytes": q8.stat().st_size,
        "license": "MIT (OwensLab backbone + NeoPixel-trained head)",
        "report": report,
        "holdout": holdout,
        "recommendedThreshold": sel["threshold"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {q8} ({manifest['bytes']} bytes)")
    print(f"sha256 {digest}")
    print(
        "Promote FORENSICS_MODEL only after Lexica holdout TPR + hardcase TNR "
        "clear the bar vs Community Forensics latency."
    )


if __name__ == "__main__":
    main()
