#!/usr/bin/env python3
"""
Evaluate TruthLens-related AI-image detectors on the OpenRouter + Lexica corpus.

Sources (public HF):
  - umm-maybe/AI-image-detector — Swin classifier used by
    Pikachu771/truthlens-ai-image-detector (labels: artificial / human)
  - Medsa/ai-image-authenticity-detector — TorchScript multi-branch CNN
    marketed as "TruthLens — AI Image Authenticity Detector" (32×32)

Most other HF "TruthLens" repos are text/fake-news classifiers, not image heads.

Usage:
  python3 -m pip install transformers pillow torch torchvision huggingface_hub
  npm run eval:truthlens
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmark" / "openrouter"
OUT_DIR = ROOT / "benchmark" / "model-survey"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--models",
        default="umm-maybe,medsa",
        help="Comma list: umm-maybe,medsa",
    )
    return p.parse_args()


def collect(corpus: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    for label in ("ai", "real"):
        root = corpus / label
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = str(path.relative_to(corpus)).replace("\\", "/")
            rows.append(
                {
                    "path": path,
                    "file": rel,
                    "label": label,
                    "hardCase": "/hardcases/" in rel.replace("\\", "/"),
                    "lexica": "lexica" in rel.lower(),
                }
            )
    if limit > 0:
        ai = [r for r in rows if r["label"] == "ai"]
        real = [r for r in rows if r["label"] == "real"]
        hard = [r for r in real if r["hardCase"]]
        each = max(1, limit // 2)
        real_pick = (hard + [r for r in real if not r["hardCase"]])[:each]
        rows = (ai[:each] + real_pick)[:limit]
    return rows


def summarize(rows: list[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    hard_fp = hard_n = 0
    lex_tp = lex_n = 0
    total_ms = 0.0
    for r in rows:
        total_ms += r["totalMs"]
        pred = r["confidence"] >= threshold
        actual = r["label"] == "ai"
        if actual and pred:
            tp += 1
        elif not actual and not pred:
            tn += 1
        elif not actual and pred:
            fp += 1
        else:
            fn += 1
        if r["hardCase"]:
            hard_n += 1
            if pred:
                hard_fp += 1
        if r["lexica"]:
            lex_n += 1
            if pred:
                lex_tp += 1
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "balancedAccuracy": (tpr + tnr) / 2,
        "tpr": tpr,
        "tnr": tnr,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "avgMsPerImage": total_ms / len(rows) if rows else 0.0,
        "totalMs": total_ms,
        "hardCaseFp": hard_fp,
        "hardCaseN": hard_n,
        "lexicaTp": lex_tp,
        "lexicaN": lex_n,
        "lexicaTpr": (lex_tp / lex_n) if lex_n else None,
    }


def eval_umm_maybe(images: list[dict]) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_id = "umm-maybe/AI-image-detector"
    print(f"load {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Warmup
    warm = Image.open(images[0]["path"]).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=warm, return_tensors="pt").to(device)
        model(**inputs)

    rows = []
    t0 = time.perf_counter()
    for i, image in enumerate(images):
        pil = Image.open(image["path"]).convert("RGB")
        p0 = time.perf_counter()
        inputs = processor(images=pil, return_tensors="pt").to(device)
        preprocess_ms = (time.perf_counter() - p0) * 1000
        i0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            # id2label: 0=artificial, 1=human
            conf = float(probs[0].item())
        infer_ms = (time.perf_counter() - i0) * 1000
        rows.append(
            {
                **image,
                "confidence": conf,
                "preprocessMs": preprocess_ms,
                "inferMs": infer_ms,
                "totalMs": preprocess_ms + infer_ms,
            }
        )
        if (i + 1) % 10 == 0 or i + 1 == len(images):
            print(f"\rumm-maybe {i + 1}/{len(images)}", end="", flush=True)
    print()
    wall = (time.perf_counter() - t0) * 1000
    return {
        "id": "umm-maybe/AI-image-detector",
        "role": "TruthLens Pikachu771 backend (Swin)",
        "hf": model_id,
        "imageCount": len(rows),
        "wallMs": wall,
        "at065": summarize(rows, 0.65),
        "atProduct": summarize(rows, 0.6951),
        "soylent": _public_row(next((r for r in rows if "soylent" in r["file"]), None)),
    }


def _public_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {k: v for k, v in row.items() if k != "path"}


def eval_medsa(images: list[dict]) -> dict:
    import torch
    import torchvision.transforms as T
    from huggingface_hub import hf_hub_download
    from PIL import Image

    model_id = "Medsa/ai-image-authenticity-detector"
    print(f"load {model_id}")
    path = hf_hub_download(model_id, "detector_scripted.pt")
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    warm = transform(Image.open(images[0]["path"]).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        model(warm)

    rows = []
    t0 = time.perf_counter()
    for i, image in enumerate(images):
        pil = Image.open(image["path"]).convert("RGB")
        p0 = time.perf_counter()
        tensor = transform(pil).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - p0) * 1000
        i0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor)
            logit = out[0] if isinstance(out, (tuple, list)) else out
            conf = float(torch.sigmoid(logit.reshape(-1)[0]).item())
        infer_ms = (time.perf_counter() - i0) * 1000
        rows.append(
            {
                **image,
                "confidence": conf,
                "preprocessMs": preprocess_ms,
                "inferMs": infer_ms,
                "totalMs": preprocess_ms + infer_ms,
            }
        )
        if (i + 1) % 10 == 0 or i + 1 == len(images):
            print(f"\rmedsa {i + 1}/{len(images)}", end="", flush=True)
    print()
    wall = (time.perf_counter() - t0) * 1000
    return {
        "id": "Medsa/ai-image-authenticity-detector",
        "role": "TruthLens-branded TorchScript CNN (32×32 CIFAKE)",
        "hf": model_id,
        "imageCount": len(rows),
        "wallMs": wall,
        "at065": summarize(rows, 0.65),
        "atProduct": summarize(rows, 0.6951),
        "soylent": _public_row(next((r for r in rows if "soylent" in r["file"]), None)),
    }


def write_report(results: list[dict], images: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        results,
        key=lambda r: (
            -r["at065"]["balancedAccuracy"],
            r["at065"]["avgMsPerImage"],
        ),
    )
    payload = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpusImages": len(images),
        "hardCases": sum(1 for i in images if i["hardCase"]),
        "lexicaImages": sum(1 for i in images if i["lexica"]),
        "note": (
            "TruthLens name is overloaded on HF. Image-capable public weights "
            "evaluated here: umm-maybe (used by Pikachu771 TruthLens server) and "
            "Medsa's TruthLens-branded TorchScript detector. Most other "
            "TruthLens repos are text/fake-news classifiers."
        ),
        "rankingAt065": [
            {
                "rank": i + 1,
                "id": r["id"],
                "ba": r["at065"]["balancedAccuracy"],
                "tpr": r["at065"]["tpr"],
                "tnr": r["at065"]["tnr"],
                "lexicaTpr": r["at065"]["lexicaTpr"],
                "avgMsPerImage": r["at065"]["avgMsPerImage"],
                "hardCaseFp": r["at065"]["hardCaseFp"],
                "role": r["role"],
            }
            for i, r in enumerate(ranked)
        ],
        "models": results,
    }
    def _jsonable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items() if k != "path"}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    json_path = OUT_DIR / "truthlens-compare.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    md = []
    md.append("# TruthLens-related image detector compare")
    md.append("")
    md.append(
        f"Generated `{payload['generatedAt']}` · **{payload['corpusImages']}** images · "
        f"**{payload['hardCases']}** hardcases · **{payload['lexicaImages']}** Lexica AI"
    )
    md.append("")
    md.append(payload["note"])
    md.append("")
    md.append("## Ranking @ 65%")
    md.append("")
    md.append(
        "| Rank | Model | BA | TPR | TNR | Lexica TPR | Avg ms | Hardcase FP | Role |"
    )
    md.append(
        "|-----:|-------|---:|----:|----:|-----------:|-------:|------------:|------|"
    )
    for r in payload["rankingAt065"]:
        full = next(x for x in results if x["id"] == r["id"])
        lex = (
            "—"
            if r["lexicaTpr"] is None
            else f"{r['lexicaTpr'] * 100:.1f}% ({full['at065']['lexicaTp']}/{full['at065']['lexicaN']})"
        )
        md.append(
            f"| {r['rank']} | `{r['id']}` | {r['ba'] * 100:.1f}% | {r['tpr'] * 100:.1f}% | "
            f"{r['tnr'] * 100:.1f}% | {lex} | {r['avgMsPerImage']:.1f} | "
            f"{r['hardCaseFp']}/{full['at065']['hardCaseN']} | {r['role']} |"
        )
    md.append("")
    md.append("## Product threshold @ 69.51%")
    md.append("")
    md.append("| Model | BA | Avg ms | Hardcase FP |")
    md.append("|-------|---:|-------:|------------:|")
    for r in ranked:
        md.append(
            f"| `{r['id']}` | {r['atProduct']['balancedAccuracy'] * 100:.1f}% | "
            f"{r['atProduct']['avgMsPerImage']:.1f} | "
            f"{r['atProduct']['hardCaseFp']}/{r['atProduct']['hardCaseN']} |"
        )
    md.append("")
    md.append("## Soylent hardcase proxy")
    md.append("")
    md.append("| Model | Confidence | @65% | @69.51% | ms |")
    md.append("|-------|-----------:|------|---------|---:|")
    for r in ranked:
        s = r["soylent"]
        if not s:
            md.append(f"| `{r['id']}` | — | — | — | — |")
            continue
        md.append(
            f"| `{r['id']}` | {s['confidence'] * 100:.1f}% | "
            f"{'AI' if s['confidence'] >= 0.65 else 'real/other'} | "
            f"{'AI' if s['confidence'] >= 0.6951 else 'real/other'} | "
            f"{s['totalMs']:.1f} |"
        )
    md.append("")
    md_path = OUT_DIR / "truthlens-compare.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    wanted = {s.strip() for s in args.models.split(",") if s.strip()}
    images = collect(args.corpus, args.limit)
    print(
        f"Corpus: {len(images)} images "
        f"({sum(1 for i in images if i['hardCase'])} hardcases, "
        f"{sum(1 for i in images if i['lexica'])} lexica)"
    )
    results = []
    if "umm-maybe" in wanted:
        results.append(eval_umm_maybe(images))
        print(
            f"umm-maybe BA@0.65={results[-1]['at065']['balancedAccuracy'] * 100:.1f}% "
            f"avg={results[-1]['at065']['avgMsPerImage']:.1f}ms"
        )
    if "medsa" in wanted:
        results.append(eval_medsa(images))
        print(
            f"medsa BA@0.65={results[-1]['at065']['balancedAccuracy'] * 100:.1f}% "
            f"avg={results[-1]['at065']['avgMsPerImage']:.1f}ms"
        )
    if not results:
        raise SystemExit(f"No models selected from {wanted}")
    write_report(results, images)


if __name__ == "__main__":
    main()
