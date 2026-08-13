#!/usr/bin/env node
/**
 * Compare candidate ONNX AI-image detectors on the OpenRouter + hardcase corpus.
 *
 * Usage: node scripts/eval-model-compare.mjs
 * Env:
 *   EVAL_COMPARE_LIMIT=N   balanced subset (0 = all)
 *   EVAL_COMPARE_MODELS=id1,id2
 */
import { createHash } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  renameSync,
} from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const require = createRequire(import.meta.url);
const ort = require("onnxruntime-node");

/** Requested shortlist + substitutes when HF artifacts are missing. */
const MODELS = [
  {
    id: "ai-human-generated-image-detection-ONNX",
    requested: true,
    status: "unavailable",
    reason:
      "Hugging Face repo has no weights (only .gitattributes). Substituted by zonn-ai/ai-image-detection-ONNX.",
    substituteFor: null,
  },
  {
    id: "deepguard-ai",
    requested: true,
    status: "unavailable",
    reason:
      "Repo ships EfficientNet .pth + inswapper face-swap ONNX, not a classifier ONNX. Substituted by current distilled.",
    substituteFor: null,
  },
  {
    id: "ai-image-detection-ONNX",
    requested: true,
    status: "ok",
    hf: "onnx-community/ai-image-detection-ONNX",
    url: "https://huggingface.co/onnx-community/ai-image-detection-ONNX/resolve/main/onnx/model_fp16.onnx",
    localPath: "models/compare/ai-image-detection/model_fp16.onnx",
    inputSize: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    aiLabelIndex: 1,
    outputKind: "logits2",
    preprocess: "stretch",
  },
  {
    id: "Deep-Fake-Detector-v2-Model-ONNX",
    requested: true,
    status: "ok",
    hf: "onnx-community/Deep-Fake-Detector-v2-Model-ONNX",
    url: "https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX/resolve/main/onnx/model_fp16.onnx",
    localPath: "models/compare/deepfake-detector-v2/model_fp16.onnx",
    inputSize: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    aiLabelIndex: 1,
    outputKind: "logits2",
    preprocess: "stretch",
  },
  {
    id: "sdxl-detector",
    requested: true,
    status: "ok",
    hf: "Organika/sdxl-detector",
    url: "https://huggingface.co/Organika/sdxl-detector/resolve/main/onnx/model.onnx",
    localPath: "models/compare/sdxl-detector/model.onnx",
    inputSize: 224,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    aiLabelIndex: 0, // artificial
    outputKind: "logits2",
    preprocess: "stretch",
  },
  {
    id: "detectra-v1",
    requested: true,
    status: "ok",
    hf: "Vontra/detectra-v1",
    url: "https://huggingface.co/Vontra/detectra-v1/resolve/main/model.onnx",
    localPath: "models/compare/detectra-v1/model.onnx",
    sha256:
      "1414b9aafaa01a644ed706224973f09b53a1388282104409862df9893b1b962b",
    inputSize: 384,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    aiLabelIndex: 0,
    outputKind: "logit",
    preprocess: "short440-center384",
  },
  {
    id: "zonn-ai-image-detection-ONNX",
    requested: false,
    status: "ok",
    substituteFor: "ai-human-generated-image-detection-ONNX",
    hf: "zonn-ai/ai-image-detection-ONNX",
    url: "https://huggingface.co/zonn-ai/ai-image-detection-ONNX/resolve/main/onnx/model.onnx",
    localPath: "models/compare/zonn-ai-image-detection/model.onnx",
    inputSize: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    aiLabelIndex: 1,
    outputKind: "logits2",
    preprocess: "stretch",
  },
  {
    id: "ai-image-detect-distilled",
    requested: false,
    status: "ok",
    substituteFor: "deepguard-ai",
    hf: "onnx-community/ai-image-detect-distilled-ONNX",
    url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model_fp16.onnx",
    localPath: "models/ai-image-detect-distilled/model_fp16.onnx",
    sha256:
      "9594bacb70d9c65fcaa656e0d17038c5cac7a6c48d04cd10f2ebf972a01ba3ee",
    inputSize: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    aiLabelIndex: 0, // fake
    outputKind: "logits2",
    preprocess: "stretch",
  },
  {
    // Private HF fine-tune — do not hotlink / redistribute third-party quants.
    // Train ours via `npm run distill:accurate` instead.
    id: "proofmark-webwild-v3",
    requested: false,
    status: "unavailable",
    reason:
      "Proofmark/proofmark-webwild-v3 is private on HF; we do not fetch or ship third-party bundled quants. Distill our own accurate head instead.",
    substituteFor: null,
  },
  {
    id: "truepixel-accurate-v1",
    requested: false,
    status: "ok",
    hf: "local distill (OwensLab/commfor-model-384 backbone + TruePixel head)",
    url: null,
    localPath: "models/truepixel-accurate-v1/model_quantized.onnx",
    seedPaths: [],
    // Filled by npm run distill:accurate → models/truepixel-accurate-v1/manifest.json
    sha256: null,
    inputSize: 384,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    aiLabelIndex: 0,
    outputKind: "logit",
    preprocess: "short440-center384",
  },
];

function softmax2(a, b) {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  return eb / (ea + eb); // prob of class 1 when used carefully — caller picks index
}

function softmaxPick(logits, aiIndex) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps[aiIndex] / sum;
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function digestFile(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

async function download(model) {
  const outPath = path.join(root, model.localPath);
  mkdirSync(path.dirname(outPath), { recursive: true });
  if (existsSync(outPath)) {
    if (model.sha256) {
      const digest = digestFile(outPath);
      if (digest === model.sha256) {
        console.log(`cached ${model.id}`);
        return outPath;
      }
      console.log(`checksum mismatch ${model.id}, re-fetch`);
    } else {
      console.log(`cached ${model.id}`);
      return outPath;
    }
  }

  for (const seed of model.seedPaths ?? []) {
    if (!seed || !existsSync(seed)) continue;
    if (model.sha256 && digestFile(seed) !== model.sha256) continue;
    writeFileSync(outPath, readFileSync(seed));
    console.log(`seeded ${model.id} ← ${seed}`);
    return outPath;
  }

  if (!model.url) {
    throw new Error(
      `no weights for ${model.id}: run npm run distill:accurate or place ONNX at ${model.localPath}`,
    );
  }

  console.log(`download ${model.id} ← ${model.url}`);
  const res = await fetch(model.url, { redirect: "follow" });
  if (!res.ok || !res.body) {
    throw new Error(`download failed ${model.id}: HTTP ${res.status}`);
  }
  const tmp = `${outPath}.partial`;
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  if (model.sha256) {
    const digest = digestFile(tmp);
    if (digest !== model.sha256) {
      throw new Error(
        `checksum ${model.id}: expected ${model.sha256}, got ${digest}`,
      );
    }
  }
  renameSync(tmp, outPath);
  console.log(`saved ${outPath} (${statSize(outPath)} bytes)`);
  return outPath;
}

function statSize(p) {
  return readFileSync(p).byteLength;
}

async function preprocess(bytes, model) {
  const img = sharp(bytes).ensureAlpha().removeAlpha();
  const meta = await img.metadata();
  const w = meta.width ?? 0;
  const h = meta.height ?? 0;
  let pipelineImg = img;
  const size = model.inputSize;

  if (model.preprocess === "short440-center384") {
    const short = Math.min(w, h);
    const scale = 440 / short;
    const nw = Math.round(w * scale);
    const nh = Math.round(h * scale);
    const left = Math.max(0, Math.floor((nw - 384) / 2));
    const top = Math.max(0, Math.floor((nh - 384) / 2));
    pipelineImg = img
      .resize(nw, nh, { kernel: sharp.kernel.bilinear })
      .extract({ left, top, width: 384, height: 384 });
  } else {
    pipelineImg = img.resize(size, size, { fit: "fill", kernel: "lanczos3" });
  }

  const { data, info } = await pipelineImg
    .raw()
    .toBuffer({ resolveWithObject: true });
  const hw = info.width * info.height;
  const out = new Float32Array(3 * hw);
  const [m0, m1, m2] = model.mean;
  const [s0, s1, s2] = model.std;
  for (let i = 0, p = 0; i < hw; i += 1, p += info.channels) {
    out[i] = (data[p] / 255 - m0) / s0;
    out[hw + i] = (data[p + 1] / 255 - m1) / s1;
    out[2 * hw + i] = (data[p + 2] / 255 - m2) / s2;
  }
  return { float: out, width: info.width, height: info.height };
}

function scoreOutput(tensor, model) {
  const data = Array.from(tensor.data);
  if (model.outputKind === "logit") {
    return sigmoid(Number(data[0]));
  }
  // logits2
  if (data.length < 2) {
    return sigmoid(Number(data[0]));
  }
  return softmaxPick(data.slice(0, Math.max(2, data.length)), model.aiLabelIndex);
}

function loadCorpus() {
  const base = path.join(root, "benchmark/openrouter");
  const ai = JSON.parse(readFileSync(path.join(base, "index.json"), "utf8"));
  const real = JSON.parse(
    readFileSync(path.join(base, "real-index.json"), "utf8"),
  );
  let images = [
    ...ai.images.map((i) => ({
      file: i.file,
      label: "ai",
      hardCase: false,
      abs: path.join(base, i.file),
    })),
    ...real.images.map((i) => ({
      file: i.file,
      label: "real",
      hardCase: Boolean(i.hardCase),
      abs: path.join(base, i.file),
    })),
  ].filter((i) => existsSync(i.abs));

  const limit = Number(process.env.EVAL_COMPARE_LIMIT ?? "0");
  if (Number.isFinite(limit) && limit > 0) {
    const aiImgs = images.filter((i) => i.label === "ai");
    const realImgs = images.filter((i) => i.label === "real");
    const hard = realImgs.filter((i) => i.hardCase);
    const each = Math.max(1, Math.floor(limit / 2));
    const reals = [
      ...hard,
      ...realImgs.filter((i) => !i.hardCase),
    ].slice(0, each);
    images = [...aiImgs.slice(0, each), ...reals].slice(0, limit);
  }
  return images;
}

function summarize(rows, threshold) {
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  let sum = 0;
  let hardFp = 0;
  let hardN = 0;
  let lexicaTp = 0;
  let lexicaN = 0;
  for (const r of rows) {
    sum += r.inferMs + r.preprocessMs;
    const predAi = r.confidence >= threshold;
    const actualAi = r.label === "ai";
    if (actualAi && predAi) tp += 1;
    else if (!actualAi && !predAi) tn += 1;
    else if (!actualAi && predAi) fp += 1;
    else fn += 1;
    if (r.hardCase) {
      hardN += 1;
      if (predAi) hardFp += 1;
    }
    if (r.file.includes("lexica")) {
      lexicaN += 1;
      if (predAi) lexicaTp += 1;
    }
  }
  const tpr = tp + fn === 0 ? 0 : tp / (tp + fn);
  const tnr = tn + fp === 0 ? 0 : tn / (tn + fp);
  return {
    threshold,
    balancedAccuracy: (tpr + tnr) / 2,
    tpr,
    tnr,
    confusion: { tp, tn, fp, fn },
    avgMsPerImage: rows.length ? sum / rows.length : 0,
    totalMs: sum,
    hardCaseFp: hardFp,
    hardCaseN: hardN,
    lexicaTp,
    lexicaN,
    lexicaTpr: lexicaN === 0 ? null : lexicaTp / lexicaN,
  };
}

async function evalModel(model, images) {
  const modelPath = await download(model);
  // "all" breaks several fp16 ViT exports (SimplifiedLayerNormFusion).
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "disabled",
  });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  // warmup
  {
    const warm = images[0];
    const bytes = readFileSync(warm.abs);
    const { float, width, height } = await preprocess(bytes, model);
    const feeds = {
      [inputName]: new ort.Tensor("float32", float, [1, 3, height, width]),
    };
    await session.run(feeds);
  }

  const rows = [];
  const t0 = performance.now();
  for (const image of images) {
    const bytes = readFileSync(image.abs);
    const p0 = performance.now();
    const { float, width, height } = await preprocess(bytes, model);
    const preprocessMs = performance.now() - p0;
    const feeds = {
      [inputName]: new ort.Tensor("float32", float, [1, 3, height, width]),
    };
    const i0 = performance.now();
    const out = await session.run(feeds);
    const inferMs = performance.now() - i0;
    const confidence = scoreOutput(out[outputName], model);
    rows.push({
      file: image.file,
      label: image.label,
      hardCase: image.hardCase,
      confidence,
      preprocessMs,
      inferMs,
      totalMs: preprocessMs + inferMs,
    });
  }
  const wallMs = performance.now() - t0;
  await session.release?.();

  return {
    id: model.id,
    hf: model.hf,
    requested: model.requested,
    substituteFor: model.substituteFor ?? null,
    imageCount: rows.length,
    wallMs,
    at065: summarize(rows, 0.65),
    atProduct: summarize(rows, 0.6951),
    hardCases: rows
      .filter((r) => r.hardCase)
      .map((r) => ({
        file: r.file,
        confidence: Number(r.confidence.toFixed(4)),
        predAt065: r.confidence >= 0.65 ? "ai" : "real",
        predAtProduct: r.confidence >= 0.6951 ? "ai" : "real",
        ms: Number(r.totalMs.toFixed(2)),
      })),
    soylent:
      rows.find((r) => r.file.includes("soylent")) ??
      null,
  };
}

const filter = process.env.EVAL_COMPARE_MODELS
  ? new Set(process.env.EVAL_COMPARE_MODELS.split(",").map((s) => s.trim()))
  : null;

const images = loadCorpus();
console.log(`Corpus: ${images.length} images (${images.filter((i) => i.hardCase).length} hardcases)`);

const unavailable = MODELS.filter((m) => m.status === "unavailable");
const runnable = MODELS.filter((m) => {
  if (m.status !== "ok") return false;
  if (filter && !filter.has(m.id)) return false;
  const local = path.join(root, m.localPath);
  if (existsSync(local) || m.url) return true;
  console.log(`skip ${m.id}: no local weights (npm run distill:accurate)`);
  return false;
});

const results = [];
for (const model of runnable) {
  console.log(`\n=== ${model.id} ===`);
  const result = await evalModel(model, images);
  results.push(result);
  console.log(
    `BA@0.65=${(result.at065.balancedAccuracy * 100).toFixed(1)}% ` +
      `avg=${result.at065.avgMsPerImage.toFixed(1)}ms ` +
      `total=${(result.wallMs / 1000).toFixed(1)}s ` +
      `hardFP=${result.at065.hardCaseFp}/${result.at065.hardCaseN}`,
  );
}

const ranked = [...results].sort(
  (a, b) =>
    b.at065.balancedAccuracy - a.at065.balancedAccuracy ||
    a.at065.avgMsPerImage - b.at065.avgMsPerImage,
);

const lexicaCount = images.filter((i) => i.file.includes("lexica")).length;
const report = {
  generatedAt: new Date().toISOString(),
  corpusImages: images.length,
  hardCases: images.filter((i) => i.hardCase).length,
  lexicaImages: lexicaCount,
  note: "Two requested models lacked usable detector ONNX weights; substitutes noted below. Soylent hardcase is a local proxy. We do not fetch private Proofmark quants — train truepixel-accurate-v1 with npm run distill:accurate.",
  unavailable: unavailable.map((m) => ({
    id: m.id,
    reason: m.reason,
  })),
  rankingAt065: ranked.map((r, i) => ({
    rank: i + 1,
    id: r.id,
    ba: r.at065.balancedAccuracy,
    avgMsPerImage: r.at065.avgMsPerImage,
    totalWallSec: r.wallMs / 1000,
    tpr: r.at065.tpr,
    tnr: r.at065.tnr,
    hardCaseFp: r.at065.hardCaseFp,
    lexicaTpr: r.at065.lexicaTpr,
    substituteFor: r.substituteFor,
  })),
  models: results,
};

const outDir = path.join(root, "benchmark/model-survey");
mkdirSync(outDir, { recursive: true });
const jsonPath = path.join(outDir, "compare-top6-latest.json");
writeFileSync(jsonPath, JSON.stringify(report, null, 2) + "\n");

const md = [];
md.push("# Model compare report (OpenRouter + hardcases)");
md.push("");
md.push(
  `Generated \`${report.generatedAt}\` · **${report.corpusImages}** images · **${report.hardCases}** hardcases · **${report.lexicaImages}** Lexica AI`,
);
md.push("");
md.push(report.note);
md.push("");
md.push("## Unavailable requested models");
md.push("");
for (const u of report.unavailable) {
  md.push(`- **${u.id}** — ${u.reason}`);
}
md.push("");
md.push("## Ranking @ 65% threshold (bounty)");
md.push("");
md.push(
  "| Rank | Model | BA | TPR | TNR | Lexica TPR | Avg ms/img | Total s | Hardcase FP | Notes |",
);
md.push(
  "|-----:|-------|---:|----:|----:|-----------:|-----------:|--------:|------------:|-------|",
);
for (const r of report.rankingAt065) {
  const full = results.find((x) => x.id === r.id);
  const note = r.substituteFor
    ? `sub for ${r.substituteFor}`
    : r.id === "truepixel-accurate-v1"
      ? "our distill"
      : "requested";
  const lex =
    full.at065.lexicaTpr == null
      ? "—"
      : `${(full.at065.lexicaTpr * 100).toFixed(1)}% (${full.at065.lexicaTp}/${full.at065.lexicaN})`;
  md.push(
    `| ${r.rank} | \`${r.id}\` | ${(r.ba * 100).toFixed(1)}% | ${(full.at065.tpr * 100).toFixed(1)}% | ${(full.at065.tnr * 100).toFixed(1)}% | ${lex} | ${r.avgMsPerImage.toFixed(1)} | ${r.totalWallSec.toFixed(1)} | ${r.hardCaseFp}/${full.at065.hardCaseN} | ${note} |`,
  );
}
md.push("");
md.push("## Product threshold @ 69.51%");
md.push("");
md.push("| Model | BA | Avg ms/img | Hardcase FP |");
md.push("|-------|---:|-----------:|------------:|");
for (const r of ranked) {
  md.push(
    `| \`${r.id}\` | ${(r.atProduct.balancedAccuracy * 100).toFixed(1)}% | ${r.atProduct.avgMsPerImage.toFixed(1)} | ${r.atProduct.hardCaseFp}/${r.atProduct.hardCaseN} |`,
  );
}
md.push("");
md.push("## Soylent hardcase proxy");
md.push("");
md.push("| Model | Confidence | @65% | @69.51% | ms |");
md.push("|-------|-----------:|------|---------|---:|");
for (const r of ranked) {
  const s = r.soylent;
  if (!s) {
    md.push(`| \`${r.id}\` | — | — | — | — |`);
    continue;
  }
  md.push(
    `| \`${r.id}\` | ${(s.confidence * 100).toFixed(1)}% | ${s.confidence >= 0.65 ? "AI" : "real/other"} | ${s.confidence >= 0.6951 ? "AI" : "real/other"} | ${s.totalMs.toFixed(1)} |`,
  );
}
md.push("");
md.push("## Verdict");
md.push("");
if (ranked[0]) {
  const best = ranked[0];
  const fastest = [...ranked].sort(
    (a, b) => a.at065.avgMsPerImage - b.at065.avgMsPerImage,
  )[0];
  md.push(
    `- **Best BA @65%:** \`${best.id}\` at **${(best.at065.balancedAccuracy * 100).toFixed(1)}%** (avg **${best.at065.avgMsPerImage.toFixed(1)} ms**/image, **${(best.wallMs / 1000).toFixed(1)} s** total).`,
  );
  md.push(
    `- **Fastest:** \`${fastest.id}\` at **${fastest.at065.avgMsPerImage.toFixed(1)} ms**/image (**${(fastest.wallMs / 1000).toFixed(1)} s** total).`,
  );
  const bestHard = [...ranked].sort(
    (a, b) =>
      a.at065.hardCaseFp - b.at065.hardCaseFp ||
      b.at065.balancedAccuracy - a.at065.balancedAccuracy,
  )[0];
  md.push(
    `- **Best hardcase FP control:** \`${bestHard.id}\` with **${bestHard.at065.hardCaseFp}/${bestHard.at065.hardCaseN}** hardcase false positives @65%.`,
  );
}

const mdPath = path.join(outDir, "compare-top6-latest.md");
writeFileSync(mdPath, md.join("\n") + "\n");
console.log(`\nWrote ${jsonPath}`);
console.log(`Wrote ${mdPath}`);
