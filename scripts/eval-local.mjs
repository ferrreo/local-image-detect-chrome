#!/usr/bin/env node
/**
 * Local evaluation harness against benchmark/openrouter (preferred).
 *
 * Default: real ONNX model via onnxruntime-node (TRUEPIXEL_STUB=0).
 * Set TRUEPIXEL_STUB=1 to use the heuristic stub instead.
 *
 * Prints per-image confidence, prediction, and timing breakdown.
 */
import { createServer } from "vite";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const require = createRequire(import.meta.url);
const stubVisual = process.env.TRUEPIXEL_STUB === "1";

function loadDataset() {
  const openrouterAi = path.join(root, "benchmark/openrouter/index.json");
  const openrouterReal = path.join(root, "benchmark/openrouter/real-index.json");
  if (existsSync(openrouterAi) && existsSync(openrouterReal)) {
    const ai = JSON.parse(readFileSync(openrouterAi, "utf8"));
    const real = JSON.parse(readFileSync(openrouterReal, "utf8"));
    return {
      name: "openrouter",
      root: path.join(root, "benchmark/openrouter"),
      images: [
        ...ai.images.map((i) => ({
          file: i.file,
          label: "ai",
          model: i.model,
        })),
        ...real.images.map((i) => ({
          file: i.file,
          label: "real",
          model: null,
        })),
      ],
    };
  }

  const fixtures = path.join(root, "tests/fixtures/images/index.json");
  const index = JSON.parse(readFileSync(fixtures, "utf8"));
  return {
    name: "synthetic-fixtures",
    root: path.join(root, "tests/fixtures/images"),
    images: index.images.map((i) => ({
      file: i.file,
      label: i.label,
      model: null,
    })),
  };
}

const dataset = loadDataset();
console.log(
  `Dataset: ${dataset.name} (${dataset.images.length} images), stubVisual=${stubVisual}`,
);

const server = await createServer({
  configFile: false,
  root,
  server: { middlewareMode: true },
  resolve: {
    alias: {
      "@shared": path.join(root, "extension/src/shared"),
      "@lib": path.join(root, "extension/src/lib"),
    },
  },
  ssr: {
    external: ["onnxruntime-node", "sharp"],
  },
});

await server.ssrLoadModule("/tests/setup/canvas-polyfill.ts");
const { detectAiImage } = await server.ssrLoadModule(
  "/extension/src/lib/pipeline.ts",
);
const { balancedAccuracy, isAiAtThreshold } = await server.ssrLoadModule(
  "/extension/src/lib/fusion.ts",
);
const { analyzeProvenance } = await server.ssrLoadModule(
  "/extension/src/lib/provenance.ts",
);
const { analyzeSpectral } = await server.ssrLoadModule(
  "/extension/src/lib/spectral.ts",
);
const { fuseDetection } = await server.ssrLoadModule(
  "/extension/src/lib/fusion.ts",
);
const { asAiConfidence } = await server.ssrLoadModule(
  "/extension/src/shared/types.ts",
);
const { decodeImageBytes, guessMimeType, rasterizeForSpectral } =
  await server.ssrLoadModule("/extension/src/lib/image-decode.ts");

let classifyVisualNodeFromBytes;
let warmVisualNode;
if (!stubVisual) {
  const nodeVisual = await server.ssrLoadModule(
    "/extension/src/lib/visual-node.ts",
  );
  classifyVisualNodeFromBytes = nodeVisual.classifyVisualNodeFromBytes;
  warmVisualNode = nodeVisual.warmVisualNode;
  console.log("Warming ONNX session…");
  const warmStart = performance.now();
  await warmVisualNode();
  console.log(`ONNX ready in ${(performance.now() - warmStart).toFixed(1)} ms`);
}

async function detectWithTimings(item, bytes) {
  const totalStart = performance.now();
  const bytesView = new Uint8Array(bytes);
  const mimeType = guessMimeType(bytesView);

  const provStart = performance.now();
  const provenance = analyzeProvenance(bytesView);
  const provenanceMs = performance.now() - provStart;

  let spectralScore = asAiConfidence(0.5);
  let spectralDetail = "skipped";
  let spectralFeatures;
  let spectralMs = 0;
  let visualScore = asAiConfidence(0.5);
  let visualSecondary;
  let visualDetail = "skipped";
  let visualMs = 0;
  let inferMs = 0;
  let preprocessMs = 0;
  let backend = { kind: "none" };

  if (!provenance.shortCircuit) {
    const decodeStart = performance.now();
    const decoded = await decodeImageBytes(bytes, mimeType);
    const decodeMs = performance.now() - decodeStart;
    try {
      const specStart = performance.now();
      const spectralImage = await rasterizeForSpectral(decoded.bitmap);
      const spectral = analyzeSpectral(spectralImage);
      spectralMs = performance.now() - specStart;
      spectralScore = spectral.score;
      spectralDetail = spectral.detail;
      spectralFeatures = spectral.features;

      const visStart = performance.now();
      if (stubVisual) {
        const result = await detectAiImage({
          imageId: item.file,
          bytes,
          stubVisual: true,
          threshold: 0.65,
        });
        // Re-extract visual from tiers for reporting; total timing still measured below.
        const visualTier = result.tiers.find((t) => t.tier === "visual");
        visualScore = visualTier?.aiScore ?? asAiConfidence(0.5);
        visualDetail = visualTier?.detail ?? "stub";
        backend = result.backend;
        visualMs = performance.now() - visStart;
      } else {
        const visual = await classifyVisualNodeFromBytes(bytesView);
        visualScore = visual.score;
        visualSecondary = visual.secondaryScore;
        visualDetail = visual.detail;
        backend = visual.backend;
        inferMs = visual.inferMs;
        preprocessMs = visual.preprocessMs;
        visualMs = performance.now() - visStart;
      }
      void decodeMs;
    } finally {
      decoded.bitmap.close();
    }
  }

  const fuseStart = performance.now();
  const fused = fuseDetection({
    provenance: {
      tier: "provenance",
      aiScore: provenance.score,
      weight: 0.08,
      detail: provenance.detail,
      shortCircuit: provenance.shortCircuit,
    },
    spectral: {
      tier: "spectral",
      aiScore: spectralScore,
      weight: 0.2,
      detail: spectralDetail,
    },
    visual: {
      tier: "visual",
      aiScore: visualScore,
      weight: 0.72,
      detail: visualDetail,
    },
    ...(visualSecondary !== undefined
      ? {
          visualSecondary: {
            tier: "visual",
            aiScore: visualSecondary,
            weight: 0.5,
            detail: "community-forensics",
          },
        }
      : {}),
    ...(spectralFeatures !== undefined ? { spectralFeatures } : {}),
    threshold: 0.65,
  });
  const fuseMs = performance.now() - fuseStart;
  const totalMs = performance.now() - totalStart;

  return {
    confidence: fused.confidence,
    label: fused.label,
    backend,
    tiers: fused.tiers,
    timing: {
      totalMs,
      provenanceMs,
      spectralMs,
      visualMs,
      preprocessMs,
      inferMs,
      fuseMs,
    },
  };
}

let tp = 0;
let tn = 0;
let fp = 0;
let fn = 0;
const byModel = new Map();
const rows = [];
let totalMsSum = 0;

for (const item of dataset.images) {
  const file = path.join(dataset.root, item.file);
  if (!existsSync(file)) {
    console.warn(`missing ${item.file}`);
    continue;
  }
  const buf = readFileSync(file);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const result = await detectWithTimings(item, bytes);
  const predictedAi = isAiAtThreshold(result.confidence, 0.65);
  const actualAi = item.label === "ai";
  if (actualAi && predictedAi) tp += 1;
  else if (!actualAi && !predictedAi) tn += 1;
  else if (!actualAi && predictedAi) fp += 1;
  else fn += 1;

  if (item.model) {
    const prev = byModel.get(item.model) ?? { n: 0, hit: 0 };
    prev.n += 1;
    if (predictedAi) prev.hit += 1;
    byModel.set(item.model, prev);
  }

  totalMsSum += result.timing.totalMs;
  rows.push({
    file: item.file,
    label: item.label,
    model: item.model,
    confidence: result.confidence,
    predicted: predictedAi ? "ai" : "real",
    correct: predictedAi === actualAi,
    backend: result.backend.kind,
    ...result.timing,
    tiers: result.tiers.map((t) => `${t.tier}:${Number(t.aiScore).toFixed(3)}`).join("|"),
  });

  const mark = predictedAi === actualAi ? "ok" : "MISS";
  console.log(
    `${mark.padEnd(4)} ${item.file.padEnd(58)} truth=${item.label.padEnd(4)} conf=${result.confidence.toFixed(3)} pred=${predictedAi ? "ai" : "real"} total=${result.timing.totalMs.toFixed(1)}ms infer=${result.timing.inferMs.toFixed(1)}ms visual=${result.timing.visualMs.toFixed(1)}ms spectral=${result.timing.spectralMs.toFixed(1)}ms`,
  );
}

const bal = balancedAccuracy({
  truePositive: tp,
  trueNegative: tn,
  falsePositive: fp,
  falseNegative: fn,
});

const n = rows.length || 1;
console.log(
  `\nConfusion: tp=${tp} tn=${tn} fp=${fp} fn=${fn}\nBalanced accuracy @65%: ${(bal * 100).toFixed(1)}%`,
);
console.log(
  `Timing: avg total=${(totalMsSum / n).toFixed(1)}ms  sum=${totalMsSum.toFixed(1)}ms  n=${rows.length}`,
);

if (byModel.size > 0) {
  console.log("\nPer-model AI recall:");
  for (const [model, stats] of [...byModel.entries()].sort()) {
    console.log(
      `  ${model.padEnd(48)} ${stats.hit}/${stats.n} (${((100 * stats.hit) / stats.n).toFixed(0)}%)`,
    );
  }
}

const outDir = path.join(root, "benchmark/openrouter");
mkdirSync(outDir, { recursive: true });
const reportPath = path.join(outDir, "eval-report.json");
writeFileSync(
  reportPath,
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      stubVisual,
      threshold: 0.65,
      balancedAccuracy: bal,
      confusion: { tp, tn, fp, fn },
      timing: {
        avgTotalMs: totalMsSum / n,
        sumTotalMs: totalMsSum,
        count: rows.length,
      },
      rows,
    },
    null,
    2,
  ) + "\n",
);
console.log(`\nWrote ${reportPath}`);

await server.close();
if (Number.isNaN(bal)) process.exit(1);
if (bal < 1) process.exitCode = 2;
void require;
