#!/usr/bin/env node
/**
 * Host-side eval matrix: Node ORT CPU (distilled / dual).
 * Used by eval-suite.mjs; can also run alone:
 *   node scripts/eval-suite-host.mjs
 */
import { createServer } from "vite";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadOpenRouterCorpus, scoreRows } from "./lib/corpus.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const require = createRequire(import.meta.url);
const threshold = 0.65;

const DEFAULT_MODES = ["js-node-cpu-distilled", "js-node-cpu-dual"];

const modes = (process.env.EVAL_SUITE_HOST_MODES ?? DEFAULT_MODES.join(","))
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const { images } = loadOpenRouterCorpus(root);

const server = await createServer({
  configFile: false,
  root,
  server: { middlewareMode: true, hmr: false },
  resolve: {
    alias: {
      "@shared": path.join(root, "extension/src/shared"),
      "@lib": path.join(root, "extension/src/lib"),
    },
  },
  ssr: { external: ["onnxruntime-node", "sharp"] },
});

await server.ssrLoadModule("/tests/setup/canvas-polyfill.ts");
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
const nodeVisual = await server.ssrLoadModule(
  "/extension/src/lib/visual-node.ts",
);

const ort = require("onnxruntime-node");
const sharp = (await import("sharp")).default;

async function nodeDistilledOnly(bytes) {
  const size = 224;
  const mean = 0.5;
  const std = 0.5;
  if (!nodeDistilledOnly.session) {
    const modelPath = path.join(
      root,
      "models/ai-image-detect-distilled/model_fp16.onnx",
    );
    nodeDistilledOnly.session = await ort.InferenceSession.create(
      new Uint8Array(readFileSync(modelPath)),
      { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
    );
  }
  const { data } = await sharp(bytes)
    .resize(size, size, { fit: "fill", kernel: "lanczos3" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i += 1, p += 3) {
    out[i] = (data[p] / 255 - mean) / std;
    out[plane + i] = (data[p + 1] / 255 - mean) / std;
    out[2 * plane + i] = (data[p + 2] / 255 - mean) / std;
  }
  const t0 = performance.now();
  const o = await nodeDistilledOnly.session.run({
    [nodeDistilledOnly.session.inputNames[0]]: new ort.Tensor(
      "float32",
      out,
      [1, 3, size, size],
    ),
  });
  const inferMs = performance.now() - t0;
  const d = Array.from(o[nodeDistilledOnly.session.outputNames[0]].data, Number);
  const m = Math.max(d[0], d[1]);
  const ea = Math.exp(d[0] - m);
  const eb = Math.exp(d[1] - m);
  const p0 = ea / (ea + eb);
  return {
    score: asAiConfidence(p0),
    detail: `distilled=${p0.toFixed(3)}`,
    inferMs,
  };
}

async function fusePipeline(item, visual) {
  const buf = readFileSync(item.abs);
  const bytes = new Uint8Array(buf);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const t0 = performance.now();
  const prov = analyzeProvenance(bytes);
  let spectralScore = asAiConfidence(0.5);
  let spectralDetail = "skipped";
  let spectralFeatures;
  let visualScore = asAiConfidence(0.5);
  let visualSecondary;
  let visualDetail = "skipped";
  let inferMs = 0;
  let backend = "none";

  if (!prov.shortCircuit) {
    const decoded = await decodeImageBytes(ab, guessMimeType(bytes));
    try {
      const spectral = analyzeSpectral(
        await rasterizeForSpectral(decoded.bitmap),
      );
      spectralScore = spectral.score;
      spectralDetail = spectral.detail;
      spectralFeatures = spectral.features;
      const v = await visual({
        abs: item.abs,
        bytes,
        spectralScore,
        spectralFeatures,
      });
      visualScore = v.score;
      visualSecondary = v.secondaryScore;
      visualDetail = v.detail;
      inferMs = v.inferMs ?? 0;
      backend = v.backend ?? "cpu";
    } finally {
      decoded.bitmap.close();
    }
  }

  const fused = fuseDetection({
    provenance: {
      tier: "provenance",
      aiScore: prov.score,
      weight: 0.08,
      detail: prov.detail,
      shortCircuit: prov.shortCircuit,
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
            detail: "neopixel-accurate",
          },
        }
      : {}),
    ...(spectralFeatures !== undefined ? { spectralFeatures } : {}),
    threshold,
  });

  return {
    file: item.file,
    label: item.label,
    model: item.model,
    confidence: fused.confidence,
    predicted: fused.confidence >= threshold ? "ai" : "real",
    correct: (fused.confidence >= threshold) === (item.label === "ai"),
    backend,
    totalMs: performance.now() - t0,
    inferMs,
    tiers: fused.tiers
      .map((t) => `${t.tier}:${Number(t.aiScore).toFixed(3)}`)
      .join("|"),
  };
}

async function runMode(mode) {
  console.log(`\n=== host mode: ${mode} ===`);
  const started = Date.now();
  await nodeVisual.warmVisualNode();
  const meta = {
    mode,
    runtime: "host",
    engine: "onnxruntime-node",
    preferEp: "CPU",
    device: "cpu",
  };

  const rows = [];
  for (const item of images) {
    let row;
    if (mode === "js-node-cpu-distilled") {
      row = await fusePipeline(item, async ({ bytes }) => {
        const v = await nodeDistilledOnly(bytes);
        return { ...v, backend: "cpu" };
      });
    } else if (mode === "js-node-cpu-dual") {
      row = await fusePipeline(item, async ({ bytes }) => {
        const v = await nodeVisual.classifyVisualNodeFromBytes(bytes);
        return {
          score: v.score,
          secondaryScore: v.secondaryScore,
          detail: v.detail,
          inferMs: v.inferMs,
          backend: "cpu",
        };
      });
    } else {
      throw new Error(`Unknown host mode: ${mode}`);
    }
    rows.push(row);
    process.stdout.write(row.correct ? "." : "X");
  }
  process.stdout.write("\n");

  const scored = scoreRows(rows, threshold);
  const report = {
    ...meta,
    threshold,
    ...scored,
    wallMs: Date.now() - started,
    rows,
  };
  console.log(
    `${mode}: BA=${(scored.balancedAccuracy * 100).toFixed(1)}% avg=${scored.timing.avgTotalMs.toFixed(1)}ms`,
    scored.confusion,
  );
  return report;
}

const results = [];
for (const mode of modes) {
  try {
    results.push(await runMode(mode));
  } catch (error) {
    results.push({
      mode,
      skipped: false,
      error: error instanceof Error ? error.message : String(error),
    });
    console.error(mode, error);
  }
}

await server.close();

const outDir = path.join(root, "benchmark/eval-suite");
mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, "host-latest.json");
writeFileSync(
  outPath,
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      host: process.platform,
      arch: process.arch,
      modes,
      results,
    },
    null,
    2,
  ) + "\n",
);
console.log(`Wrote ${outPath}`);
