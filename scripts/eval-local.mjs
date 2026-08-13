#!/usr/bin/env node
/**
 * Local evaluation harness.
 * Prefers benchmark/openrouter (real OpenRouter AI samples + Picsum reals).
 * Falls back to synthetic tests/fixtures/images.
 *
 * Uses stub visual by default (TRUEPIXEL_STUB=1). Set TRUEPIXEL_STUB=0 and
 * install models to exercise the ONNX path outside the extension.
 */
import { createServer } from "vite";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const stubVisual = process.env.TRUEPIXEL_STUB !== "0";

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
});

await server.ssrLoadModule("/tests/setup/canvas-polyfill.ts");
const { detectAiImage } = await server.ssrLoadModule(
  "/extension/src/lib/pipeline.ts",
);
const { balancedAccuracy, isAiAtThreshold } = await server.ssrLoadModule(
  "/extension/src/lib/fusion.ts",
);

let tp = 0;
let tn = 0;
let fp = 0;
let fn = 0;
const byModel = new Map();

for (const item of dataset.images) {
  const file = path.join(dataset.root, item.file);
  if (!existsSync(file)) {
    console.warn(`missing ${item.file}`);
    continue;
  }
  const buf = readFileSync(file);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const result = await detectAiImage({
    imageId: item.file,
    bytes,
    stubVisual,
    threshold: 0.65,
  });
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

  console.log(
    `${item.file.padEnd(64)} truth=${item.label.padEnd(4)} conf=${result.confidence.toFixed(3)} pred=${predictedAi ? "ai" : "real"}`,
  );
}

const bal = balancedAccuracy({
  truePositive: tp,
  trueNegative: tn,
  falsePositive: fp,
  falseNegative: fn,
});

console.log(
  `\nConfusion: tp=${tp} tn=${tn} fp=${fp} fn=${fn}\nBalanced accuracy @65%: ${(bal * 100).toFixed(1)}%`,
);

if (byModel.size > 0) {
  console.log("\nPer-model AI recall (stub/onnx depending on env):");
  for (const [model, stats] of [...byModel.entries()].sort()) {
    console.log(
      `  ${model.padEnd(48)} ${stats.hit}/${stats.n} (${((100 * stats.hit) / stats.n).toFixed(0)}%)`,
    );
  }
}

await server.close();
if (Number.isNaN(bal)) process.exit(1);
