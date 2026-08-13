#!/usr/bin/env node
/**
 * Local smoke evaluation on generated fixtures using the stub visual path.
 * Not the private bounty benchmark.
 */
import { createServer } from "vite";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const index = JSON.parse(
  readFileSync(path.join(root, "tests/fixtures/images/index.json"), "utf8"),
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

for (const item of index.images) {
  const file = path.join(root, "tests/fixtures/images", item.file);
  const buf = readFileSync(file);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const result = await detectAiImage({
    imageId: item.file,
    bytes,
    stubVisual: true,
    threshold: 0.65,
  });
  const predictedAi = isAiAtThreshold(result.confidence, 0.65);
  const actualAi = item.label === "ai";
  if (actualAi && predictedAi) tp += 1;
  else if (!actualAi && !predictedAi) tn += 1;
  else if (!actualAi && predictedAi) fp += 1;
  else fn += 1;
  console.log(
    `${item.file.padEnd(18)} truth=${item.label.padEnd(4)} conf=${result.confidence.toFixed(3)} pred=${predictedAi ? "ai" : "real"} backend=${result.backend.kind}`,
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

await server.close();
if (Number.isNaN(bal)) process.exit(1);
