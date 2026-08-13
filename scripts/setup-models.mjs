#!/usr/bin/env node
/**
 * One-time download of public model weights into ./models for packaging/tests.
 * The extension itself downloads into Cache Storage on first setup.
 * Proofmark is seeded from a local vendor path (no public HF URL).
 */
import { createHash } from "node:crypto";
import {
  createWriteStream,
  mkdirSync,
  existsSync,
  readFileSync,
  writeFileSync,
  renameSync,
  copyFileSync,
} from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const MODELS = [
  {
    id: "ai-image-detect-distilled",
    url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model_fp16.onnx",
    outPath: path.join(
      root,
      "models/ai-image-detect-distilled/model_fp16.onnx",
    ),
    sha256: "9594bacb70d9c65fcaa656e0d17038c5cac7a6c48d04cd10f2ebf972a01ba3ee",
    bytes: 29_273_325,
    license: "MIT",
  },
  {
    id: "ai-image-detect-distilled-fp32",
    url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model.onnx",
    outPath: path.join(root, "models/ai-image-detect-distilled/model.onnx"),
    sha256: "87b4331f22418a4cb50901851a1c28f64a0ca4f58728442d073b4bed9922ba86",
    bytes: 58_410_332,
    license: "MIT",
  },
  {
    id: "proofmark-webwild-v3",
    // HF `Proofmark/proofmark-webwild-v3` is private; same Q8 bytes are public
    // in Dyno-man/Dino-ImageGen-Ext (upstream backbone OwensLab/commfor-model-384).
    url: "https://raw.githubusercontent.com/Dyno-man/Dino-ImageGen-Ext/main/public/models/Proofmark/proofmark-webwild-v3/onnx/model_quantized.onnx",
    outPath: path.join(
      root,
      "models/proofmark-webwild-v3/model_quantized.onnx",
    ),
    sha256:
      "ed17ceb332bef84d0adcc2fa537eef85ed3ac6fb32c30393c326321fbbe54683",
    bytes: 24_031_833,
    license: "MIT (Proofmark head; OwensLab/commfor-model-384 backbone)",
    seedPaths: [
      path.join(
        root,
        "models/compare/proofmark-webwild-v3/model_quantized.onnx",
      ),
      "/tmp/Dino-ImageGen-Ext/public/models/Proofmark/proofmark-webwild-v3/onnx/model_quantized.onnx",
      path.join(
        root,
        "vendor/Dino-ImageGen-Ext/public/models/Proofmark/proofmark-webwild-v3/onnx/model_quantized.onnx",
      ),
    ],
  },
];

function digestFile(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

async function download(model) {
  mkdirSync(path.dirname(model.outPath), { recursive: true });
  if (existsSync(model.outPath)) {
    const digest = digestFile(model.outPath);
    if (digest === model.sha256) {
      console.log(`Already present: ${model.outPath}`);
      return;
    }
    console.log(`Checksum mismatch for ${model.id}, re-fetching…`);
  }

  for (const seed of model.seedPaths ?? []) {
    if (!seed || !existsSync(seed)) continue;
    if (digestFile(seed) !== model.sha256) continue;
    copyFileSync(seed, model.outPath);
    console.log(`Seeded ${model.id} ← ${seed}`);
    return;
  }

  if (!model.url) {
    throw new Error(
      `No weights for ${model.id}. Place ONNX at ${model.outPath} or one of: ${(model.seedPaths ?? []).join(", ")}`,
    );
  }

  console.log(`Downloading ${model.url}`);
  const response = await fetch(model.url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed: HTTP ${response.status}`);
  }

  const tmp = `${model.outPath}.partial`;
  await pipeline(Readable.fromWeb(response.body), createWriteStream(tmp));
  const digest = digestFile(tmp);
  if (digest !== model.sha256) {
    throw new Error(
      `Checksum mismatch: expected ${model.sha256}, got ${digest}`,
    );
  }
  renameSync(tmp, model.outPath);
  console.log(
    `Saved ${model.outPath} (${readFileSync(model.outPath).byteLength} bytes, sha256=${digest})`,
  );
}

for (const model of MODELS) {
  await download(model);
}

writeFileSync(
  path.join(root, "models/manifest.json"),
  JSON.stringify(
    {
      version: "distilled-fp16+fp32+proofmark-webwild-v3-q8-v1",
      models: MODELS.map((m) => ({
        id: m.id,
        path: path.relative(path.join(root, "models"), m.outPath),
        sha256: m.sha256,
        bytes: m.bytes,
        source: m.url || "(vendored seed)",
        license: m.license,
      })),
    },
    null,
    2,
  ) + "\n",
);

console.log("Model setup complete.");
