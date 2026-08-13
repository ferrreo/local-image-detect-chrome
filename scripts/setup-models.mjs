#!/usr/bin/env node
/**
 * One-time download of public model weights into ./models for packaging/tests.
 * The extension itself downloads into Cache Storage on first setup.
 */
import { createHash } from "node:crypto";
import {
  createWriteStream,
  mkdirSync,
  existsSync,
  readFileSync,
  writeFileSync,
  renameSync,
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
    id: "ai-image-detection-capcheck",
    url: "https://huggingface.co/onnx-community/ai-image-detection-ONNX/resolve/main/onnx/model_q4.onnx",
    outPath: path.join(root, "models/ai-image-detection/model_q4.onnx"),
    sha256: "28c7f06d5aa87bc7e023c023eab1fbf473deef54e9c62f9838a99e50422810ec",
    bytes: 56_757_898,
    license: "Apache-2.0",
  },
];

async function download(model) {
  mkdirSync(path.dirname(model.outPath), { recursive: true });
  if (existsSync(model.outPath)) {
    const existing = readFileSync(model.outPath);
    const digest = createHash("sha256").update(existing).digest("hex");
    if (digest === model.sha256) {
      console.log(`Already present: ${model.outPath}`);
      return;
    }
    console.log(`Checksum mismatch for ${model.id}, re-downloading…`);
  }

  console.log(`Downloading ${model.url}`);
  const response = await fetch(model.url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed: HTTP ${response.status}`);
  }

  const tmp = `${model.outPath}.partial`;
  await pipeline(Readable.fromWeb(response.body), createWriteStream(tmp));
  const buf = readFileSync(tmp);
  const digest = createHash("sha256").update(buf).digest("hex");
  if (digest !== model.sha256) {
    throw new Error(
      `Checksum mismatch: expected ${model.sha256}, got ${digest}`,
    );
  }
  renameSync(tmp, model.outPath);
  console.log(
    `Saved ${model.outPath} (${buf.byteLength} bytes, sha256=${digest})`,
  );
}

for (const model of MODELS) {
  await download(model);
}

writeFileSync(
  path.join(root, "models/manifest.json"),
  JSON.stringify(
    {
      version: "distilled-fp16+capcheck-q4-v1",
      models: MODELS.map((m) => ({
        id: m.id,
        path: path.relative(path.join(root, "models"), m.outPath),
        sha256: m.sha256,
        bytes: m.bytes,
        source: m.url,
        license: m.license,
      })),
    },
    null,
    2,
  ) + "\n",
);

console.log("Model setup complete.");
