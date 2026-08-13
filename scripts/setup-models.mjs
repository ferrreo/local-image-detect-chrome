#!/usr/bin/env node
/**
 * One-time download of public model weights into ./models for packaging/tests.
 * The extension itself downloads into Cache Storage on first setup.
 */
import { createHash } from "node:crypto";
import { createWriteStream, mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const MODEL = {
  id: "ai-image-detect-distilled",
  url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model_fp16.onnx",
  outPath: path.join(
    root,
    "models/ai-image-detect-distilled/model_fp16.onnx",
  ),
  sha256: "9594bacb70d9c65fcaa656e0d17038c5cac7a6c48d04cd10f2ebf972a01ba3ee",
  bytes: 29_273_325,
};

async function download(url, dest) {
  mkdirSync(path.dirname(dest), { recursive: true });
  if (existsSync(dest)) {
    const existing = readFileSync(dest);
    const digest = createHash("sha256").update(existing).digest("hex");
    if (digest === MODEL.sha256) {
      console.log(`Already present: ${dest}`);
      return;
    }
    console.log(`Checksum mismatch for cached file, re-downloading…`);
  }

  console.log(`Downloading ${url}`);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed: HTTP ${response.status}`);
  }

  const tmp = `${dest}.partial`;
  await pipeline(Readable.fromWeb(response.body), createWriteStream(tmp));
  const buf = readFileSync(tmp);
  const digest = createHash("sha256").update(buf).digest("hex");
  if (digest !== MODEL.sha256) {
    throw new Error(`Checksum mismatch: expected ${MODEL.sha256}, got ${digest}`);
  }
  writeFileSync(dest, buf);
  console.log(`Saved ${dest} (${buf.byteLength} bytes, sha256=${digest})`);
}

await download(MODEL.url, MODEL.outPath);

writeFileSync(
  path.join(root, "models/manifest.json"),
  JSON.stringify(
    {
      version: "ai-image-detect-distilled-fp16-v1",
      models: [
        {
          id: MODEL.id,
          path: "ai-image-detect-distilled/model_fp16.onnx",
          sha256: MODEL.sha256,
          bytes: MODEL.bytes,
          source: MODEL.url,
          license: "MIT",
        },
      ],
    },
    null,
    2,
  ),
);

console.log("Model setup complete.");
