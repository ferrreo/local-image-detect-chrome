#!/usr/bin/env node
/**
 * One-time download of public model weights into ./models for packaging/tests.
 * The accurate head is packaged from a local distill artifact (no public URL).
 * The extension itself downloads/seeds into Cache Storage on first setup.
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

const MODEL_VERSION =
  "distilled-fp16+fp32+neopixel-accurate-v1-q8+opensynthid-q8";

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
    id: "neopixel-accurate-v1",
    url: "",
    outPath: path.join(
      root,
      "models/neopixel-accurate-v1/model_quantized.onnx",
    ),
    sha256: "25ef06372b8e5eb5cb183a85d34cc3e9a670c47d6eb7a72c109d7107aa467b0e",
    bytes: 24_044_443,
    license: "MIT (OwensLab backbone + NeoPixel-trained head)",
  },
  {
    id: "opensynthid-detect",
    url: "",
    outPath: path.join(
      root,
      "models/opensynthid-detect/model_quantized.onnx",
    ),
    sha256: "d3801422608b2a0f7b51a08e7417946a0dc88c5d7879e5a6fb8fc4008aaf630f",
    bytes: 21_665_653,
    license: "Apache-2.0 (OpenSynthID SynthID surrogate)",
  },
];

function digestFile(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

async function ensureLocal(model) {
  mkdirSync(path.dirname(model.outPath), { recursive: true });
  if (existsSync(model.outPath)) {
    const digest = digestFile(model.outPath);
    if (digest === model.sha256) {
      console.log(`Already present: ${model.outPath}`);
      return;
    }
    throw new Error(
      `Checksum mismatch for local ${model.id}: expected ${model.sha256}, got ${digest}. Re-run npm run distill:accurate / convert:opensynthid.`,
    );
  }
  throw new Error(
    `Missing packaged model at ${model.outPath}. Accurate head: npm run distill:accurate. OpenSynthID: npm run convert:opensynthid.`,
  );
}

async function download(model) {
  mkdirSync(path.dirname(model.outPath), { recursive: true });
  if (existsSync(model.outPath)) {
    const digest = digestFile(model.outPath);
    if (digest === model.sha256) {
      console.log(`Already present: ${model.outPath}`);
      return;
    }
    console.log(`Checksum mismatch for ${model.id}, re-downloading…`);
  }

  if (!model.url) {
    await ensureLocal(model);
    return;
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
  if (model.url) {
    await download(model);
  } else {
    await ensureLocal(model);
  }
}

writeFileSync(
  path.join(root, "models/manifest.json"),
  JSON.stringify(
    {
      version: MODEL_VERSION,
      models: MODELS.map((m) => ({
        id: m.id,
        path: path.relative(path.join(root, "models"), m.outPath),
        sha256: m.sha256,
        bytes: m.bytes,
        source: m.url || "local distill (npm run distill:accurate)",
        license: m.license,
      })),
      note: "Accurate head is NeoPixel-trained Q8; not fetched from third-party fine-tune quants.",
    },
    null,
    2,
  ) + "\n",
);

console.log("Model setup complete.");
