/**
 * Node.js ONNX path for local evaluation (onnxruntime-node + sharp preprocess).
 * Extension runtime keeps using onnxruntime-web in visual-classifier.ts.
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import sharp from "sharp";
import { VISUAL_MODEL } from "./model-manifest";
import { asAiConfidence, type InferenceBackend } from "../shared/types";
import type { VisualClassification } from "./visual-stub";

const require = createRequire(import.meta.url);

type OrtNode = {
  InferenceSession: {
    create: (
      pathOrBuffer: string | Uint8Array,
      options?: Record<string, unknown>,
    ) => Promise<{
      inputNames: string[];
      outputNames: string[];
      run: (
        feeds: Record<string, unknown>,
      ) => Promise<Record<string, { data: ArrayLike<number> }>>;
    }>;
  };
  Tensor: new (
    type: string,
    data: Float32Array,
    dims: number[],
  ) => unknown;
};

let sessionPromise:
  | Promise<{
      ort: OrtNode;
      session: Awaited<ReturnType<OrtNode["InferenceSession"]["create"]>>;
    }>
  | undefined;

function resolveModelPath(): string {
  const candidates = [
    path.resolve("models/ai-image-detect-distilled/model_fp16.onnx"),
    path.resolve(
      process.cwd(),
      "models/ai-image-detect-distilled/model_fp16.onnx",
    ),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    "ONNX model missing. Run npm run setup:models before TRUEPIXEL_STUB=0 eval.",
  );
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = require("onnxruntime-node") as OrtNode;
      const modelPath = resolveModelPath();
      const bytes = new Uint8Array(readFileSync(modelPath));
      // fp16 export breaks SimplifiedLayerNormFusion under ORT node; keep opts off.
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["cpu"],
        graphOptimizationLevel: "disabled",
      });
      return { ort, session };
    })();
  }
  return sessionPromise;
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

/** Center-crop + bilinear resize to model size, then NCHW normalize (ViT mean/std 0.5). */
async function bytesToModelTensor(bytes: Uint8Array): Promise<Float32Array> {
  const size = VISUAL_MODEL.inputSize;
  const meta = await sharp(bytes).metadata();
  const width = meta.width ?? size;
  const height = meta.height ?? size;
  const side = Math.min(width, height);
  const left = Math.floor((width - side) / 2);
  const top = Math.floor((height - side) / 2);

  const { data } = await sharp(bytes)
    .extract({ left, top, width: side, height: side })
    .resize(size, size, { kernel: "lanczos3", fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  const mean = VISUAL_MODEL.mean;
  const std = VISUAL_MODEL.std;
  for (let i = 0, p = 0; i < plane; i += 1, p += 3) {
    out[i] = ((data[p] ?? 0) / 255 - mean[0]) / std[0];
    out[plane + i] = ((data[p + 1] ?? 0) / 255 - mean[1]) / std[1];
    out[2 * plane + i] = ((data[p + 2] ?? 0) / 255 - mean[2]) / std[2];
  }
  return out;
}

export async function classifyVisualNodeFromBytes(
  bytes: Uint8Array,
): Promise<VisualClassification & { inferMs: number; preprocessMs: number }> {
  const prepStarted = performance.now();
  const tensorData = await bytesToModelTensor(bytes);
  const preprocessMs = performance.now() - prepStarted;

  const { ort, session } = await getSession();
  const inputName = session.inputNames[0];
  if (!inputName) throw new Error("Model has no inputs");

  const feeds: Record<string, unknown> = {
    [inputName]: new ort.Tensor("float32", tensorData, [
      1,
      3,
      VISUAL_MODEL.inputSize,
      VISUAL_MODEL.inputSize,
    ]),
  };

  const started = performance.now();
  const output = await session.run(feeds);
  const inferMs = performance.now() - started;

  const outName = session.outputNames[0];
  if (!outName) throw new Error("Model has no outputs");
  const outTensor = output[outName];
  if (!outTensor) throw new Error("Missing output tensor");

  const logits = Array.from(outTensor.data, (v) => Number(v));
  const [p0, p1] = softmax2(logits[0] ?? 0, logits[1] ?? 0);
  const aiProb = VISUAL_MODEL.aiLabelIndex === 0 ? p0 : p1;
  const backend: InferenceBackend = { kind: "wasm" };

  return {
    score: asAiConfidence(aiProb),
    backend,
    detail: `node-ort logits=[${(logits[0] ?? 0).toFixed(3)},${(logits[1] ?? 0).toFixed(3)}]`,
    inferMs,
    preprocessMs,
  };
}

export async function warmVisualNode(): Promise<void> {
  await getSession();
}
