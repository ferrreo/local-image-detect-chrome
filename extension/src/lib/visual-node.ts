/**
 * Node.js ONNX path for local evaluation (onnxruntime-node + sharp preprocess).
 * Extension runtime keeps using onnxruntime-web in visual-classifier.ts.
 */
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import sharp from "sharp";
import {
  ALL_MODELS,
  DISTILLED_MODEL,
  FORENSICS_MODEL,
  type ModelArtifact,
} from "./model-manifest";
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

type LoadedSession = {
  model: ModelArtifact;
  ort: OrtNode;
  session: Awaited<ReturnType<OrtNode["InferenceSession"]["create"]>>;
};

let sessionsPromise: Promise<LoadedSession[]> | undefined;

function resolveModelPath(model: ModelArtifact): string {
  const candidates = [
    path.resolve(model.localPath),
    path.resolve(process.cwd(), model.localPath),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    `ONNX model missing (${model.id}). Run npm run setup:models before eval.`,
  );
}

async function getSessions(): Promise<LoadedSession[]> {
  if (!sessionsPromise) {
    sessionsPromise = (async () => {
      const ort = require("onnxruntime-node") as OrtNode;
      const loaded: LoadedSession[] = [];
      for (const model of ALL_MODELS) {
        const bytes = new Uint8Array(readFileSync(resolveModelPath(model)));
        const session = await ort.InferenceSession.create(bytes, {
          executionProviders: ["cpu"],
          graphOptimizationLevel: model.graphOptimizationLevel,
        });
        loaded.push({ model, ort, session });
      }
      return loaded;
    })();
  }
  return sessionsPromise;
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

async function bytesToModelTensor(
  bytes: Uint8Array,
  size: number,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Promise<Float32Array> {
  const { data } = await sharp(bytes)
    .resize(size, size, { kernel: "lanczos3", fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i += 1, p += 3) {
    out[i] = ((data[p] ?? 0) / 255 - mean[0]) / std[0];
    out[plane + i] = ((data[p + 1] ?? 0) / 255 - mean[1]) / std[1];
    out[2 * plane + i] = ((data[p + 2] ?? 0) / 255 - mean[2]) / std[2];
  }
  return out;
}

async function runOne(
  loaded: LoadedSession,
  bytes: Uint8Array,
): Promise<{ aiProb: number; inferMs: number; detail: string }> {
  const prep = await bytesToModelTensor(
    bytes,
    loaded.model.inputSize,
    loaded.model.mean,
    loaded.model.std,
  );
  const inputName = loaded.session.inputNames[0];
  if (!inputName) throw new Error(`${loaded.model.id}: no inputs`);
  const feeds: Record<string, unknown> = {
    [inputName]: new loaded.ort.Tensor("float32", prep, [
      1,
      3,
      loaded.model.inputSize,
      loaded.model.inputSize,
    ]),
  };
  const started = performance.now();
  const output = await loaded.session.run(feeds);
  const inferMs = performance.now() - started;
  const outName = loaded.session.outputNames[0];
  if (!outName) throw new Error(`${loaded.model.id}: no outputs`);
  const outTensor = output[outName];
  if (!outTensor) throw new Error(`${loaded.model.id}: missing tensor`);
  const logits = Array.from(outTensor.data, (v) => Number(v));
  const [p0, p1] = softmax2(logits[0] ?? 0, logits[1] ?? 0);
  const aiProb = loaded.model.aiLabelIndex === 0 ? p0 : p1;
  return {
    aiProb,
    inferMs,
    detail: `${loaded.model.id}=${aiProb.toFixed(3)}`,
  };
}

export async function classifyVisualNodeFromBytes(
  bytes: Uint8Array,
): Promise<VisualClassification & { inferMs: number; preprocessMs: number }> {
  const sessions = await getSessions();
  const started = performance.now();
  const byId = new Map<string, Awaited<ReturnType<typeof runOne>>>();
  for (const session of sessions) {
    byId.set(session.model.id, await runOne(session, bytes));
  }
  const inferMs = performance.now() - started;

  const distilled = byId.get(DISTILLED_MODEL.id);
  const forensics = byId.get(FORENSICS_MODEL.id);
  if (!distilled) throw new Error("Distilled model result missing");

  const backend: InferenceBackend = { kind: "wasm" };
  return {
    score: asAiConfidence(distilled.aiProb),
    ...(forensics
      ? { secondaryScore: asAiConfidence(forensics.aiProb) }
      : {}),
    backend,
    detail: [...byId.values()].map((r) => r.detail).join(","),
    inferMs,
    preprocessMs: 0,
  };
}

export async function warmVisualNode(): Promise<void> {
  await getSessions();
}
