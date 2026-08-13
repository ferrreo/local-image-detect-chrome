import {
  ALL_MODELS,
  DISTILLED_MODEL,
  FORENSICS_MODEL,
  VISUAL_MODEL,
  type ModelArtifact,
} from "./model-manifest";
import { readCachedModel } from "./model-cache";
import {
  imageDataToNchwFloat32,
  rasterizeForModel,
} from "./image-decode";
import {
  asAiConfidence,
  type InferenceBackend,
} from "../shared/types";
import {
  stubVisualClassify,
  type VisualClassification,
} from "./visual-stub";

export type { VisualClassification };

type OrtTensor = {
  data: ArrayLike<number>;
};

type OrtSession = {
  inputNames: string[];
  outputNames: string[];
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>>;
};

type OrtModule = {
  env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
      simd: boolean;
    };
  };
  Tensor: new (
    type: string,
    data: Float32Array,
    dims: number[],
  ) => unknown;
  InferenceSession: {
    create: (
      model: ArrayBuffer,
      options: Record<string, unknown>,
    ) => Promise<OrtSession>;
  };
};

type LoadedSession = {
  model: ModelArtifact;
  session: OrtSession;
};

let ortModulePromise: Promise<OrtModule> | undefined;
let sessionsPromise: Promise<LoadedSession[]> | undefined;
let activeBackend: InferenceBackend = { kind: "none" };

async function loadOrt(): Promise<OrtModule> {
  if (!ortModulePromise) {
    ortModulePromise = import("onnxruntime-web").then((mod) => {
      const resolved = ("default" in mod ? mod.default : mod) as unknown;
      return resolved as OrtModule;
    });
  }
  return ortModulePromise;
}

function configureWasmPaths(ort: OrtModule): void {
  const base =
    typeof chrome !== "undefined" && chrome.runtime?.getURL
      ? chrome.runtime.getURL("ort/")
      : "/ort/";
  ort.env.wasm.wasmPaths = base;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

async function createOneSession(
  ort: OrtModule,
  model: ModelArtifact,
  providers: string[],
): Promise<OrtSession> {
  const modelBytes = await readCachedModel(model);
  if (!modelBytes) {
    throw new Error(`${model.id} missing from cache. Run setup first.`);
  }
  try {
    return await ort.InferenceSession.create(modelBytes, {
      executionProviders: providers,
      graphOptimizationLevel: model.graphOptimizationLevel,
    });
  } catch {
    return ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: model.graphOptimizationLevel,
    });
  }
}

async function createSessions(ort: OrtModule): Promise<LoadedSession[]> {
  configureWasmPaths(ort);

  const providers: string[] = [];
  try {
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      providers.push("webgpu");
    }
  } catch {
    // ignore
  }
  providers.push("wasm");

  const loaded: LoadedSession[] = [];
  for (const model of ALL_MODELS) {
    const session = await createOneSession(ort, model, providers);
    loaded.push({ model, session });
  }
  activeBackend =
    providers[0] === "webgpu" ? { kind: "webgpu" } : { kind: "wasm" };
  return loaded;
}

async function getSessions(): Promise<LoadedSession[]> {
  if (!sessionsPromise) {
    sessionsPromise = (async () => {
      const ort = await loadOrt();
      return createSessions(ort);
    })();
  }
  return sessionsPromise;
}

export function getVisualBackend(): InferenceBackend {
  return activeBackend;
}

export async function warmVisualClassifier(): Promise<InferenceBackend> {
  await getSessions();
  return activeBackend;
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

export { stubVisualClassify };

async function runModel(
  ort: OrtModule,
  loaded: LoadedSession,
  bitmap: ImageBitmap,
): Promise<number> {
  const imageData = await rasterizeForModel(bitmap, loaded.model.inputSize);
  const tensorData = imageDataToNchwFloat32(
    imageData,
    loaded.model.mean,
    loaded.model.std,
  );
  const inputName = loaded.session.inputNames[0];
  if (!inputName) throw new Error(`${loaded.model.id}: no inputs`);
  const feeds: Record<string, unknown> = {
    [inputName]: new ort.Tensor("float32", tensorData, [
      1,
      3,
      loaded.model.inputSize,
      loaded.model.inputSize,
    ]),
  };
  const output = await loaded.session.run(feeds);
  const outName = loaded.session.outputNames[0];
  if (!outName) throw new Error(`${loaded.model.id}: no outputs`);
  const outTensor = output[outName];
  if (!outTensor) throw new Error(`${loaded.model.id}: missing tensor`);
  const logits = Array.from(outTensor.data, (v) => Number(v));
  const [p0, p1] = softmax2(logits[0] ?? 0, logits[1] ?? 0);
  return loaded.model.aiLabelIndex === 0 ? p0 : p1;
}

export async function classifyVisual(
  bitmap: ImageBitmap,
  options?: { stub?: boolean },
): Promise<VisualClassification> {
  if (options?.stub) {
    const imageData = await rasterizeForModel(bitmap, VISUAL_MODEL.inputSize);
    return stubVisualClassify(imageData);
  }

  const ort = await loadOrt();
  const sessions = await getSessions();
  const scores = new Map<string, number>();
  for (const loaded of sessions) {
    scores.set(loaded.model.id, await runModel(ort, loaded, bitmap));
  }

  const distilled = scores.get(DISTILLED_MODEL.id);
  const forensics = scores.get(FORENSICS_MODEL.id);
  if (distilled === undefined) {
    throw new Error("Distilled visual score missing");
  }

  return {
    score: asAiConfidence(distilled),
    ...(forensics !== undefined
      ? { secondaryScore: asAiConfidence(forensics) }
      : {}),
    backend: activeBackend.kind === "none" ? { kind: "wasm" } : activeBackend,
    detail: [...scores.entries()]
      .map(([id, v]) => `${id}=${v.toFixed(3)}`)
      .join(","),
  };
}

export function resetVisualClassifierForTests(): void {
  sessionsPromise = undefined;
  ortModulePromise = undefined;
  activeBackend = { kind: "none" };
}
