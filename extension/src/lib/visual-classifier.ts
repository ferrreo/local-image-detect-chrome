import { VISUAL_MODEL } from "./model-manifest";
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

let ortModulePromise: Promise<OrtModule> | undefined;
let sessionPromise: Promise<OrtSession> | undefined;
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

async function createSession(ort: OrtModule): Promise<OrtSession> {
  const modelBytes = await readCachedModel(VISUAL_MODEL);
  if (!modelBytes) {
    throw new Error("Visual model missing from cache. Run setup first.");
  }

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

  try {
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
    activeBackend =
      providers[0] === "webgpu" ? { kind: "webgpu" } : { kind: "wasm" };
    return session;
  } catch (webgpuError) {
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    activeBackend = { kind: "wasm" };
    void webgpuError;
    return session;
  }
}

async function getSession(): Promise<OrtSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await loadOrt();
      return createSession(ort);
    })();
  }
  return sessionPromise;
}

export function getVisualBackend(): InferenceBackend {
  return activeBackend;
}

export async function warmVisualClassifier(): Promise<InferenceBackend> {
  await getSession();
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

export async function classifyVisual(
  bitmap: ImageBitmap,
  options?: { stub?: boolean },
): Promise<VisualClassification> {
  const imageData = await rasterizeForModel(bitmap, VISUAL_MODEL.inputSize);

  if (options?.stub) {
    return stubVisualClassify(imageData);
  }

  const ort = await loadOrt();
  const session = await getSession();
  const tensorData = imageDataToNchwFloat32(
    imageData,
    VISUAL_MODEL.mean,
    VISUAL_MODEL.std,
  );

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

  const output = await session.run(feeds);
  const outName = session.outputNames[0];
  if (!outName) throw new Error("Model has no outputs");
  const outTensor = output[outName];
  if (!outTensor) throw new Error("Missing output tensor");

  const logits = Array.from(outTensor.data, (v) => Number(v));
  if (logits.length < 2) {
    throw new Error(`Unexpected logits length ${logits.length}`);
  }

  const [p0, p1] = softmax2(logits[0] ?? 0, logits[1] ?? 0);
  const aiProb = VISUAL_MODEL.aiLabelIndex === 0 ? p0 : p1;

  return {
    score: asAiConfidence(aiProb),
    backend: activeBackend.kind === "none" ? { kind: "wasm" } : activeBackend,
    detail: `logits=[${(logits[0] ?? 0).toFixed(3)},${(logits[1] ?? 0).toFixed(3)}]`,
  };
}

export function resetVisualClassifierForTests(): void {
  sessionPromise = undefined;
  ortModulePromise = undefined;
  activeBackend = { kind: "none" };
}
