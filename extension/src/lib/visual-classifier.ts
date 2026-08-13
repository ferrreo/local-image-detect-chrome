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
  type VisualProvider,
} from "../shared/types";
import {
  stubVisualClassify,
  type VisualClassification,
} from "./visual-stub";
import { needsForensicsCascade } from "./forensics-cascade";

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
let preferredProvider: VisualProvider["kind"] = "auto";

export function isGpuAvailable(): boolean {
  try {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  } catch {
    return false;
  }
}

export function setPreferredVisualProvider(
  provider: VisualProvider["kind"],
): void {
  preferredProvider = provider;
}

export function getPreferredVisualProvider(): VisualProvider["kind"] {
  return preferredProvider;
}

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

function providerList(prefer: VisualProvider["kind"]): string[] {
  const gpu = isGpuAvailable();
  if (prefer === "wasm") return ["wasm"];
  if (prefer === "webgpu") {
    if (!gpu) throw new Error("WebGPU requested but navigator.gpu is unavailable");
    return ["webgpu"];
  }
  // auto
  return gpu ? ["webgpu", "wasm"] : ["wasm"];
}

async function createOneSession(
  ort: OrtModule,
  model: ModelArtifact,
  providers: string[],
): Promise<{ session: OrtSession; backend: InferenceBackend }> {
  const modelBytes = await readCachedModel(model);
  if (!modelBytes) {
    throw new Error(`${model.id} missing from cache. Run setup first.`);
  }

  let lastError: unknown;
  for (const provider of providers) {
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [provider],
        graphOptimizationLevel: model.graphOptimizationLevel,
      });
      return {
        session,
        backend: provider === "webgpu" ? { kind: "webgpu" } : { kind: "wasm" },
      };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(`Failed to create ORT session for ${model.id}`);
}

async function createSessions(ort: OrtModule): Promise<LoadedSession[]> {
  configureWasmPaths(ort);
  const providers = providerList(preferredProvider);

  const loaded: LoadedSession[] = [];
  let backend: InferenceBackend = { kind: "wasm" };
  for (const model of ALL_MODELS) {
    const created = await createOneSession(ort, model, providers);
    loaded.push({ model, session: created.session });
    backend = created.backend;
  }
  activeBackend = backend;
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

export type ClassifyVisualOptions = {
  stub?: boolean;
  /**
   * Cascade dual (default true): run distilled, then Community Forensics only
   * when `needsForensicsCascade` fires. Set false for always-dual.
   */
  cascade?: boolean;
  spectral?: {
    score: number;
    laplacianVariance: number;
    chromaFlatness: number;
  };
  /** Force which heads to run. Overrides cascade when set. */
  runDistilled?: boolean;
  runForensics?: boolean;
};

export async function classifyVisual(
  bitmap: ImageBitmap,
  options?: ClassifyVisualOptions,
): Promise<VisualClassification> {
  if (options?.stub) {
    const imageData = await rasterizeForModel(bitmap, VISUAL_MODEL.inputSize);
    return stubVisualClassify(imageData);
  }

  const ort = await loadOrt();
  const sessions = await getSessions();
  const distilledSession = sessions.find((s) => s.model.id === DISTILLED_MODEL.id);
  const forensicsSession = sessions.find((s) => s.model.id === FORENSICS_MODEL.id);
  if (!distilledSession) {
    throw new Error("Distilled visual session missing");
  }

  const forceDistilled = options?.runDistilled;
  const forceForensics = options?.runForensics;
  const cascade = options?.cascade !== false;

  const runDistilled = forceDistilled !== false;
  let runForensics = forceForensics === true;
  if (forceForensics === undefined && forceDistilled === undefined) {
    // Default path: distilled always; forensics via cascade or always-dual.
    runForensics = !cascade;
  }

  const scores = new Map<string, number>();
  if (runDistilled) {
    scores.set(
      DISTILLED_MODEL.id,
      await runModel(ort, distilledSession, bitmap),
    );
  }

  const distilled = scores.get(DISTILLED_MODEL.id);
  if (runDistilled && distilled === undefined) {
    throw new Error("Distilled visual score missing");
  }

  if (
    forceForensics === undefined &&
    forceDistilled === undefined &&
    cascade &&
    forensicsSession &&
    distilled !== undefined &&
    options?.spectral
  ) {
    runForensics = needsForensicsCascade({
      distilled,
      spectral: options.spectral.score,
      laplacianVariance: options.spectral.laplacianVariance,
      chromaFlatness: options.spectral.chromaFlatness,
    });
  }

  if (runForensics) {
    if (!forensicsSession) {
      throw new Error("Community Forensics session missing");
    }
    scores.set(
      FORENSICS_MODEL.id,
      await runModel(ort, forensicsSession, bitmap),
    );
  }

  const forensics = scores.get(FORENSICS_MODEL.id);
  if (distilled === undefined && forensics === undefined) {
    throw new Error("No visual score produced");
  }
  const primary = distilled !== undefined ? distilled : forensics;
  if (primary === undefined) {
    throw new Error("No visual score produced");
  }

  return {
    // Primary head is always distilled when present (fusion expects that).
    score: asAiConfidence(primary),
    ...(forensics !== undefined
      ? { secondaryScore: asAiConfidence(forensics) }
      : {}),
    backend: activeBackend.kind === "none" ? { kind: "wasm" } : activeBackend,
    detail: [
      ...[...scores.entries()].map(([id, v]) => `${id}=${v.toFixed(3)}`),
      cascade ? "cascade" : "dual",
      runForensics ? "ranForensics" : "skipForensics",
    ].join(","),
  };
}

export function resetVisualClassifierForTests(): void {
  sessionsPromise = undefined;
  ortModulePromise = undefined;
  activeBackend = { kind: "none" };
}

/** Drop sessions so the next warm recreates with the current provider preference. */
export function resetVisualClassifier(): void {
  sessionsPromise = undefined;
  activeBackend = { kind: "none" };
}
