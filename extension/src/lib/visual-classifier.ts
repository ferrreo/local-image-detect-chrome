import {
  DISTILLED_MODEL,
  DISTILLED_MODEL_FP32,
  FORENSICS_MODEL,
  VISUAL_MODEL,
  isDistilledModelId,
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
  backend: InferenceBackend;
};

export type WebGpuCaps = {
  available: boolean;
  /** Adapter reports WGSL shader-f16. Required for distilled fp16 on WebGPU. */
  shaderF16: boolean;
};

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: (opts?: {
      powerPreference?: string;
    }) => Promise<{
      features?: { has: (feature: string) => boolean };
    } | null>;
  };
};

let ortModulePromise: Promise<OrtModule> | undefined;
let sessionsPromise: Promise<LoadedSession[]> | undefined;
let activeBackend: InferenceBackend = { kind: "none" };
let preferredProvider: VisualProvider["kind"] = "auto";
/** Cached adapter probe — `'gpu' in navigator` alone is flaky under headless. */
let gpuCapsPromise: Promise<WebGpuCaps> | undefined;

export function isGpuAvailable(): boolean {
  try {
    return typeof navigator !== "undefined" && "gpu" in navigator;
  } catch {
    return false;
  }
}

/** Probe WebGPU once: adapter presence + shader-f16. */
export async function probeWebGpuCapabilities(): Promise<WebGpuCaps> {
  if (!isGpuAvailable()) return { available: false, shaderF16: false };
  if (!gpuCapsPromise) {
    gpuCapsPromise = (async (): Promise<WebGpuCaps> => {
      const gpu = (navigator as GpuNavigator).gpu;
      if (!gpu?.requestAdapter) return { available: false, shaderF16: false };
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const adapter = await gpu.requestAdapter({
            powerPreference: "high-performance",
          });
          if (adapter) {
            return {
              available: true,
              shaderF16: adapter.features?.has("shader-f16") === true,
            };
          }
        } catch {
          // retry
        }
        await new Promise((r) => setTimeout(r, 50 * (attempt + 1)));
      }
      return { available: false, shaderF16: false };
    })();
  }
  return gpuCapsPromise;
}

export async function probeWebGpuAvailable(): Promise<boolean> {
  return (await probeWebGpuCapabilities()).available;
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
    // WebGPU entry includes WASM fallback providers.
    ortModulePromise = import("onnxruntime-web/webgpu").then((mod) => {
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

async function providerList(
  prefer: VisualProvider["kind"],
): Promise<string[]> {
  if (prefer === "wasm") return ["wasm"];
  const caps = await probeWebGpuCapabilities();
  if (prefer === "webgpu") {
    // Prefer WebGPU but keep WASM fallback — adapter presence flickers
    // under headless / early offscreen boot.
    return caps.available ? ["webgpu", "wasm"] : ["wasm"];
  }
  return caps.available ? ["webgpu", "wasm"] : ["wasm"];
}

type EpConfig =
  | string
  | {
      name: string;
      /**
       * ORT #29599 — MatMulNBits f32 accumulators (needed for q4 CF on WebGPU).
       * Ignored on builds that lack the option.
       */
      preferredMatmulAccumulatorPrecision?: "f16" | "f32";
    };

function epConfigs(providers: string[]): EpConfig[] {
  return providers.map((provider) => {
    if (provider === "webgpu") {
      return {
        name: "webgpu",
        // Portable strict-rounding fix from microsoft/onnxruntime#29599.
        preferredMatmulAccumulatorPrecision: "f32",
      };
    }
    return provider;
  });
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
        executionProviders: epConfigs([provider]),
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

/**
 * Pick distilled artifact + EP so WebGPU never sees an unsupported graph.
 * - WebGPU + shader-f16 → fp16 on WebGPU (preferred)
 * - WebGPU without f16 → fp32 on WebGPU (fp16 Transpose requires shader-f16)
 * - WASM → fp16
 *
 * Community Forensics q4: prefer WebGPU with ORT #29599
 * `preferredMatmulAccumulatorPrecision: "f32"`, fall back to WASM.
 */
async function createSessions(ort: OrtModule): Promise<LoadedSession[]> {
  configureWasmPaths(ort);
  const providers = await providerList(preferredProvider);
  const caps = await probeWebGpuCapabilities();
  const wantWebgpu = providers[0] === "webgpu";

  let distilledModel: ModelArtifact = DISTILLED_MODEL;
  let distilledProviders = providers;
  if (wantWebgpu && caps.available) {
    if (caps.shaderF16) {
      distilledModel = DISTILLED_MODEL;
      distilledProviders = ["webgpu", "wasm"];
    } else {
      distilledModel = DISTILLED_MODEL_FP32;
      // fp32 on WebGPU; if session create fails, fall back to fp16 WASM.
      distilledProviders = ["webgpu", "wasm"];
    }
  } else {
    distilledModel = DISTILLED_MODEL;
    distilledProviders = ["wasm"];
  }

  const loaded: LoadedSession[] = [];
  try {
    const distilled = await createOneSession(
      ort,
      distilledModel,
      distilledProviders,
    );
    loaded.push({
      model: distilledModel,
      session: distilled.session,
      backend: distilled.backend,
    });
  } catch (error) {
    if (distilledModel.id !== DISTILLED_MODEL.id) {
      const fallback = await createOneSession(ort, DISTILLED_MODEL, ["wasm"]);
      loaded.push({
        model: DISTILLED_MODEL,
        session: fallback.session,
        backend: fallback.backend,
      });
    } else {
      throw error;
    }
  }

  // CF q4 on WebGPU needs #29599 f32 MatMulNBits accumulators; else WASM.
  const forensicsProviders =
    wantWebgpu && caps.available ? ["webgpu", "wasm"] : ["wasm"];
  const forensics = await createOneSession(
    ort,
    FORENSICS_MODEL,
    forensicsProviders,
  );
  loaded.push({
    model: FORENSICS_MODEL,
    session: forensics.session,
    backend: forensics.backend,
  });

  const distilledLoaded = loaded.find((s) => isDistilledModelId(s.model.id));
  activeBackend = distilledLoaded?.backend ?? { kind: "wasm" };
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
  // Prime adapter cache before session create so prefer=webgpu is stable.
  await probeWebGpuCapabilities();
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
  const distilledSession = sessions.find((s) => isDistilledModelId(s.model.id));
  const forensicsSession = sessions.find(
    (s) => s.model.id === FORENSICS_MODEL.id,
  );
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

  let distilledMs = 0;
  let forensicsMs = 0;
  const scores = new Map<string, number>();
  if (runDistilled) {
    const t0 = performance.now();
    scores.set(
      DISTILLED_MODEL.id,
      await runModel(ort, distilledSession, bitmap),
    );
    distilledMs = performance.now() - t0;
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
    const t0 = performance.now();
    scores.set(
      FORENSICS_MODEL.id,
      await runModel(ort, forensicsSession, bitmap),
    );
    forensicsMs = performance.now() - t0;
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
      distilledSession.model.id === DISTILLED_MODEL_FP32.id
        ? "distilled=fp32"
        : "distilled=fp16",
      cascade ? "cascade" : "dual",
      runForensics ? "ranForensics" : "skipForensics",
      `distilledMs=${distilledMs.toFixed(1)}`,
      `forensicsMs=${forensicsMs.toFixed(1)}`,
    ].join(","),
    distilledMs,
    forensicsMs,
  };
}

export function resetVisualClassifierForTests(): void {
  sessionsPromise = undefined;
  ortModulePromise = undefined;
  activeBackend = { kind: "none" };
  gpuCapsPromise = undefined;
}

/** Drop sessions so the next warm recreates with the current provider preference. */
export function resetVisualClassifier(): void {
  sessionsPromise = undefined;
  activeBackend = { kind: "none" };
  // Keep gpuCapsPromise — re-probing every reset is what made GPU look flaky.
}
