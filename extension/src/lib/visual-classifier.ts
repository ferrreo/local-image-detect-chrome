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
    webgpu?: {
      powerPreference?: "high-performance" | "low-power";
      forceFallbackAdapter?: boolean;
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

let forensicsBackend: InferenceBackend = { kind: "none" };
/** Why WebGPU was skipped / abandoned (empty when WebGPU distilled is active). */
let webgpuSkipReason = "";

export function getForensicsBackend(): InferenceBackend {
  return forensicsBackend;
}

type LoadedSession = {
  model: ModelArtifact;
  session: OrtSession;
  backend: InferenceBackend;
};

export type WebGpuCaps = {
  /** `navigator.gpu.requestAdapter` returned something. */
  available: boolean;
  /** Adapter reports WGSL shader-f16 (required for distilled fp16). */
  shaderF16: boolean;
  /** Fallback / SwiftShader-class adapter — unusable for ML latency. */
  software: boolean;
  /**
   * Prefer WebGPU for distilled only when this is true.
   * Requires a non-software adapter with shader-f16.
   */
  usableForMl: boolean;
};

type GpuAdapterLike = {
  features?: { has: (feature: string) => boolean };
  isFallbackAdapter?: boolean;
  info?: {
    isFallbackAdapter?: boolean;
    vendor?: string;
    architecture?: string;
    description?: string;
  };
  requestAdapterInfo?: () => Promise<{
    isFallbackAdapter?: boolean;
    vendor?: string;
    architecture?: string;
    description?: string;
  }>;
};

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: (opts?: {
      powerPreference?: string;
    }) => Promise<GpuAdapterLike | null>;
  };
};

const NO_GPU: WebGpuCaps = {
  available: false,
  shaderF16: false,
  software: false,
  usableForMl: false,
};

/** Second warm distilled run above this → abandon WebGPU (software/broken EP). */
const WEBGPU_DISTILLED_BUDGET_MS = 600;

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

async function classifyAdapter(adapter: GpuAdapterLike): Promise<WebGpuCaps> {
  const shaderF16 = adapter.features?.has("shader-f16") === true;
  let info = adapter.info;
  if (!info && typeof adapter.requestAdapterInfo === "function") {
    try {
      info = await adapter.requestAdapterInfo();
    } catch {
      // older Chromium
    }
  }
  const blob = [
    info?.vendor,
    info?.architecture,
    info?.description,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  const software =
    adapter.isFallbackAdapter === true ||
    info?.isFallbackAdapter === true ||
    /swiftshader|llvmpipe|soft\s*raster|microsoft basic render/.test(blob);
  return {
    available: true,
    shaderF16,
    software,
    usableForMl: shaderF16 && !software,
  };
}

/** Probe WebGPU once: adapter + f16 + software/fallback heuristics. */
export async function probeWebGpuCapabilities(): Promise<WebGpuCaps> {
  if (!isGpuAvailable()) return NO_GPU;
  if (!gpuCapsPromise) {
    gpuCapsPromise = (async (): Promise<WebGpuCaps> => {
      const gpu = (navigator as GpuNavigator).gpu;
      if (!gpu?.requestAdapter) return NO_GPU;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const adapter = await gpu.requestAdapter({
            powerPreference: "high-performance",
          });
          if (adapter) return classifyAdapter(adapter);
        } catch {
          // retry
        }
        await new Promise((r) => setTimeout(r, 50 * (attempt + 1)));
      }
      return NO_GPU;
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

function configureOrt(ort: OrtModule): void {
  const base =
    typeof chrome !== "undefined" && chrome.runtime?.getURL
      ? chrome.runtime.getURL("ort/")
      : "/ort/";
  ort.env.wasm.wasmPaths = base;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  if (ort.env.webgpu) {
    ort.env.webgpu.powerPreference = "high-performance";
    ort.env.webgpu.forceFallbackAdapter = false;
  }
}

async function providerList(
  prefer: VisualProvider["kind"],
): Promise<string[]> {
  if (prefer === "wasm") return ["wasm"];
  const caps = await probeWebGpuCapabilities();
  // Only advertise WebGPU when fp16 distilled can actually run there.
  // Adapter-without-f16 used to select fp32-on-WebGPU and tank latency.
  if (prefer === "webgpu") {
    return caps.usableForMl ? ["webgpu", "wasm"] : ["wasm"];
  }
  return caps.usableForMl ? ["webgpu", "wasm"] : ["wasm"];
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

function webgpuEp(withAccF32: boolean): EpConfig {
  if (!withAccF32) return "webgpu";
  return {
    name: "webgpu",
    // Portable strict-rounding fix from microsoft/onnxruntime#29599.
    preferredMatmulAccumulatorPrecision: "f32",
  };
}

async function createOneSession(
  ort: OrtModule,
  model: ModelArtifact,
  providers: string[],
  opts?: { matmulAccF32?: boolean },
): Promise<{ session: OrtSession; backend: InferenceBackend }> {
  const modelBytes = await readCachedModel(model);
  if (!modelBytes) {
    throw new Error(`${model.id} missing from cache. Run setup first.`);
  }

  let lastError: unknown;
  for (const provider of providers) {
    const attempts: EpConfig[] =
      provider === "webgpu"
        ? opts?.matmulAccF32
          ? [webgpuEp(true), webgpuEp(false)]
          : [webgpuEp(false)]
        : [provider];
    for (const ep of attempts) {
      try {
        const session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: [ep],
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
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(`Failed to create ORT session for ${model.id}`);
}

async function warmBitmap(size: number): Promise<ImageBitmap> {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("OffscreenCanvas 2d unavailable");
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, size, size);
  return canvas.transferToImageBitmap();
}

/**
 * Compile shaders / settle EP, then time a second run. Pathological WebGPU
 * (software, broken Dawn) loses to WASM fp16 by a wide margin.
 */
async function webgpuDistilledIsFastEnough(
  ort: OrtModule,
  loaded: LoadedSession,
): Promise<boolean> {
  const bitmap = await warmBitmap(loaded.model.inputSize);
  try {
    await runModel(ort, loaded, bitmap);
    const t0 = performance.now();
    await runModel(ort, loaded, bitmap);
    const ms = performance.now() - t0;
    return ms <= WEBGPU_DISTILLED_BUDGET_MS;
  } finally {
    bitmap.close();
  }
}

/**
 * Pick distilled artifact + EP for latency-safe cascade:
 * - usable WebGPU (shader-f16, not software) → fp16 distilled on WebGPU
 * - otherwise → fp16 distilled on WASM (never fp32-on-WebGPU)
 * - Community Forensics q4 → WASM (WebGPU+f32 MatMulNBits is multi-second)
 *
 * ORT #29599 WebGPU CF remains available via createOneSession(..., {matmulAccF32})
 * but is not the default cascade path.
 */
async function createSessions(ort: OrtModule): Promise<LoadedSession[]> {
  configureOrt(ort);
  const caps = await probeWebGpuCapabilities();
  const providers = await providerList(preferredProvider);
  const wantWebgpu = providers[0] === "webgpu";
  webgpuSkipReason = "";

  if (preferredProvider !== "wasm" && !caps.usableForMl) {
    if (!caps.available) webgpuSkipReason = "no-adapter";
    else if (caps.software) webgpuSkipReason = "software-adapter";
    else if (!caps.shaderF16) webgpuSkipReason = "no-shader-f16";
  }

  const distilledModel: ModelArtifact = DISTILLED_MODEL;
  const distilledProviders = wantWebgpu ? ["webgpu", "wasm"] : ["wasm"];

  const loaded: LoadedSession[] = [];
  let distilled = await createOneSession(
    ort,
    distilledModel,
    distilledProviders,
  );

  if (distilled.backend.kind === "webgpu") {
    const ok = await webgpuDistilledIsFastEnough(ort, {
      model: distilledModel,
      session: distilled.session,
      backend: distilled.backend,
    });
    if (!ok) {
      webgpuSkipReason = `distilled>${WEBGPU_DISTILLED_BUDGET_MS}ms`;
      // Software / misconfigured WebGPU: prefer WASM fp16 over multi-second GPU.
      distilled = await createOneSession(ort, distilledModel, ["wasm"]);
    }
  }

  loaded.push({
    model: distilledModel,
    session: distilled.session,
    backend: distilled.backend,
  });

  // CF stays on WASM — q4 @ 384 with f32 MatMulNBits on WebGPU was ~10–25s/img.
  const forensics = await createOneSession(ort, FORENSICS_MODEL, ["wasm"]);
  forensicsBackend = forensics.backend;
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
      `distilledEp=${(activeBackend.kind === "none" ? "wasm" : activeBackend.kind)}`,
      `forensicsEp=${forensicsBackend.kind === "none" ? "wasm" : forensicsBackend.kind}`,
      ...(webgpuSkipReason ? [`webgpuSkip=${webgpuSkipReason}`] : []),
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
  forensicsBackend = { kind: "none" };
  webgpuSkipReason = "";
  gpuCapsPromise = undefined;
}

/** Drop sessions so the next warm recreates with the current provider preference. */
export function resetVisualClassifier(): void {
  sessionsPromise = undefined;
  activeBackend = { kind: "none" };
  forensicsBackend = { kind: "none" };
  webgpuSkipReason = "";
  // Keep gpuCapsPromise — re-probing every reset is what made GPU look flaky.
}
