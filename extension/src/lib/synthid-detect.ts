import { asAiConfidence, type AiConfidence } from "../shared/types";
import { OPENSYNTHID_MODEL } from "./model-manifest";
import { ensureModelsReady, readCachedModel } from "./model-cache";
import {
  buildSynthIdInput,
  resizeToSynthIdSquare,
  SYNTHID_INPUT_SIZE,
} from "./synthid-preprocess";

export type SynthIdHit = {
  score: AiConfidence;
  detail: string;
  shortCircuit: boolean;
};

type OrtModule = {
  Tensor: new (
    type: string,
    data: Float32Array,
    dims: number[],
  ) => { data: Float32Array };
  InferenceSession: {
    create: (
      model: ArrayBuffer,
      options?: { executionProviders?: string[]; graphOptimizationLevel?: string },
    ) => Promise<{
      run: (
        feeds: Record<string, unknown>,
      ) => Promise<Record<string, { data: Float32Array }>>;
      inputNames: string[];
      outputNames: string[];
    }>;
  };
};

let ortModule: OrtModule | undefined;
let sessionPromise:
  | Promise<{
      run: (
        feeds: Record<string, unknown>,
      ) => Promise<Record<string, { data: Float32Array }>>;
      inputNames: string[];
      outputNames: string[];
    } | null>
  | undefined;

async function loadOrt(): Promise<OrtModule | null> {
  if (ortModule) return ortModule;
  try {
    const mod = (await import("onnxruntime-web")) as unknown as OrtModule;
    ortModule = mod;
    return mod;
  } catch {
    return null;
  }
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      let bytes = await readCachedModel(OPENSYNTHID_MODEL);
      if (!bytes) {
        await ensureModelsReady();
        bytes = await readCachedModel(OPENSYNTHID_MODEL);
      }
      if (!bytes) return null;
      const ort = await loadOrt();
      if (!ort) return null;
      try {
        return await ort.InferenceSession.create(bytes, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: OPENSYNTHID_MODEL.graphOptimizationLevel,
        });
      } catch {
        return null;
      }
    })();
  }
  return sessionPromise;
}

/**
 * Pixel-level SynthID surrogate (OpenSynthID). Short-circuits when the
 * watermark probability clears the threshold. Soft-fails (no SC) when the
 * model is not cached yet — Soft Binding metadata remains the open standard
 * path for Google/OpenAI marks.
 */
export async function analyzeSynthIdPixels(
  imageData: ImageData,
  threshold = 0.55,
): Promise<SynthIdHit> {
  if (imageData.width < 64 || imageData.height < 64) {
    return {
      score: asAiConfidence(0.5),
      detail: "synthid-pixel:too-small",
      shortCircuit: false,
    };
  }

  const session = await getSession();
  if (!session) {
    return {
      score: asAiConfidence(0.5),
      detail: "synthid-pixel:model-unavailable",
      shortCircuit: false,
    };
  }

  const ort = await loadOrt();
  if (!ort) {
    return {
      score: asAiConfidence(0.5),
      detail: "synthid-pixel:ort-unavailable",
      shortCircuit: false,
    };
  }

  const { r, g, b } = await resizeToSynthIdSquare(imageData, SYNTHID_INPUT_SIZE);
  const input = buildSynthIdInput(r, g, b, SYNTHID_INPUT_SIZE);
  const tensor = new ort.Tensor("float32", input, [
    1,
    6,
    SYNTHID_INPUT_SIZE,
    SYNTHID_INPUT_SIZE,
  ]);
  const feeds: Record<string, unknown> = {
    [session.inputNames[0] ?? "input"]: tensor,
  };
  const result = await session.run(feeds);
  const outName = session.outputNames[0] ?? "probability";
  const raw = result[outName]?.data[0] ?? 0.5;
  const prob = asAiConfidence(Math.min(1, Math.max(0, raw)));
  if (prob >= threshold) {
    return {
      score: asAiConfidence(Math.max(prob, 0.95)),
      detail: `watermark-synthid-pixel:${prob.toFixed(3)}`,
      shortCircuit: true,
    };
  }
  return {
    score: asAiConfidence(0.5 + (prob - 0.5) * 0.2),
    detail: `synthid-pixel:${prob.toFixed(3)}`,
    shortCircuit: false,
  };
}

/** Test helper to clear cached session between unit runs. */
export function resetSynthIdSessionForTests(): void {
  sessionPromise = undefined;
}
