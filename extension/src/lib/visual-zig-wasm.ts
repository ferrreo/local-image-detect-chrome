/**
 * Browser Zig+ORT WASM visual backend.
 *
 * Same cascade dual options as onnxruntime-web (`ClassifyVisualOptions`).
 * Requires `libonnxruntime_webassembly.a` linked into the Zig wasm module
 * (`tp_has_ort_session() === 1`). Until then this reports unavailable —
 * callers must not fall back to the heuristic stub.
 */
import type { InferenceBackend } from "../shared/types";
import type { ClassifyVisualOptions } from "./visual-classifier";
import type { VisualClassification } from "./visual-stub";
import { VISUAL_MODEL } from "./model-manifest";
import { rasterizeForModel } from "./image-decode";
import { stubVisualClassify } from "./visual-stub";

type ZigWasmExports = {
  memory: WebAssembly.Memory;
  tp_abi_version: () => number;
  tp_has_ort_session: () => number;
  tp_session_run?: (
    session: number,
    nchwPtr: number,
    size: number,
    aiLabel: number,
  ) => number;
};

let exportsPromise: Promise<ZigWasmExports | null> | undefined;

function wasmUrl(): string {
  if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
    return chrome.runtime.getURL("wasm/truepixel_infer.wasm");
  }
  return "/wasm/truepixel_infer.wasm";
}

async function loadExports(): Promise<ZigWasmExports | null> {
  if (!exportsPromise) {
    exportsPromise = (async () => {
      try {
        const result = await WebAssembly.instantiateStreaming(fetch(wasmUrl()), {
          env: {},
        });
        return result.instance.exports as unknown as ZigWasmExports;
      } catch {
        return null;
      }
    })();
  }
  return exportsPromise;
}

export async function isZigWasmOrtReady(): Promise<boolean> {
  const exp = await loadExports();
  return Boolean(exp && exp.tp_has_ort_session() === 1 && exp.tp_session_run);
}

export async function warmVisualZigWasm(): Promise<InferenceBackend> {
  const ready = await isZigWasmOrtReady();
  if (!ready) {
    throw new Error(
      "Zig+ORT WASM not linked (tp_has_ort_session=0). Build ORT static wasm and rebuild zig -Dwasm=true.",
    );
  }
  return { kind: "wasm" };
}

/**
 * Cascade-aware classify mirroring `classifyVisual` (onnxruntime-web).
 * Pipeline already passes spectral + cascade; session exports land with the
 * static ORT link.
 */
export async function classifyVisualZigWasm(
  bitmap: ImageBitmap,
  options?: ClassifyVisualOptions,
): Promise<VisualClassification> {
  if (options?.stub) {
    const imageData = await rasterizeForModel(bitmap, VISUAL_MODEL.inputSize);
    return stubVisualClassify(imageData);
  }

  const exp = await loadExports();
  if (!exp || exp.tp_has_ort_session() !== 1 || !exp.tp_session_run) {
    throw new Error(
      "Zig+ORT WASM sessions unavailable — refusing stub fallback",
    );
  }

  throw new Error(
    "Zig+ORT WASM session exports not implemented in this build yet",
  );
}

export function resetVisualZigWasm(): void {
  exportsPromise = undefined;
}
