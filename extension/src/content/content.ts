import { newRequestId } from "../shared/messages";
import type {
  AiConcealMode,
  AnalyzeImageResponse,
  DetectionResult,
  ExtensionOptions,
} from "../shared/types";
import {
  asAiConfidence,
  DEFAULT_OPTIONS,
  parseAiConcealMode,
} from "../shared/types";

const MIN_SIDE = 96;
const OVERLAY_ATTR = "data-truepixel-id";
const BADGE_CLASS = "truepixel-badge";
const CONCEAL_BLUR = "truepixel-conceal-blur";
const CONCEAL_BLANK = "truepixel-conceal-blank";

type TrackedImage = {
  id: string;
  element: HTMLImageElement;
  src: string;
  inFlight: boolean;
  result?: DetectionResult;
  /** User clicked the AI badge to temporarily show the image. */
  revealed: boolean;
};

const tracked = new Map<string, TrackedImage>();
let options: ExtensionOptions = { ...DEFAULT_OPTIONS };
let observer: MutationObserver | undefined;
let scanTimer: number | undefined;

function imageKey(img: HTMLImageElement): string {
  return img.currentSrc || img.src || "";
}

function isEligible(img: HTMLImageElement): boolean {
  if (!img.isConnected) return false;
  const src = imageKey(img);
  if (!src || src.startsWith("data:image/svg")) return false;
  const rect = img.getBoundingClientRect();
  if (rect.width < MIN_SIDE || rect.height < MIN_SIDE) return false;
  // Skip tiny decorative icons even if CSS scales them.
  if (img.naturalWidth > 0 && img.naturalWidth < MIN_SIDE) return false;
  if (img.naturalHeight > 0 && img.naturalHeight < MIN_SIDE) return false;
  return true;
}

function clearConcealClasses(img: HTMLImageElement): void {
  img.classList.remove(CONCEAL_BLUR, CONCEAL_BLANK);
}

function applyConcealment(entry: TrackedImage): void {
  const img = entry.element;
  clearConcealClasses(img);
  const mode: AiConcealMode = options.aiConceal;
  const isAi = entry.result?.label.kind === "ai";
  if (!isAi || mode === "none" || entry.revealed) return;
  if (mode === "blur") img.classList.add(CONCEAL_BLUR);
  else img.classList.add(CONCEAL_BLANK);
}

function ensureBadge(img: HTMLImageElement, id: string): HTMLElement {
  const parent = img.parentElement ?? document.body;
  const computed = getComputedStyle(parent);
  if (computed.position === "static") {
    parent.style.position = "relative";
  }

  let badge = parent.querySelector<HTMLElement>(
    `.${BADGE_CLASS}[${OVERLAY_ATTR}="${id}"]`,
  );
  if (!badge) {
    badge = document.createElement("button");
    badge.type = "button";
    badge.className = `${BADGE_CLASS} truepixel-pending`;
    badge.setAttribute(OVERLAY_ATTR, id);
    badge.textContent = "…";
    badge.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const entry = tracked.get(id);
      if (!entry || entry.result?.label.kind !== "ai") return;
      if (options.aiConceal === "none") return;
      entry.revealed = !entry.revealed;
      renderResult(entry.element, entry.result);
    });
    parent.appendChild(badge);
  }
  positionBadge(img, badge);
  return badge;
}

function positionBadge(img: HTMLImageElement, badge: HTMLElement): void {
  const parent = img.parentElement ?? document.body;
  const imgRect = img.getBoundingClientRect();
  const parentRect = parent.getBoundingClientRect();
  badge.style.top = `${Math.max(0, imgRect.top - parentRect.top) + 6}px`;
  badge.style.left = `${Math.max(0, imgRect.left - parentRect.left) + 6}px`;
}

function renderResult(img: HTMLImageElement, result: DetectionResult): void {
  const entry = tracked.get(result.imageId);
  const badge = ensureBadge(img, result.imageId);
  badge.classList.remove(
    "truepixel-pending",
    "truepixel-ai",
    "truepixel-real",
    "truepixel-uncertain",
    "truepixel-error",
    "truepixel-clickable",
  );

  const pct = Math.round(result.confidence * 100);
  const timingHint = result.timing
    ? ` · ${result.timing.totalMs.toFixed(0)}ms` +
      (result.timing.ranForensics
        ? ` (cf ${result.timing.forensicsMs.toFixed(0)}ms)`
        : "")
    : result.elapsedMs
      ? ` · ${result.elapsedMs.toFixed(0)}ms`
      : "";
  switch (result.label.kind) {
    case "ai": {
      badge.classList.add("truepixel-ai");
      const concealed =
        options.aiConceal !== "none" && entry && !entry.revealed;
      badge.textContent = concealed
        ? `AI ${pct}% · show`
        : entry?.revealed && options.aiConceal !== "none"
          ? `AI ${pct}% · hide`
          : `AI ${pct}%`;
      badge.title = concealed
        ? `TruePixel: likely AI-generated (${pct}%)${timingHint}. Click to reveal.`
        : entry?.revealed && options.aiConceal !== "none"
          ? `TruePixel: revealed. Click to hide again.${timingHint}`
          : `TruePixel: likely AI-generated (${pct}% confidence)${timingHint}`;
      if (options.aiConceal !== "none") {
        badge.classList.add("truepixel-clickable");
      }
      break;
    }
    case "real":
      badge.classList.add("truepixel-real");
      badge.textContent = `Real ${pct}%`;
      badge.title = `TruePixel: likely real photograph (${pct}% confidence)${timingHint}`;
      break;
    case "uncertain":
      badge.classList.add("truepixel-uncertain");
      badge.textContent = `? ${pct}%`;
      badge.title = `TruePixel: uncertain (${pct}% AI confidence)${timingHint}`;
      break;
    case "error":
      badge.classList.add("truepixel-error");
      badge.textContent = "n/a";
      badge.title = `TruePixel error: ${result.label.message}`;
      break;
    default: {
      const _exhaustive: never = result.label;
      void _exhaustive;
    }
  }

  if (entry) {
    entry.result = result;
    applyConcealment(entry);
  }
}

async function analyze(img: HTMLImageElement, id: string): Promise<void> {
  const entry = tracked.get(id);
  if (!entry || entry.inFlight) return;
  entry.inFlight = true;
  ensureBadge(img, id);

  try {
    const response = (await chrome.runtime.sendMessage({
      kind: "analyze-image",
      requestId: newRequestId(),
      imageId: id,
      src: imageKey(img),
      width: img.naturalWidth || img.width,
      height: img.naturalHeight || img.height,
      // Overlay must stay snappy; CF on ORT WASM is ~1s+ per image.
      speedMode: "realtime",
    })) as AnalyzeImageResponse;

    if (response.kind !== "analyze-image-result") return;
    entry.result = response.result;
    renderResult(img, response.result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    renderResult(img, {
      imageId: id,
      label: { kind: "error", message },
      confidence: asAiConfidence(0.5),
      tiers: [],
      backend: { kind: "none" },
      elapsedMs: 0,
    });
  } finally {
    entry.inFlight = false;
  }
}

function discover(): void {
  if (!options.autoScan) return;
  const images = document.querySelectorAll("img");
  for (const img of images) {
    if (!(img instanceof HTMLImageElement)) continue;
    if (!isEligible(img)) continue;
    const src = imageKey(img);
    let id = img.getAttribute(OVERLAY_ATTR);
    if (!id) {
      id = `tp_${Math.random().toString(16).slice(2)}_${tracked.size}`;
      img.setAttribute(OVERLAY_ATTR, id);
    }
    const existing = tracked.get(id);
    if (existing && existing.src === src && existing.result) {
      renderResult(img, existing.result);
      continue;
    }
    tracked.set(id, {
      id,
      element: img,
      src,
      inFlight: false,
      revealed: existing?.revealed ?? false,
      ...(existing?.result ? { result: existing.result } : {}),
    });
    void analyze(img, id);
  }
}

function reapplyAllResults(): void {
  for (const entry of tracked.values()) {
    if (!entry.result) continue;
    renderResult(entry.element, entry.result);
  }
}

function scheduleScan(): void {
  if (scanTimer !== undefined) window.clearTimeout(scanTimer);
  scanTimer = window.setTimeout(() => {
    discover();
  }, 250);
}

async function refreshOptions(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({
      kind: "get-status",
      requestId: newRequestId(),
    });
    if (
      typeof response === "object" &&
      response !== null &&
      "autoScan" in response &&
      "threshold" in response
    ) {
      const aiConceal =
        "aiConceal" in response
          ? parseAiConcealMode(
              (response as { aiConceal?: unknown }).aiConceal,
            )
          : DEFAULT_OPTIONS.aiConceal;
      options = {
        ...options,
        autoScan: Boolean(
          (response as { autoScan?: unknown }).autoScan,
        ),
        threshold:
          Number((response as { threshold?: unknown }).threshold) ||
          DEFAULT_OPTIONS.threshold,
        aiConceal,
      };
    }
  } catch {
    // ignore
  }
}

function start(): void {
  void refreshOptions().then(() => {
    discover();
    observer = new MutationObserver(() => scheduleScan());
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset"],
    });
    window.addEventListener("scroll", scheduleScan, { passive: true });
    window.addEventListener("resize", scheduleScan);
  });
}

chrome.runtime.onMessage.addListener((message: unknown) => {
  if (typeof message !== "object" || message === null || !("kind" in message)) {
    return;
  }
  if (message.kind === "truepixel-rescan") {
    for (const entry of tracked.values()) {
      delete entry.result;
      entry.revealed = false;
      clearConcealClasses(entry.element);
    }
    scheduleScan();
    return;
  }
  if (message.kind === "truepixel-options") {
    void refreshOptions().then(() => {
      reapplyAllResults();
      scheduleScan();
    });
  }
});

start();
