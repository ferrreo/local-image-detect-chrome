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
const CONCEAL_PENDING = "truepixel-conceal-pending";

type TrackedImage = {
  id: string;
  element: HTMLImageElement;
  src: string;
  inFlight: boolean;
  result?: DetectionResult;
  /** User clicked the badge to temporarily show a hidden image. */
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
  img.classList.remove(CONCEAL_BLUR, CONCEAL_BLANK, CONCEAL_PENDING);
}

/** True until we have a decisive AI / Real label (not pending / uncertain). */
function isUndecided(entry: TrackedImage): boolean {
  if (entry.inFlight) return true;
  const kind = entry.result?.label.kind;
  return kind === undefined || kind === "uncertain" || kind === "error";
}

function applyConcealment(entry: TrackedImage): void {
  const img = entry.element;
  clearConcealClasses(img);
  if (entry.revealed) return;

  // Blank placeholder until sure — don't flash AI pixels during refine.
  if (isUndecided(entry)) {
    img.classList.add(CONCEAL_PENDING);
    return;
  }

  if (entry.result?.label.kind !== "ai") return;
  const mode: AiConcealMode = options.aiConceal;
  if (mode === "blur") img.classList.add(CONCEAL_BLUR);
  else if (mode === "blank") img.classList.add(CONCEAL_BLANK);
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
      if (!entry) return;
      // Allow peek while pending/uncertain or when AI is concealed.
      const canToggle =
        isUndecided(entry) ||
        (entry.result?.label.kind === "ai" && options.aiConceal !== "none");
      if (!canToggle) return;
      entry.revealed = !entry.revealed;
      if (entry.result) renderResult(entry.element, entry.result);
      else {
        applyConcealment(entry);
        badge.classList.toggle("truepixel-clickable", true);
        badge.textContent = entry.revealed ? "hide" : "…";
        badge.title = entry.revealed
          ? "TruePixel: click to hide again"
          : "TruePixel: analyzing… click to peek";
      }
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
      badge.classList.add("truepixel-clickable");
      badge.textContent = entry?.revealed ? `? ${pct}% · hide` : `? ${pct}%`;
      badge.title = entry?.revealed
        ? `TruePixel: still checking (${pct}% AI). Click to hide.`
        : `TruePixel: checking (${pct}% AI) — image hidden until sure. Click to peek.`;
      break;
    case "error":
      badge.classList.add("truepixel-error");
      badge.classList.add("truepixel-clickable");
      badge.textContent = entry?.revealed ? "n/a · hide" : "n/a";
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

/** AI probability from a labeled result (Real badges store 1−p_ai). */
function aiScoreOf(result: DetectionResult): number {
  if (result.label.kind === "real") return 1 - result.confidence;
  if (result.label.kind === "error") return 0.5;
  return result.confidence;
}

/**
 * Distilled-only often underscores StyleGAN / hard gens (~0.5–0.62).
 * Community Forensics recovers those — refine when the fast pass is ambiguous.
 */
function needsForensicsRefine(result: DetectionResult): boolean {
  if (result.label.kind === "error") return false;
  if (result.tiers.some((t) => t.tier === "provenance" && t.shortCircuit)) {
    return false;
  }
  if (result.label.kind === "uncertain") return true;
  const pAi = aiScoreOf(result);
  return pAi >= 0.3 && pAi < 0.65;
}

async function requestAnalyze(
  img: HTMLImageElement,
  id: string,
  speedMode: "realtime" | "accurate",
): Promise<DetectionResult> {
  const response = (await chrome.runtime.sendMessage({
    kind: "analyze-image",
    requestId: newRequestId(),
    imageId: id,
    src: imageKey(img),
    width: img.naturalWidth || img.width,
    height: img.naturalHeight || img.height,
    speedMode,
  })) as AnalyzeImageResponse;
  if (response.kind !== "analyze-image-result") {
    throw new Error("analyze-image failed");
  }
  return response.result;
}

async function analyze(img: HTMLImageElement, id: string): Promise<void> {
  const entry = tracked.get(id);
  if (!entry || entry.inFlight) return;
  entry.inFlight = true;
  entry.revealed = false;
  const pendingBadge = ensureBadge(img, id);
  pendingBadge.classList.add("truepixel-pending", "truepixel-clickable");
  pendingBadge.textContent = "…";
  pendingBadge.title = "TruePixel: analyzing… image hidden until sure";
  applyConcealment(entry);

  try {
    // Fast badge from distilled+spectral, then CF cascade when ambiguous.
    const quick = await requestAnalyze(img, id, "realtime");
    entry.inFlight = needsForensicsRefine(quick);
    entry.result = quick;
    renderResult(img, quick);

    if (needsForensicsRefine(quick)) {
      const refined = await requestAnalyze(img, id, "accurate");
      // Ignore stale refine if the img src changed mid-flight.
      if (tracked.get(id)?.src === entry.src) {
        entry.result = refined;
        entry.inFlight = false;
        renderResult(img, refined);
      }
    } else {
      entry.inFlight = false;
      applyConcealment(entry);
    }
  } catch (error) {
    entry.inFlight = false;
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
    const current = tracked.get(id);
    if (current) current.inFlight = false;
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
    const entry: TrackedImage = {
      id,
      element: img,
      src,
      inFlight: false,
      revealed: false,
      ...(existing?.result ? { result: existing.result } : {}),
    };
    tracked.set(id, entry);
    if (!entry.result) applyConcealment(entry);
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
