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
const STATE_ATTR = "data-truepixel-state";
const BADGE_CLASS = "truepixel-badge";
const VEIL_CLASS = "truepixel-veil";

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
  const attrW = Number(img.getAttribute("width")) || 0;
  const attrH = Number(img.getAttribute("height")) || 0;
  const w = img.naturalWidth || img.width || attrW;
  const h = img.naturalHeight || img.height || attrH;
  // Unknown size: still eligible (blur first, clear later) to avoid flash.
  if (w > 0 && w < MIN_SIDE) return false;
  if (h > 0 && h < MIN_SIDE) return false;
  const rect = img.getBoundingClientRect();
  if (rect.width > 0 && rect.width < MIN_SIDE) return false;
  if (rect.height > 0 && rect.height < MIN_SIDE) return false;
  return true;
}

function ensureParentPositioned(img: HTMLImageElement): HTMLElement {
  const parent = img.parentElement ?? document.body;
  const computed = getComputedStyle(parent);
  if (computed.position === "static") {
    parent.style.position = "relative";
  }
  return parent;
}

function positionOverImage(
  img: HTMLImageElement,
  el: HTMLElement,
  inset = 0,
): void {
  const parent = ensureParentPositioned(img);
  const imgRect = img.getBoundingClientRect();
  const parentRect = parent.getBoundingClientRect();
  const top = Math.max(0, imgRect.top - parentRect.top) + inset;
  const left = Math.max(0, imgRect.left - parentRect.left) + inset;
  el.style.top = `${top}px`;
  el.style.left = `${left}px`;
  if (inset === 0) {
    el.style.width = `${Math.max(0, imgRect.width)}px`;
    el.style.height = `${Math.max(0, imgRect.height)}px`;
  }
}

function removeVeil(id: string): void {
  document
    .querySelectorAll(`.${VEIL_CLASS}[${OVERLAY_ATTR}="${id}"]`)
    .forEach((node) => node.remove());
}

function ensureVeil(
  img: HTMLImageElement,
  id: string,
  mode: "blur" | "blank",
): HTMLElement {
  const parent = ensureParentPositioned(img);
  let veil = parent.querySelector<HTMLElement>(
    `.${VEIL_CLASS}[${OVERLAY_ATTR}="${id}"]`,
  );
  if (!veil) {
    veil = document.createElement("div");
    veil.className = VEIL_CLASS;
    veil.setAttribute(OVERLAY_ATTR, id);
    veil.setAttribute("aria-hidden", "true");
    parent.appendChild(veil);
  }
  veil.classList.toggle("truepixel-veil-blur", mode === "blur");
  veil.classList.toggle("truepixel-veil-blank", mode === "blank");
  positionOverImage(img, veil);
  return veil;
}

/**
 * Cover the image with a veil while pending, or when labeled AI (per setting).
 * CSS filter on <img> is unreliable across sites — use an overlay instead.
 */
function applyConcealment(entry: TrackedImage): void {
  const img = entry.element;
  const id = entry.id;

  if (entry.revealed) {
    removeVeil(id);
    img.setAttribute(STATE_ATTR, "clear");
    return;
  }

  if (entry.inFlight || !entry.result) {
    ensureVeil(img, id, "blur");
    img.setAttribute(STATE_ATTR, "pending");
    return;
  }

  if (entry.result.label.kind === "ai") {
    const mode: AiConcealMode = options.aiConceal;
    if (mode === "blur" || mode === "blank") {
      ensureVeil(img, id, mode);
      img.setAttribute(STATE_ATTR, "ai");
    } else {
      removeVeil(id);
      img.setAttribute(STATE_ATTR, "clear");
    }
    return;
  }

  // real / uncertain / error — not over AI threshold → show
  removeVeil(id);
  img.setAttribute(STATE_ATTR, "clear");
}

function ensureBadge(img: HTMLImageElement, id: string): HTMLElement {
  const parent = ensureParentPositioned(img);

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
      const canToggle =
        entry.inFlight ||
        !entry.result ||
        (entry.result.label.kind === "ai" && options.aiConceal !== "none");
      if (!canToggle) return;
      entry.revealed = !entry.revealed;
      if (entry.result) renderResult(entry.element, entry.result);
      else {
        applyConcealment(entry);
        badge.classList.add("truepixel-clickable");
        badge.textContent = entry.revealed ? "hide" : "…";
        badge.title = entry.revealed
          ? "TruePixel: click to hide again"
          : "TruePixel: analyzing… click to peek";
      }
    });
    parent.appendChild(badge);
  }
  positionOverImage(img, badge, 6);
  return badge;
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

  // Floor uncertain % so "? 70%" cannot appear when score is still under a 70% threshold.
  const pct =
    result.label.kind === "uncertain"
      ? Math.floor(result.confidence * 100 + 1e-9)
      : Math.round(result.confidence * 100);
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
      badge.title = `TruePixel: below AI threshold (${pct}% AI confidence). Blur/blank only applies to AI labels.${timingHint}`;
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

async function requestAnalyze(
  img: HTMLImageElement,
  id: string,
): Promise<DetectionResult> {
  const response = (await chrome.runtime.sendMessage({
    kind: "analyze-image",
    requestId: newRequestId(),
    imageId: id,
    src: imageKey(img),
    width: img.naturalWidth || img.width,
    height: img.naturalHeight || img.height,
    // Always cascade (distilled → CF when gated). Realtime-only misses
    // StyleGAN/TPDNE that Community Forensics recovers.
    speedMode: "accurate",
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
  pendingBadge.title = "TruePixel: analyzing… blurred until scored";
  applyConcealment(entry);

  try {
    const result = await requestAnalyze(img, id);
    if (tracked.get(id)?.src !== entry.src) return;
    entry.inFlight = false;
    entry.result = result;
    renderResult(img, result);
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
    if (current) {
      current.inFlight = false;
      applyConcealment(current);
    }
  }
}

/** Same URL can decode new bytes (TPDNE). Rescore on reload after the first. */
function armLoadRescore(img: HTMLImageElement, id: string): void {
  if (img.dataset.truepixelLoadArmed === "1") return;
  img.dataset.truepixelLoadArmed = "1";
  let loads = 0;
  img.addEventListener("load", () => {
    loads += 1;
    // First decode is handled by trackAndMaybeAnalyze → analyze.
    if (loads === 1) return;
    const entry = tracked.get(id);
    if (!entry || entry.inFlight) return;
    entry.src = imageKey(img);
    delete entry.result;
    entry.revealed = false;
    void analyze(img, id);
  });
}

function trackAndMaybeAnalyze(img: HTMLImageElement): void {
  if (!options.autoScan) return;
  if (!isEligible(img)) return;
  const src = imageKey(img);
  let id = img.getAttribute(OVERLAY_ATTR);
  if (!id) {
    id = `tp_${Math.random().toString(16).slice(2)}_${tracked.size}`;
    img.setAttribute(OVERLAY_ATTR, id);
  }
  const existing = tracked.get(id);
  if (existing && existing.src === src && existing.result && !existing.inFlight) {
    renderResult(img, existing.result);
    return;
  }
  if (existing?.inFlight) return;

  const entry: TrackedImage = {
    id,
    element: img,
    src,
    inFlight: false,
    revealed: false,
    ...(existing?.result && existing.src === src
      ? { result: existing.result }
      : {}),
  };
  tracked.set(id, entry);
  armLoadRescore(img, id);
  // Blur immediately on discovery — before analyze round-trip.
  if (!entry.result) {
    entry.inFlight = true;
    applyConcealment(entry);
    entry.inFlight = false;
  }
  void analyze(img, id);
}

function discover(): void {
  if (!options.autoScan) return;
  for (const img of document.querySelectorAll("img")) {
    if (img instanceof HTMLImageElement) trackAndMaybeAnalyze(img);
  }
}

function reapplyAllResults(): void {
  for (const entry of tracked.values()) {
    applyConcealment(entry);
    if (entry.result) renderResult(entry.element, entry.result);
  }
}

function scheduleScan(): void {
  if (scanTimer !== undefined) window.clearTimeout(scanTimer);
  scanTimer = window.setTimeout(() => {
    discover();
  }, 80);
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
        autoScan: Boolean((response as { autoScan?: unknown }).autoScan),
        threshold:
          Number((response as { threshold?: unknown }).threshold) ||
          DEFAULT_OPTIONS.threshold,
        realThreshold:
          Number((response as { realThreshold?: unknown }).realThreshold) ||
          DEFAULT_OPTIONS.realThreshold,
        aiConceal,
      };
    }
  } catch {
    // ignore — use defaults until SW is up
  }
}

function start(): void {
  // Blur ASAP (document_start); options refresh in parallel.
  observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "attributes" && mutation.target instanceof HTMLImageElement) {
        trackAndMaybeAnalyze(mutation.target);
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLImageElement) trackAndMaybeAnalyze(node);
        else if (node instanceof Element) {
          for (const img of node.querySelectorAll("img")) {
            if (img instanceof HTMLImageElement) trackAndMaybeAnalyze(img);
          }
        }
      }
    }
  });
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset"],
  });
  discover();
  window.addEventListener("scroll", scheduleScan, { passive: true });
  window.addEventListener("resize", scheduleScan);

  void refreshOptions().then(() => {
    discover();
    reapplyAllResults();
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
      entry.inFlight = false;
      removeVeil(entry.id);
      entry.element.removeAttribute(STATE_ATTR);
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
