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
/**
 * Soft budget for in-memory tracked entries. Lexica-style infinite feeds
 * recycle DOM nodes — we prune disconnected / offscreen entries instead of
 * hard-stopping (the old hard cap made blur+badges vanish after ~40 images).
 */
const MAX_TRACKED = 80;
const MAX_CONTENT_IN_FLIGHT = 2;

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
const observed = new WeakSet<HTMLImageElement>();
let options: ExtensionOptions = { ...DEFAULT_OPTIONS };
let mutationObserver: MutationObserver | undefined;
let intersectionObserver: IntersectionObserver | undefined;
let scanTimer: number | undefined;
let contentInFlight = 0;
const contentWaiters: Array<() => void> = [];

async function withContentSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (contentInFlight >= MAX_CONTENT_IN_FLIGHT) {
    await new Promise<void>((resolve) => {
      contentWaiters.push(resolve);
    });
  }
  contentInFlight += 1;
  try {
    return await fn();
  } finally {
    contentInFlight -= 1;
    contentWaiters.shift()?.();
  }
}

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

  let badge = parent.querySelector<HTMLButtonElement>(
    `button.${BADGE_CLASS}[${OVERLAY_ATTR}="${id}"]`,
  );
  if (!badge) {
    const created = document.createElement("button");
    created.type = "button";
    created.className = `${BADGE_CLASS} truepixel-pending`;
    created.setAttribute(OVERLAY_ATTR, id);
    created.textContent = "…";
    created.addEventListener("click", (event) => {
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
        created.classList.add("truepixel-clickable");
        created.textContent = entry.revealed ? "hide" : "…";
        created.title = entry.revealed
          ? "TruePixel: click to hide again"
          : "TruePixel: analyzing… click to peek";
      }
    });
    parent.appendChild(created);
    badge = created;
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
  speedMode: "realtime" | "accurate",
  bypassCache = false,
): Promise<DetectionResult> {
  const response = (await chrome.runtime.sendMessage({
    kind: "analyze-image",
    requestId: newRequestId(),
    imageId: id,
    src: imageKey(img),
    width: img.naturalWidth || img.width,
    height: img.naturalHeight || img.height,
    speedMode,
    ...(bypassCache ? { bypassCache: true } : {}),
  })) as AnalyzeImageResponse;
  if (response.kind !== "analyze-image-result") {
    throw new Error("analyze-image failed");
  }
  return response.result;
}

/**
 * Realtime distilled often scores Lexica-class AI near zero (labeled real).
 * Refine any non-AI first paint so the Proofmark accurate head can recover.
 */
function needsAccurateRefine(result: DetectionResult): boolean {
  if (result.label.kind === "error") return false;
  if (result.timing?.ranForensics) return false;
  return result.confidence < options.threshold;
}

async function analyze(
  img: HTMLImageElement,
  id: string,
  bypassCache = false,
): Promise<void> {
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
    await withContentSlot(async () => {
      // Fast path first (distilled-only), like Proofmark's single-head latency.
      const fast = await requestAnalyze(img, id, "realtime", bypassCache);
      if (tracked.get(id)?.src !== entry.src) return;
      entry.result = fast;
      renderResult(img, fast);

      if (!needsAccurateRefine(fast)) return;
      const accurate = await requestAnalyze(img, id, "accurate", bypassCache);
      if (tracked.get(id)?.src !== entry.src) return;
      entry.result = accurate;
      renderResult(img, accurate);
    });
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
    void analyze(img, id, true);
  });
}

function forgetEntry(id: string): void {
  const entry = tracked.get(id);
  if (!entry) return;
  removeVeil(id);
  document
    .querySelectorAll(`.${BADGE_CLASS}[${OVERLAY_ATTR}="${id}"]`)
    .forEach((node) => node.remove());
  entry.element.removeAttribute(OVERLAY_ATTR);
  entry.element.removeAttribute(STATE_ATTR);
  intersectionObserver?.unobserve(entry.element);
  observed.delete(entry.element);
  tracked.delete(id);
}

/** Drop detached DOM nodes (virtualized feeds) and free budget for new ones. */
function pruneTracked(preferKeep?: HTMLImageElement): void {
  for (const [id, entry] of tracked) {
    if (entry.element === preferKeep) continue;
    if (!entry.element.isConnected && !entry.inFlight) {
      forgetEntry(id);
    }
  }
  if (tracked.size < MAX_TRACKED) return;
  // Evict completed off-viewport entries first (oldest Map order).
  for (const [id, entry] of tracked) {
    if (tracked.size < MAX_TRACKED) break;
    if (entry.element === preferKeep || entry.inFlight) continue;
    const rect = entry.element.getBoundingClientRect();
    const margin = 400;
    const onscreen =
      rect.bottom >= -margin &&
      rect.top <= window.innerHeight + margin &&
      rect.right >= -margin &&
      rect.left <= window.innerWidth + margin;
    if (!onscreen) forgetEntry(id);
  }
}

function trackAndMaybeAnalyze(img: HTMLImageElement): void {
  if (!options.autoScan) return;
  if (!isEligible(img)) return;
  pruneTracked(img);
  if (tracked.size >= MAX_TRACKED && !img.getAttribute(OVERLAY_ATTR)) {
    // Still over budget after prune — skip this one; next scroll will retry.
    return;
  }
  const src = imageKey(img);
  let id = img.getAttribute(OVERLAY_ATTR);
  if (!id) {
    id = `tp_${Math.random().toString(16).slice(2)}_${Date.now().toString(36)}`;
    img.setAttribute(OVERLAY_ATTR, id);
  }
  const existing = tracked.get(id);
  if (existing && existing.src === src && existing.result && !existing.inFlight) {
    // Parent may have been recreated by the site — re-paint overlays.
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
  // Blur immediately when entering the analyze path — before round-trip.
  if (!entry.result) {
    entry.inFlight = true;
    applyConcealment(entry);
    entry.inFlight = false;
  }
  void analyze(img, id).catch(() => {
    // Keep the content script alive if a single image fails.
  });
}

function observeImage(img: HTMLImageElement): void {
  if (!options.autoScan) return;
  if (observed.has(img)) return;
  observed.add(img);
  intersectionObserver?.observe(img);
}

function discover(root: ParentNode = document): void {
  if (!options.autoScan) return;
  pruneTracked();
  if (root instanceof HTMLImageElement) {
    observeImage(root);
    return;
  }
  for (const img of root.querySelectorAll("img")) {
    if (img instanceof HTMLImageElement) observeImage(img);
  }
}

function reapplyAllResults(): void {
  pruneTracked();
  for (const entry of tracked.values()) {
    if (!entry.element.isConnected) continue;
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
  // Viewport-gated analysis (Proofmark): only spend inference near the fold.
  intersectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        if (!(entry.target instanceof HTMLImageElement)) continue;
        trackAndMaybeAnalyze(entry.target);
      }
    },
    { rootMargin: "180px 0px", threshold: 0.01 },
  );

  mutationObserver = new MutationObserver((mutations) => {
    let removed = false;
    for (const mutation of mutations) {
      if (
        mutation.type === "attributes" &&
        mutation.target instanceof HTMLImageElement
      ) {
        const img = mutation.target;
        observed.delete(img);
        // Src change on a recycled tile — drop stale result and re-analyze.
        // IntersectionObserver will not re-fire for an already-visible tile.
        const id = img.getAttribute(OVERLAY_ATTR);
        if (id) {
          const entry = tracked.get(id);
          if (entry && entry.src !== imageKey(img)) {
            delete entry.result;
            entry.src = imageKey(img);
            entry.revealed = false;
            entry.inFlight = false;
            removeVeil(id);
          }
        }
        observeImage(img);
        trackAndMaybeAnalyze(img);
        continue;
      }
      for (const node of mutation.removedNodes) {
        if (node instanceof HTMLImageElement) {
          const id = node.getAttribute(OVERLAY_ATTR);
          if (id) forgetEntry(id);
          else observed.delete(node);
          removed = true;
        } else if (node instanceof Element) {
          for (const img of node.querySelectorAll("img")) {
            if (!(img instanceof HTMLImageElement)) continue;
            const id = img.getAttribute(OVERLAY_ATTR);
            if (id) forgetEntry(id);
            else observed.delete(img);
            removed = true;
          }
        }
      }
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLImageElement) observeImage(node);
        else if (node instanceof Element) discover(node);
      }
    }
    if (removed) pruneTracked();
  });
  mutationObserver.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset"],
  });
  discover();
  window.addEventListener("scroll", scheduleScan, { passive: true });
  window.addEventListener("resize", scheduleScan);
  // Infinite feeds keep mutating; periodic prune + rediscover keeps budget open.
  window.setInterval(() => {
    pruneTracked();
    discover();
  }, 2000);

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
      observed.delete(entry.element);
    }
    tracked.clear();
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
