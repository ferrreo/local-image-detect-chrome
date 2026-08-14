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
import { resolveAnalyzeUrl } from "../lib/best-image-url";

/** Skip favicons / 1x1 trackers only — thumbs still get scored. */
const MIN_SIDE = 48;
const OVERLAY_ATTR = "data-neopixel-id";
const STATE_ATTR = "data-neopixel-state";
const BADGE_CLASS = "neopixel-badge";
const VEIL_CLASS = "neopixel-veil";
/**
 * Soft memory budget only — never refuse to score an image. Evict detached /
 * far-offscreen completed entries when the map grows large so infinite feeds
 * don't OOM, but new visible images always get analyzed.
 */
const SOFT_TRACKED_PRUNE_AT = 500;
/** Parallel content→background analyzes (was 2 — starved Google Image grids). */
const MAX_CONTENT_IN_FLIGHT = 8;

type TrackedImage = {
  id: string;
  element: HTMLImageElement;
  src: string;
  inFlight: boolean;
  result?: DetectionResult;
  /** User clicked the badge to temporarily show a hidden image. */
  revealed: boolean;
  /** True after a successful score on a fully decoded frame of `src`. */
  scoredComplete: boolean;
};

const tracked = new Map<string, TrackedImage>();
const observed = new WeakSet<HTMLImageElement>();
let options: ExtensionOptions = { ...DEFAULT_OPTIONS };
let mutationObserver: MutationObserver | undefined;
let intersectionObserver: IntersectionObserver | undefined;
let scanTimer: number | undefined;
let contentInFlight = 0;

/** Higher = sooner. Visible center first, then near-fold, then far below. */
function viewportPriority(img: HTMLImageElement): number {
  if (!img.isConnected) return -1e9;
  const rect = img.getBoundingClientRect();
  const vh = window.innerHeight || 1;
  const vw = window.innerWidth || 1;
  const intersects =
    rect.bottom > 0 &&
    rect.top < vh &&
    rect.right > 0 &&
    rect.left < vw &&
    rect.width > 0 &&
    rect.height > 0;
  if (intersects) {
    const cy = (rect.top + rect.bottom) / 2;
    const dist = Math.abs(cy - vh / 2);
    return 1_000_000 - dist;
  }
  if (rect.top >= vh) {
    // Below the fold — closer to fold wins.
    return 500_000 - Math.min(500_000, rect.top - vh);
  }
  // Above the fold (scrolled past).
  return 250_000 - Math.min(250_000, Math.max(0, -rect.bottom));
}

type AnalyzeJob = {
  img: HTMLImageElement;
  id: string;
  bypassCache: boolean;
  priority: number;
};
const analyzeQueue: AnalyzeJob[] = [];
let analyzePumpScheduled = false;

function enqueueAnalyze(
  img: HTMLImageElement,
  id: string,
  bypassCache = false,
): void {
  const priority = viewportPriority(img);
  const existing = analyzeQueue.find((job) => job.id === id);
  if (existing) {
    existing.priority = priority;
    existing.bypassCache = existing.bypassCache || bypassCache;
    existing.img = img;
  } else {
    analyzeQueue.push({ img, id, bypassCache, priority });
  }
  scheduleAnalyzePump();
}

function scheduleAnalyzePump(): void {
  if (analyzePumpScheduled) return;
  analyzePumpScheduled = true;
  queueMicrotask(() => {
    analyzePumpScheduled = false;
    pumpAnalyzeQueue();
  });
}

/** True when this tile is on-screen (or barely near-fold). */
function isPriorityVisible(priority: number): boolean {
  return priority >= 900_000;
}

/**
 * Offscreen in-flight work should pause when the queue has visible tiles waiting.
 * Frees a slot so scroll targets score first.
 */
function shouldYieldToVisibleQueue(img: HTMLImageElement, id: string): boolean {
  const mine = viewportPriority(img);
  if (isPriorityVisible(mine)) return false;
  for (const job of analyzeQueue) {
    if (job.id === id) continue;
    if (!job.img.isConnected) continue;
    if (isPriorityVisible(viewportPriority(job.img))) return true;
  }
  return false;
}

function pumpAnalyzeQueue(): void {
  // Refresh priorities so a scroll that revealed tiles promotes them.
  for (const job of analyzeQueue) {
    job.priority = viewportPriority(job.img);
  }
  analyzeQueue.sort((a, b) => b.priority - a.priority);

  while (contentInFlight < MAX_CONTENT_IN_FLIGHT && analyzeQueue.length > 0) {
    const job = analyzeQueue.shift();
    if (!job) break;
    if (!job.img.isConnected || !tracked.has(job.id)) continue;
    const entry = tracked.get(job.id);
    if (entry?.inFlight) continue;
    if (entry?.result && entry.src === imageKey(job.img) && !job.bypassCache) {
      if (entry.result.label.kind !== "error") continue;
    }

    // Prefer visible: defer far-offscreen starts while visible work is queued.
    job.priority = viewportPriority(job.img);
    if (
      !isPriorityVisible(job.priority) &&
      analyzeQueue.some((j) => isPriorityVisible(viewportPriority(j.img)))
    ) {
      analyzeQueue.push(job);
      break;
    }

    contentInFlight += 1;
    void analyze(job.img, job.id, job.bypassCache)
      .catch(() => {
        // Keep the content script alive if a single image fails.
      })
      .finally(() => {
        contentInFlight -= 1;
        scheduleAnalyzePump();
      });
  }
}



function imageKey(img: HTMLImageElement, mode: "fast" | "full" = "fast"): string {
  // Fast path: displayed/srcset bytes. Full path: linked original / CDN large
  // — only used to confirm AI or recover Lexica-class misses.
  return resolveAnalyzeUrl(img, mode) || img.currentSrc || img.src || "";
}

/** Decode finished with real pixels — early discover often sees naturalWidth 0. */
function imageDecodeReady(img: HTMLImageElement): boolean {
  return (
    img.complete &&
    img.naturalWidth >= MIN_SIDE &&
    img.naturalHeight >= MIN_SIDE
  );
}

/** Enough to start a fetch — painted size + URL, even if naturalWidth is still 0. */
function canStartAnalyze(img: HTMLImageElement): boolean {
  if (!imageKey(img)) return false;
  if (imageDecodeReady(img)) return true;
  const rect = img.getBoundingClientRect();
  return (
    rect.width >= MIN_SIDE &&
    rect.height >= MIN_SIDE &&
    Boolean(img.currentSrc || img.src)
  );
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
  veil.classList.toggle("neopixel-veil-blur", mode === "blur");
  veil.classList.toggle("neopixel-veil-blank", mode === "blank");
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

  // real / uncertain / error — not AI → show (badges for Real/? only in debug)
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
    created.className = `${BADGE_CLASS} neopixel-pending`;
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
        created.classList.add("neopixel-clickable");
        created.textContent = entry.revealed ? "hide" : "…";
        created.title = entry.revealed
          ? "NeoPixel: click to hide again"
          : "NeoPixel: analyzing… click to peek";
      }
    });
    parent.appendChild(created);
    badge = created;
  }
  positionOverImage(img, badge, 6);
  return badge;
}

function removeBadge(id: string): void {
  document
    .querySelectorAll(`button.${BADGE_CLASS}[${OVERLAY_ATTR}="${id}"]`)
    .forEach((node) => node.remove());
}

function renderResult(img: HTMLImageElement, result: DetectionResult): void {
  const entry = tracked.get(result.imageId);
  if (entry) entry.result = result;

  // Real / uncertain are debug-only — default UI only shows AI (+ confidence).
  // Keep blur while analyze() still has inFlight (e.g. accurate refine).
  if (
    (result.label.kind === "real" || result.label.kind === "uncertain") &&
    !options.debug
  ) {
    removeBadge(result.imageId);
    if (entry) applyConcealment(entry);
    else {
      removeVeil(result.imageId);
      img.setAttribute(STATE_ATTR, "clear");
    }
    return;
  }

  const badge = ensureBadge(img, result.imageId);
  badge.classList.remove(
    "neopixel-pending",
    "neopixel-ai",
    "neopixel-real",
    "neopixel-uncertain",
    "neopixel-error",
    "neopixel-clickable",
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
      badge.classList.add("neopixel-ai");
      const concealed =
        options.aiConceal !== "none" && entry && !entry.revealed;
      badge.textContent = concealed
        ? `AI ${pct}% · show`
        : entry?.revealed && options.aiConceal !== "none"
          ? `AI ${pct}% · hide`
          : `AI ${pct}%`;
      badge.title = concealed
        ? `NeoPixel: likely AI-generated (${pct}%)${timingHint}. Click to reveal.`
        : entry?.revealed && options.aiConceal !== "none"
          ? `NeoPixel: revealed. Click to hide again.${timingHint}`
          : `NeoPixel: likely AI-generated (${pct}% confidence)${timingHint}`;
      if (options.aiConceal !== "none") {
        badge.classList.add("neopixel-clickable");
      }
      break;
    }
    case "real":
      badge.classList.add("neopixel-real");
      badge.textContent = `Real ${pct}%`;
      badge.title = `NeoPixel: likely real photograph (${pct}% confidence)${timingHint}`;
      break;
    case "uncertain":
      badge.classList.add("neopixel-uncertain");
      badge.textContent = `? ${pct}%`;
      badge.title = `NeoPixel: below AI threshold (${pct}% AI confidence). Blur/blank only applies to AI labels.${timingHint}`;
      break;
    case "error":
      badge.classList.add("neopixel-error");
      badge.classList.add("neopixel-clickable");
      badge.textContent = entry?.revealed ? "n/a · hide" : "n/a";
      badge.title = `NeoPixel error: ${result.label.message}`;
      break;
    default: {
      const _exhaustive: never = result.label;
      void _exhaustive;
    }
  }

  if (entry) applyConcealment(entry);
}

async function requestAnalyze(
  img: HTMLImageElement,
  id: string,
  speedMode: "realtime" | "accurate",
  bypassCache = false,
  srcOverride?: string,
): Promise<DetectionResult> {
  const src = srcOverride || imageKey(img);
  const response = (await chrome.runtime.sendMessage({
    kind: "analyze-image",
    requestId: newRequestId(),
    imageId: id,
    src,
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

async function analyzeWithFallback(
  img: HTMLImageElement,
  id: string,
  speedMode: "realtime" | "accurate",
  bypassCache: boolean,
  preferredSrc: string,
): Promise<DetectionResult> {
  const primary = await requestAnalyze(
    img,
    id,
    speedMode,
    bypassCache,
    preferredSrc,
  );
  if (primary.label.kind !== "error") return primary;
  const thumb = img.currentSrc || img.src || "";
  if (!thumb || thumb === preferredSrc) return primary;
  return requestAnalyze(img, id, speedMode, bypassCache, thumb);
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
  let requeueAfter = false;
  // Default UI: blur while calculating, no badge until AI (or debug Real/?).
  if (options.debug) {
    const pendingBadge = ensureBadge(img, id);
    pendingBadge.classList.add("neopixel-pending", "neopixel-clickable");
    pendingBadge.textContent = "…";
    pendingBadge.title = "NeoPixel: analyzing… blurred until scored";
  } else {
    removeBadge(id);
  }
  applyConcealment(entry);

  const yieldToVisible = (): boolean => {
    if (!shouldYieldToVisibleQueue(img, id)) return false;
    requeueAfter = true;
    return true;
  };

  try {
    if (yieldToVisible()) return;

    const fastUrl = imageKey(img, "fast");
    const fullUrl = imageKey(img, "full");
    // Always realtime-first on the cheap asset. Accurate-first on every Google
    // thumb was a major scroll stall; confirm/refine below covers hard cases.
    const fast = await analyzeWithFallback(
      img,
      id,
      "realtime",
      bypassCache,
      fastUrl,
    );
    if (tracked.get(id)?.src !== entry.src) return;

    // Scrolled away mid-flight and visible tiles are waiting — free the slot.
    if (yieldToVisible()) {
      if (fast.label.kind === "ai") {
        entry.result = fast;
        entry.scoredComplete = imageDecodeReady(img);
        renderResult(img, fast);
        requeueAfter = false;
      }
      return;
    }

    entry.result = fast;
    entry.scoredComplete = imageDecodeReady(img);
    renderResult(img, fast);

    const canUpgrade = Boolean(fullUrl && fullUrl !== fastUrl);
    let scoredFull = false;

    // Lexica-class miss recovery: accurate head on a better asset when possible.
    if (needsAccurateRefine(fast)) {
      applyConcealment(entry);
      if (yieldToVisible()) return;

      const refineUrl = canUpgrade ? fullUrl : fastUrl;
      scoredFull = canUpgrade;
      const accurate = await analyzeWithFallback(
        img,
        id,
        "accurate",
        bypassCache,
        refineUrl,
      );
      if (tracked.get(id)?.src !== entry.src) return;
      if (yieldToVisible()) {
        if (accurate.label.kind === "ai") {
          entry.result = accurate;
          entry.scoredComplete = imageDecodeReady(img);
          renderResult(img, accurate);
          requeueAfter = false;
        }
        return;
      }
      entry.result = accurate;
      entry.scoredComplete = imageDecodeReady(img);
      renderResult(img, accurate);
    }

    // Confirm AI on a larger asset when the first pass only saw the thumb.
    // Avoids feed/lightbox disagreement without fetching large for every tile.
    const current = entry.result ?? fast;
    if (current.label.kind === "ai" && canUpgrade && !scoredFull) {
      applyConcealment(entry);
      if (yieldToVisible()) return;
      const confirmed = await analyzeWithFallback(
        img,
        id,
        "accurate",
        bypassCache,
        fullUrl,
      );
      if (tracked.get(id)?.src !== entry.src) return;
      if (confirmed.label.kind !== "error") {
        entry.result = confirmed;
        entry.scoredComplete = imageDecodeReady(img);
        renderResult(img, confirmed);
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    entry.scoredComplete = false;
    requeueAfter = false;
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
      if (requeueAfter && current.element.isConnected) {
        enqueueAnalyze(img, id, bypassCache);
      }
    }
  }
}

/**
 * Rescore when bytes actually arrive. Early discover often marks pending while
 * naturalWidth is still 0; skipping the first `load` left first-fold tiles blank
 * or stuck on placeholder scores forever.
 */
function armLoadRescore(img: HTMLImageElement, id: string): void {
  if (img.dataset.neopixelLoadArmed === "1") return;
  img.dataset.neopixelLoadArmed = "1";
  img.addEventListener("load", () => {
    if (!isEligible(img)) {
      forgetEntry(id);
      return;
    }
    const entry = tracked.get(id);
    if (!entry || entry.inFlight) return;
    const nextSrc = imageKey(img);
    const srcChanged = entry.src !== nextSrc;
    const needsFirstComplete =
      !entry.scoredComplete && imageDecodeReady(img);
    const softFail = entry.result?.label.kind === "error";
    if (!srcChanged && !needsFirstComplete && !softFail) return;
    entry.src = nextSrc;
    entry.scoredComplete = false;
    delete entry.result;
    entry.revealed = false;
    enqueueAnalyze(img, id, true);
  });
  img.addEventListener("error", () => {
    const entry = tracked.get(id);
    if (!entry || entry.result) return;
    // Broken tile — never leave it permanently blurred with no label.
    entry.inFlight = false;
    entry.scoredComplete = false;
    removeVeil(id);
    removeBadge(id);
    img.setAttribute(STATE_ATTR, "clear");
  });
}

/** Don't leave unscored tiles blurred forever if decode never reports ready. */
function armPendingDeadline(img: HTMLImageElement, id: string): void {
  if (img.dataset.neopixelDeadline === "1") return;
  img.dataset.neopixelDeadline = "1";
  window.setTimeout(() => {
    const entry = tracked.get(id);
    if (!entry || entry.result || entry.inFlight) return;
    if (!img.isConnected) {
      forgetEntry(id);
      return;
    }
    if (canStartAnalyze(img) && isEligible(img)) {
      enqueueAnalyze(img, id, true);
      return;
    }
    // Give up: unblur rather than trap the tile.
    removeVeil(id);
    removeBadge(id);
    img.setAttribute(STATE_ATTR, "clear");
    tracked.delete(id);
  }, 2_500);
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

/** Drop detached DOM nodes; soft-evict far-offscreen completed when large. */
function pruneTracked(preferKeep?: HTMLImageElement): void {
  for (const [id, entry] of tracked) {
    if (entry.element === preferKeep) continue;
    if (!entry.element.isConnected && !entry.inFlight) {
      forgetEntry(id);
    }
  }
  if (tracked.size < SOFT_TRACKED_PRUNE_AT) return;
  // Evict completed off-viewport entries first (oldest Map order).
  for (const [id, entry] of tracked) {
    if (tracked.size < SOFT_TRACKED_PRUNE_AT) break;
    if (entry.element === preferKeep || entry.inFlight) continue;
    if (!entry.result) continue;
    const rect = entry.element.getBoundingClientRect();
    const margin = 800;
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
  if (!isEligible(img)) {
    const existingId = img.getAttribute(OVERLAY_ATTR);
    if (existingId) forgetEntry(existingId);
    return;
  }
  pruneTracked(img);
  // Never skip — if still huge after soft prune, drop oldest completed.
  if (tracked.size >= SOFT_TRACKED_PRUNE_AT * 2) {
    for (const [id, entry] of tracked) {
      if (tracked.size < SOFT_TRACKED_PRUNE_AT) break;
      if (entry.element === img || entry.inFlight || !entry.result) continue;
      forgetEntry(id);
    }
  }
  const src = imageKey(img);
  let id = img.getAttribute(OVERLAY_ATTR);
  if (!id) {
    id = `tp_${Math.random().toString(16).slice(2)}_${Date.now().toString(36)}`;
  }
  markPendingVisual(img, id);
  const existing = tracked.get(id);
  if (
    existing &&
    existing.src === src &&
    existing.result &&
    !existing.inFlight &&
    existing.scoredComplete
  ) {
    // Retry transient fetch/offscreen errors on rediscover.
    if (existing.result.label.kind === "error") {
      delete existing.result;
      existing.scoredComplete = false;
    } else {
      // Parent may have been recreated by the site — re-paint overlays.
      renderResult(img, existing.result);
      return;
    }
  }
  if (existing?.inFlight) return;

  const entry: TrackedImage = {
    id,
    element: img,
    src,
    inFlight: false,
    revealed: false,
    scoredComplete: existing?.scoredComplete === true && existing.src === src,
    ...(existing?.result && existing.src === src && existing.scoredComplete
      ? { result: existing.result }
      : {}),
  };
  tracked.set(id, entry);
  armLoadRescore(img, id);
  // Blur immediately while calculating / waiting for decode — badge optional.
  if (!entry.result) {
    ensureVeil(img, id, "blur");
    img.setAttribute(STATE_ATTR, "pending");
    armPendingDeadline(img, id);
  }
  // Prefer a real decode; fall back to painted-size so tiles don't stick blurred.
  if (!canStartAnalyze(img)) return;
  enqueueAnalyze(img, id);
}

function markPendingVisual(img: HTMLImageElement, id: string): void {
  img.setAttribute(OVERLAY_ATTR, id);
  // CSS filter on pending applies before the veil is laid out — kills flash.
  if (!img.getAttribute(STATE_ATTR) || img.getAttribute(STATE_ATTR) === "pending") {
    img.setAttribute(STATE_ATTR, "pending");
  }
}

function observeImage(img: HTMLImageElement): void {
  if (!options.autoScan) return;
  if (observed.has(img)) return;
  observed.add(img);
  // Mark pending as early as possible (document_start discover / mutations).
  if (isEligible(img) && !img.getAttribute(OVERLAY_ATTR)) {
    const id = `tp_${Math.random().toString(16).slice(2)}_${Date.now().toString(36)}`;
    markPendingVisual(img, id);
  } else if (img.getAttribute(OVERLAY_ATTR) && !img.getAttribute(STATE_ATTR)) {
    img.setAttribute(STATE_ATTR, "pending");
  }
  intersectionObserver?.observe(img);
}

function discover(root: ParentNode = document): void {
  if (!options.autoScan) return;
  pruneTracked();
  const consider = (img: HTMLImageElement) => {
    observeImage(img);
    // Score every eligible tile — near-fold-only left blanks on Google grids
    // when IntersectionObserver had already fired with an empty/lazy src.
    trackAndMaybeAnalyze(img);
  };
  if (root instanceof HTMLImageElement) {
    consider(root);
    return;
  }
  for (const img of root.querySelectorAll("img")) {
    if (img instanceof HTMLImageElement) consider(img);
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
    // Re-prioritize queued work for the new viewport, then discover late tiles.
    scheduleAnalyzePump();
    discover();
  }, 32);
}

function onViewportMaybeChanged(): void {
  scheduleAnalyzePump();
  scheduleScan();
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
        debug:
          "debug" in response
            ? Boolean((response as { debug?: unknown }).debug)
            : options.debug,
      };
    }
  } catch {
    // ignore — use defaults until SW is up
  }
}

function start(): void {
  // Near-fold analysis with a large preload margin so grids fill ahead of scroll.
  intersectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        if (!(entry.target instanceof HTMLImageElement)) continue;
        trackAndMaybeAnalyze(entry.target);
      }
    },
    { rootMargin: "1200px 400px", threshold: 0.01 },
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
            entry.scoredComplete = false;
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
  // Capture phase: Google Images / infinite feeds often scroll nested containers,
  // not the window — bubble-only listeners miss those and never re-prioritize.
  window.addEventListener("scroll", onViewportMaybeChanged, {
    passive: true,
    capture: true,
  });
  window.addEventListener("resize", onViewportMaybeChanged, { passive: true });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") onViewportMaybeChanged();
  });
  // Infinite feeds: rediscover often so previously skipped / late-src tiles retry.
  window.setInterval(() => {
    pruneTracked();
    discover();
    scheduleAnalyzePump();
  }, 1000);

  void refreshOptions().then(() => {
    discover();
    reapplyAllResults();
  });
}

chrome.runtime.onMessage.addListener((message: unknown) => {
  if (typeof message !== "object" || message === null || !("kind" in message)) {
    return;
  }
  if (message.kind === "neopixel-rescan") {
    for (const entry of tracked.values()) {
      delete entry.result;
      entry.revealed = false;
      entry.inFlight = false;
      entry.scoredComplete = false;
      removeVeil(entry.id);
      entry.element.removeAttribute(STATE_ATTR);
      observed.delete(entry.element);
    }
    tracked.clear();
    scheduleScan();
    return;
  }
  if (message.kind === "neopixel-options") {
    void refreshOptions().then(() => {
      reapplyAllResults();
      scheduleScan();
    });
  }
});

start();
