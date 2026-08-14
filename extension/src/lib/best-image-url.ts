/**
 * Prefer the largest available bitmap URL for an <img>, so a 160px CSS thumb
 * that points at a 1200w srcset candidate is scored on the full asset — not the
 * upscaled mush that looks "AI".
 */

export type SrcCandidate = { url: string; score: number };

/** Parse a srcset attribute into URL + relative size score (w or x descriptor). */
export function parseSrcset(srcset: string): SrcCandidate[] {
  const out: SrcCandidate[] = [];
  for (const part of srcset.split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    // URL may contain commas in data: URIs — rare for srcset; split on last space.
    const m = trimmed.match(/^(.+?)\s+(\d+(?:\.\d+)?)([wx])$/i);
    if (m) {
      const url = m[1]!.trim();
      const value = Number(m[2]);
      const unit = m[3]!.toLowerCase();
      if (!url || !Number.isFinite(value) || value <= 0) continue;
      // Prefer width descriptors. Density (x) ≈ CSS px * density with a
      // ~400 CSS-px assumed layout width so 2x (~800) loses to 1600w.
      out.push({ url, score: unit === "w" ? value : value * 400 });
      continue;
    }
    // Bare URL — unknown size; weak score so real descriptors win.
    out.push({ url: trimmed, score: 1 });
  }
  return out;
}

export function pickBestCandidate(candidates: readonly SrcCandidate[]): string | undefined {
  if (candidates.length === 0) return undefined;
  let best = candidates[0]!;
  for (let i = 1; i < candidates.length; i += 1) {
    const c = candidates[i]!;
    if (c.score > best.score) best = c;
  }
  return best.url;
}

function pushAttr(
  candidates: SrcCandidate[],
  raw: string | null | undefined,
  score: number,
): void {
  const url = raw?.trim();
  if (!url || url.startsWith("data:image/svg")) return;
  candidates.push({ url, score });
}

function decodeGoogleEscapes(raw: string): string {
  let v = raw
    .replace(/\\u003d/gi, "=")
    .replace(/\\u0026/gi, "&")
    .replace(/\\\//g, "/");
  try {
    v = decodeURIComponent(v);
  } catch {
    // keep partially decoded
  }
  return v;
}

/** Pull imgurl=… from a query string or percent-encoded blob. */
export function extractImgurlParam(raw: string): string | undefined {
  const patterns = [
    /[?&#]imgurl=([^&]+)/i,
    /[?&#]imgrefurl=([^&]+)/i,
    /[?&#]mediaurl=([^&]+)/i,
    /imgurl%3D([^&%]+)/i,
    /"ou"\s*:\s*"(https?:\\\/\\\/[^"]+)"/i,
    /"ou"\s*:\s*"(https?:\/\/[^"]+)"/i,
  ];
  for (const re of patterns) {
    const m = raw.match(re);
    if (!m?.[1]) continue;
    const v = decodeGoogleEscapes(m[1]);
    if (/^https?:\/\//i.test(v)) return v;
  }
  return undefined;
}

/**
 * Google Images embeds `data-id` → full URL pairs inside AF_initDataCallback
 * script blobs. Cache per document so we only scan scripts once.
 */
const googleDataIdMaps = new WeakMap<Document, Map<string, string>>();

/** data-id, [thumb…], [fullUrl, w, h] */
const GOOGLE_DATA_ID_FULL_RE =
  /"([A-Za-z0-9_-]{6,32})",\s*\[\s*"https:\/\/[^"]*gstatic\.com\/images[^"]*"\s*,\s*\d+\s*,\s*\d+\s*\]\s*,\s*\[\s*"(https?:\/\/[^"]+)"/g;

/** thumb URL → following non-gstatic https URL */
const GOOGLE_TBN_FULL_RE =
  /"(https:\/\/encrypted-tbn[^"]+)"\s*,\s*\d+\s*,\s*\d+\s*\]\s*,\s*\[\s*"(https?:\/\/(?!encrypted-tbn)[^"]+)"/g;

export function parseGoogleImagesFullUrlMap(doc: Document): Map<string, string> {
  const cached = googleDataIdMaps.get(doc);
  if (cached && cached.size > 0) return cached;
  const map = new Map<string, string>();
  const scripts = doc.querySelectorAll("script");
  for (const script of scripts) {
    const text = script.textContent ?? "";
    if (!text.includes("gstatic.com/images")) continue;
    GOOGLE_DATA_ID_FULL_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = GOOGLE_DATA_ID_FULL_RE.exec(text)) !== null) {
      const id = m[1];
      const full = decodeGoogleEscapes(m[2] ?? "");
      if (!id || !/^https?:\/\//i.test(full)) continue;
      if (/gstatic\.com\/images/i.test(full)) continue;
      map.set(id, full);
    }
    GOOGLE_TBN_FULL_RE.lastIndex = 0;
    while ((m = GOOGLE_TBN_FULL_RE.exec(text)) !== null) {
      const thumb = m[1] ?? "";
      const full = decodeGoogleEscapes(m[2] ?? "");
      if (!thumb || !/^https?:\/\//i.test(full)) continue;
      if (/gstatic\.com\/images/i.test(full)) continue;
      map.set(thumb, full);
      const token = thumb.match(/tbn:[^&\s"]+/)?.[0];
      if (token) map.set(token, full);
    }
  }
  if (map.size > 0) googleDataIdMaps.set(doc, map);
  return map;
}

/** Resolve full asset URL via Google Images data-id / thumb token map. */
export function extractGoogleImagesFullUrl(
  img: HTMLImageElement,
): string | undefined {
  const doc = img.ownerDocument;
  if (!doc) return undefined;
  const map = parseGoogleImagesFullUrlMap(doc);
  if (map.size === 0) return undefined;

  let node: HTMLElement | null = img;
  for (let depth = 0; depth < 10 && node; depth += 1) {
    for (const attr of ["data-id", "data-docid", "data-ci", "data-lpage"] as const) {
      const id = node.getAttribute(attr);
      if (id && map.has(id)) return map.get(id);
    }
    node = node.parentElement;
  }

  const thumb = img.currentSrc || img.src || "";
  if (thumb && map.has(thumb)) return map.get(thumb);
  const token = thumb.match(/tbn:[^&\s"]+/)?.[0];
  if (token && map.has(token)) return map.get(token);
  return undefined;
}

/** Google Images / Bing / similar: full asset URL buried in anchors or tile JSON. */
export function extractLinkedFullImageUrl(
  img: HTMLImageElement,
): string | undefined {
  const fromGoogle = extractGoogleImagesFullUrl(img);
  if (fromGoogle) return fromGoogle;

  const blobs: string[] = [];

  const consider = (raw: string | null | undefined) => {
    if (!raw) return;
    blobs.push(raw);
  };

  if (img.closest) {
    const closest = img.closest("a");
    if (closest instanceof HTMLAnchorElement) {
      consider(closest.getAttribute("href"));
      consider(closest.href);
    }
    const roleLink = img.closest("[href], [role='link']");
    if (roleLink instanceof HTMLElement) {
      consider(roleLink.getAttribute("href"));
    }
  }

  // Walk ancestors — Google often parks imgurl a few levels up.
  let node: HTMLElement | null = img.parentElement;
  for (let depth = 0; depth < 10 && node; depth += 1) {
    if (node instanceof HTMLAnchorElement) {
      consider(node.getAttribute("href"));
      consider(node.href);
    }
    for (const a of node.querySelectorAll("a[href]")) {
      if (a instanceof HTMLAnchorElement) {
        consider(a.getAttribute("href"));
        consider(a.href);
      }
    }
    for (const attr of [
      "data-lpage",
      "data-url",
      "data-ou",
      "data-iurl",
      "data-src",
      "data-id",
      "href",
    ] as const) {
      consider(node.getAttribute(attr));
    }
    // Inline JSON sometimes on the tile root.
    const id = node.id || "";
    if (id.length > 20) consider(id);
    node = node.parentElement;
  }

  for (const blob of blobs) {
    const hit = extractImgurlParam(blob);
    if (hit) return hit;
    try {
      const u = new URL(blob, img.baseURI || undefined);
      for (const key of ["imgurl", "imgrefurl", "mediaurl", "ou"] as const) {
        const v = u.searchParams.get(key);
        if (v && /^https?:\/\//i.test(v)) return v;
      }
    } catch {
      // ignore
    }
  }
  return undefined;
}

/** True when the URL is a search-engine thumbnail CDN (prefer full asset). */
export function isSearchThumbCdn(url: string): boolean {
  return /encrypted-tbn|tse\d+\.mm\.bing|th\.bing\.com|gstatic\.com\/images/i.test(
    url,
  );
}

/**
 * Resolve the URL we should fetch for inference.
 * Order: explicit full-size data-* → linked full image → largest srcset → currentSrc.
 */
export function resolveAnalyzeUrl(img: HTMLImageElement): string {
  const candidates: SrcCandidate[] = [];

  // Explicit full-resolution hints used by many galleries / CDNs.
  for (const attr of [
    "data-full-url",
    "data-original",
    "data-orig-src",
    "data-src-large",
    "data-large-src",
    "data-hi-res",
    "data-iurl",
    "data-ou",
  ] as const) {
    pushAttr(candidates, img.getAttribute(attr), 10_000_000);
  }

  const linked = extractLinkedFullImageUrl(img);
  if (linked) pushAttr(candidates, linked, 9_000_000);

  const picture = img.closest("picture");
  if (picture) {
    for (const source of picture.querySelectorAll("source[srcset]")) {
      const srcset = source.getAttribute("srcset");
      if (srcset) candidates.push(...parseSrcset(srcset));
    }
  }

  if (img.srcset) candidates.push(...parseSrcset(img.srcset));

  const current = img.currentSrc || img.src || "";
  // Deprioritize Google/Bing thumbnail CDNs so a linked full URL always wins.
  const thumbish = isSearchThumbCdn(current);
  const naturalScore = Math.max(
    1,
    (img.naturalWidth || 0) * (img.naturalHeight || 0),
  );
  pushAttr(candidates, current, thumbish ? Math.min(naturalScore, 50) : naturalScore);
  pushAttr(
    candidates,
    img.getAttribute("src"),
    thumbish ? 25 : Math.max(1, naturalScore * 0.5),
  );

  const best = pickBestCandidate(candidates);
  if (!best) return current;

  try {
    return new URL(best, img.baseURI || document.baseURI).href;
  } catch {
    return best;
  }
}

/**
 * How much to raise the AI label floor when the decoded asset is smaller than
 * the model input (upscaling invents smoothness detectors read as synthetic).
 */
export function aiThresholdBumpForSourceSide(sourceMinSide: number): number {
  if (sourceMinSide >= 384) return 0;
  if (sourceMinSide >= 256) return 0.04;
  if (sourceMinSide >= 160) return 0.1;
  if (sourceMinSide >= 96) return 0.16;
  return 0.22;
}
