/**
 * @vitest-environment jsdom
 */
import { describe, expect, it } from "vitest";
import {
  extractGoogleImagesFullUrl,
  extractImgurlParam,
  extractLinkedFullImageUrl,
  parseGoogleImagesFullUrlMap,
  resolveAnalyzeUrl,
} from "../../extension/src/lib/best-image-url";
import { looksLikeSyntheticCgi } from "../../extension/src/lib/spectral";
import { fuseDetection } from "../../extension/src/lib/fusion";
import {
  AI_LABEL_THRESHOLD,
  asAiConfidence,
} from "../../extension/src/shared/types";

describe("Google Images URL resolution (image fetch, not page text)", () => {
  it("extracts imgurl from encoded hrefs", () => {
    const url = extractImgurlParam(
      "/imgres?imgurl=https%3A%2F%2Fcdn.example%2Ffull.jpg&imgrefurl=https%3A%2F%2Fexample.com",
    );
    expect(url).toBe("https://cdn.example/full.jpg");
  });

  it("prefers linked full image over encrypted-tbn thumb", () => {
    const link = document.createElement("a");
    link.href =
      "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example%2Fhires.png&imgrefurl=https%3A%2F%2Fexample.com";
    const img = document.createElement("img");
    img.src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:demo";
    Object.defineProperty(img, "naturalWidth", { value: 180 });
    Object.defineProperty(img, "naturalHeight", { value: 120 });
    link.append(img);
    document.body.append(link);
    expect(extractLinkedFullImageUrl(img)).toBe("https://cdn.example/hires.png");
    expect(resolveAnalyzeUrl(img)).toBe("https://cdn.example/hires.png");
    link.remove();
  });

  it("resolves full URL from AF_initDataCallback data-id map", () => {
    const script = document.createElement("script");
    script.type = "application/json";
    script.textContent =
      '[["-H96xjSoW5DsgM",["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcDemo",155,324],["https://cdn.example/ai-robot.jpg",1200,800]]]';
    document.head.append(script);
    const tile = document.createElement("div");
    tile.setAttribute("data-id", "-H96xjSoW5DsgM");
    const img = document.createElement("img");
    img.src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcDemo";
    tile.append(img);
    document.body.append(tile);
    const map = parseGoogleImagesFullUrlMap(document);
    expect(map.get("-H96xjSoW5DsgM")).toBe("https://cdn.example/ai-robot.jpg");
    expect(extractGoogleImagesFullUrl(img)).toBe(
      "https://cdn.example/ai-robot.jpg",
    );
    tile.remove();
    script.remove();
  });
});

describe("image-only CGI fusion", () => {
  function tier(
    id: "provenance" | "spectral" | "visual",
    score: number,
  ) {
    return {
      tier: id,
      aiScore: asAiConfidence(score),
      weight: 1,
      detail: id,
    };
  }

  const cgiFeats = {
    highFreqEnergyRatio: 0.32,
    laplacianVariance: 900,
    chromaFlatness: 0.62,
    axisAlignedEdgeRatio: 0.35,
    quantizedColorCount: 120,
    topColorShare: 0.4,
    blockiness: 0.12,
  };

  it("flags neon/CGI spectral shapes as synthetic", () => {
    expect(looksLikeSyntheticCgi(cgiFeats)).toBe(true);
  });

  it("boosts soft visual scores on CGI spectral evidence", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.55),
      visual: tier("visual", 0.32),
      spectralFeatures: cgiFeats,
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/cgi-spectral-boost|cgi-floor|neon-ai/);
  });

  it("does not stamp Real on non-photographic low scores", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.5),
      visual: tier("visual", 0.28),
      spectralFeatures: cgiFeats,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("promotes small CGI thumbs past small-source-hold", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.4),
      visual: tier("visual", 0.35),
      spectralFeatures: cgiFeats,
      sourceMinSide: 140,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/small-cgi-ai/);
  });

  it("promotes small soft non-photo AI thumbs", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.45),
      visual: tier("visual", 0.38),
      spectralFeatures: {
        highFreqEnergyRatio: 0.36,
        laplacianVariance: 700,
        chromaFlatness: 0.6,
        axisAlignedEdgeRatio: 0.32,
        quantizedColorCount: 90,
        topColorShare: 0.45,
        blockiness: 0.18,
      },
      sourceMinSide: 160,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/small-nonphoto-ai|small-cgi-ai/);
  });

  it("still promotes Lexica via accurate head", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.12),
      visualSecondary: tier("visual", 0.78),
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
  });
});
