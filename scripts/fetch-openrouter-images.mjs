#!/usr/bin/env node
/**
 * Incrementally fetch one sample image from each OpenRouter image-generation model.
 * Already-seen model IDs in the registry are skipped on later runs.
 *
 * Usage:
 *   OPENROUTER_API_KEY=... node scripts/fetch-openrouter-images.mjs
 *   npm run fetch:openrouter
 *
 * Env:
 *   OPENROUTER_API_KEY   required (or loaded from .env)
 *   OPENROUTER_FORCE=1   regenerate even if the model is already in the registry
 *   OPENROUTER_LIMIT=N   only process the first N unseen models (smoke tests)
 */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  readdirSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outRoot = path.join(root, "benchmark/openrouter");
const aiRoot = path.join(outRoot, "ai");
const registryPath = path.join(outRoot, "registry.json");

const PROMPTS = [
  {
    id: "street_cafe",
    text: "Photorealistic handheld photo of a busy street cafe terrace at golden hour, natural skin tones, slight motion blur, Canon EOS R5, 35mm lens, documentary style",
  },
  {
    id: "kitchen_window",
    text: "Candid photograph of a person chopping vegetables by a kitchen window, soft daylight, realistic fabric texture, no text, no watermark",
  },
  {
    id: "park_dog",
    text: "Natural outdoor photograph of a mixed-breed dog running through autumn park leaves, shallow depth of field, authentic noise, unposed",
  },
];

function loadDotEnv() {
  const envPath = path.join(root, ".env");
  if (!existsSync(envPath)) return;
  for (const line of readFileSync(envPath, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim();
    if (!(key in process.env)) process.env[key] = value;
  }
}

function slugifyModelId(modelId) {
  return modelId.replace(/[^a-zA-Z0-9._-]+/g, "__");
}

function emptyRegistry() {
  return {
    version: 1,
    updatedAt: null,
    models: {},
  };
}

function loadRegistry() {
  if (!existsSync(registryPath)) return emptyRegistry();
  return JSON.parse(readFileSync(registryPath, "utf8"));
}

function saveRegistry(registry) {
  registry.updatedAt = new Date().toISOString();
  mkdirSync(outRoot, { recursive: true });
  writeFileSync(registryPath, JSON.stringify(registry, null, 2) + "\n");
}

function guessExtFromBytes(buf) {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    return "jpg";
  }
  if (
    buf.length >= 8 &&
    buf[0] === 0x89 &&
    buf[1] === 0x50 &&
    buf[2] === 0x4e &&
    buf[3] === 0x47
  ) {
    return "png";
  }
  if (
    buf.length >= 12 &&
    buf[0] === 0x52 &&
    buf[1] === 0x49 &&
    buf[2] === 0x46 &&
    buf[3] === 0x46
  ) {
    return "webp";
  }
  const head = buf.subarray(0, Math.min(64, buf.length)).toString("utf8");
  if (head.includes("<svg") || head.includes("<?xml")) return "svg";
  return "bin";
}

function decodeImagePayload(item) {
  if (typeof item?.b64_json === "string" && item.b64_json.length > 0) {
    const raw = item.b64_json.includes(",")
      ? item.b64_json.split(",").pop()
      : item.b64_json;
    return Buffer.from(raw, "base64");
  }
  if (typeof item?.url === "string" && item.url.startsWith("data:")) {
    const raw = item.url.split(",").pop();
    return Buffer.from(raw, "base64");
  }
  return null;
}

async function fetchImageUrl(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`Image URL fetch failed: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function listImageModels(apiKey) {
  const res = await fetch("https://openrouter.ai/api/v1/images/models", {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!res.ok) {
    throw new Error(`List models failed: HTTP ${res.status} ${await res.text()}`);
  }
  const json = await res.json();
  const models = json.data ?? [];
  if (!Array.isArray(models) || models.length === 0) {
    throw new Error("No image models returned by OpenRouter");
  }
  return models;
}

function isVectorModel(model) {
  const id = String(model.id ?? "").toLowerCase();
  const name = String(model.name ?? "").toLowerCase();
  return id.includes("vector") || name.includes("vector");
}

function buildRequestBody(model, prompt) {
  const params = model.supported_parameters ?? {};
  const body = {
    model: model.id,
    prompt: prompt.text,
    n: 1,
  };

  if (params.aspect_ratio) body.aspect_ratio = "1:1";
  if (params.resolution) {
    const values = params.resolution.values ?? [];
    if (values.includes("1K")) body.resolution = "1K";
    else if (values.includes("512")) body.resolution = "512";
    else if (values.includes("2K")) body.resolution = "2K";
    else if (values[0]) body.resolution = values[0];
  }
  if (params.quality) {
    const values = params.quality.values ?? ["low", "medium", "high", "auto"];
    if (values.includes("low")) body.quality = "low";
    else if (values.includes("auto")) body.quality = "auto";
    else if (values.includes("medium")) body.quality = "medium";
  }
  if (params.output_format) {
    const values = params.output_format.values ?? [];
    if (values.includes("png")) body.output_format = "png";
    else if (values.includes("jpeg")) body.output_format = "jpeg";
    else if (values.includes("webp")) body.output_format = "webp";
  }
  if (params.seed) body.seed = 42;

  return body;
}

async function postGenerate(apiKey, body) {
  const res = await fetch("https://openrouter.ai/api/v1/images", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/ferrreo/local-image-detect-chrome",
      "X-Title": "TruePixel benchmark fetcher",
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`Non-JSON response HTTP ${res.status}: ${text.slice(0, 300)}`);
  }

  if (!res.ok) {
    const message = json?.error?.message ?? text.slice(0, 300);
    const err = new Error(`Generate failed HTTP ${res.status}: ${message}`);
    err.status = res.status;
    err.payload = json;
    err.messageText = message;
    throw err;
  }

  return json;
}

async function generateOne(apiKey, model, prompt) {
  let body = buildRequestBody(model, prompt);
  let json;
  try {
    json = await postGenerate(apiKey, body);
  } catch (error) {
    const msg = String(error.messageText ?? error.message ?? "");
    // Some models reject 1K; escalate to 2K once.
    if (
      error.status === 400 &&
      /larger resolution|at least .* output pixels|Use a larger resolution/i.test(
        msg,
      ) &&
      body.resolution &&
      body.resolution !== "2K"
    ) {
      body = { ...body, resolution: "2K" };
      console.warn(`  retry ${model.id} with resolution=2K`);
      json = await postGenerate(apiKey, body);
    } else {
      throw error;
    }
  }

  const item = json.data?.[0];
  if (!item) throw new Error("Empty image data in response");

  let bytes = decodeImagePayload(item);
  if (!bytes && typeof item.url === "string") {
    bytes = await fetchImageUrl(item.url);
  }
  if (!bytes) throw new Error("Could not decode image payload");

  return {
    bytes,
    usage: json.usage ?? null,
    request: body,
  };
}

async function sleep(ms) {
  await new Promise((r) => setTimeout(r, ms));
}

function writeIndex(registry) {
  const images = [];
  for (const [modelId, entry] of Object.entries(registry.models)) {
    if (entry.status !== "ok") continue;
    for (const file of entry.files ?? []) {
      images.push({
        file: path.relative(outRoot, path.join(aiRoot, slugifyModelId(modelId), file.name)),
        label: "ai",
        model: modelId,
        modelName: entry.name,
        promptId: file.promptId,
        sha256: file.sha256,
      });
    }
  }
  writeFileSync(
    path.join(outRoot, "index.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        count: images.length,
        images,
      },
      null,
      2,
    ) + "\n",
  );
}

async function main() {
  loadDotEnv();
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is required (set env or .env)");
  }

  const force = process.env.OPENROUTER_FORCE === "1";
  const limit = process.env.OPENROUTER_LIMIT
    ? Number(process.env.OPENROUTER_LIMIT)
    : Infinity;
  const concurrency = Math.max(
    1,
    Number(process.env.OPENROUTER_CONCURRENCY ?? 3),
  );

  mkdirSync(aiRoot, { recursive: true });
  const registry = loadRegistry();
  const models = await listImageModels(apiKey);
  console.log(`OpenRouter image models: ${models.length}`);

  const unseen = models.filter((m) => {
    if (force) return true;
    const prev = registry.models[m.id];
    // Retry previous failures; skip successful / intentionally skipped models.
    if (prev?.status === "ok") return false;
    if (prev?.status === "skipped") return false;
    return true;
  });

  const queue = unseen.slice(0, Number.isFinite(limit) ? limit : undefined);
  console.log(
    `Unseen / retryable models: ${unseen.length}; processing ${queue.length} with concurrency=${concurrency}`,
  );

  let succeeded = 0;
  let failed = 0;
  let skipped = 0;
  let cursor = 0;
  let creditsExhausted = false;

  async function processModel(model, index) {
    if (creditsExhausted) return;

    if (isVectorModel(model)) {
      registry.models[model.id] = {
        status: "skipped",
        reason: "vector-output-not-useful-for-photo-detector-eval",
        name: model.name,
        seenAt: new Date().toISOString(),
        files: [],
      };
      skipped += 1;
      console.log(`[skip] ${model.id} (vector)`);
      saveRegistry(registry);
      writeIndex(registry);
      return;
    }

    const modelDir = path.join(aiRoot, slugifyModelId(model.id));
    mkdirSync(modelDir, { recursive: true });
    const files = [];
    let modelError = null;
    const prompt = PROMPTS[(index + 1) % PROMPTS.length];
    console.log(`[gen ] ${model.id} :: ${prompt.id}`);

    let attempt = 0;
    while (attempt < 3 && !creditsExhausted) {
      attempt += 1;
      try {
        const result = await generateOne(apiKey, model, prompt);
        const ext = guessExtFromBytes(result.bytes);
        const name = `${prompt.id}.${ext}`;
        writeFileSync(path.join(modelDir, name), result.bytes);
        const sha256 = createHash("sha256").update(result.bytes).digest("hex");
        files.push({
          name,
          promptId: prompt.id,
          prompt: prompt.text,
          sha256,
          bytes: result.bytes.byteLength,
          request: result.request,
          usage: result.usage,
        });
        console.log(
          `  ok  ${model.id} → ${name} (${result.bytes.byteLength} bytes)` +
            (result.usage?.cost != null ? ` cost=${result.usage.cost}` : ""),
        );
        modelError = null;
        break;
      } catch (error) {
        modelError = error;
        const status = error.status ?? 0;
        console.warn(`  fail ${model.id} attempt ${attempt}: ${error.message}`);
        if (status === 402) {
          creditsExhausted = true;
          break;
        }
        if (status === 429 || status === 524 || status === 529 || status === 502) {
          await sleep(1500 * attempt);
          continue;
        }
        break;
      }
    }

    if (modelError || files.length === 0) {
      failed += 1;
      registry.models[model.id] = {
        status: "error",
        reason: modelError?.message ?? "unknown",
        name: model.name,
        seenAt: new Date().toISOString(),
        files,
      };
    } else {
      succeeded += 1;
      registry.models[model.id] = {
        status: "ok",
        name: model.name,
        seenAt: new Date().toISOString(),
        files,
      };
    }

    saveRegistry(registry);
    writeIndex(registry);
  }

  async function worker() {
    while (true) {
      if (creditsExhausted) return;
      const index = cursor;
      cursor += 1;
      if (index >= queue.length) return;
      const model = queue[index];
      await processModel(model, index);
      await sleep(250);
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, queue.length) }, () => worker()),
  );

  writeIndex(registry);

  const okCount = Object.values(registry.models).filter(
    (m) => m.status === "ok",
  ).length;
  console.log(
    `\nDone. thisRun: ok=${succeeded} failed=${failed} skipped=${skipped}. registry ok=${okCount}/${models.length}`,
  );
  if (creditsExhausted) {
    console.log("Stopped early: OpenRouter credits exhausted (402). Progress saved.");
  }
  console.log(`Registry: ${registryPath}`);
  console.log(`Images:   ${aiRoot}`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exitCode = 1;
});
