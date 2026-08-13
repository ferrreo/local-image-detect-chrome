import {
  test,
  expect,
  chromium,
  type BrowserContext,
} from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import http from "node:http";

const extensionPath = path.resolve("dist");
const root = path.resolve(".");
const corpusDir = path.join(root, "benchmark/openrouter");
const outDir = path.join(root, "benchmark/eval-suite");

const providers = (process.env.EVAL_SUITE_BROWSER_PROVIDERS ?? "wasm,webgpu")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

/** `ort-web` | `zig` | `auto` — comma list. Default ort-web so Zig does not shadow. */
const engines = (process.env.EVAL_SUITE_BROWSER_ENGINES ?? "ort-web")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

function contentType(filePath: string): string {
  if (filePath.endsWith(".json")) return "application/json";
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".jpg") || filePath.endsWith(".jpeg")) return "image/jpeg";
  if (filePath.endsWith(".webp")) return "image/webp";
  return "application/octet-stream";
}

async function startCorpusServer(): Promise<{
  port: number;
  close: () => Promise<void>;
}> {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const rel = decodeURIComponent(url.pathname.replace(/^\//, ""));
    const filePath = path.join(corpusDir, rel);
    if (!filePath.startsWith(corpusDir) || !fs.existsSync(filePath)) {
      res.writeHead(404);
      res.end("missing");
      return;
    }
    const body = fs.readFileSync(filePath);
    res.writeHead(200, {
      "Content-Type": contentType(filePath),
      "Cache-Control": "no-cache",
      "Access-Control-Allow-Origin": "*",
    });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () =>
      new Promise((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

async function launchExtensionContext(): Promise<{
  context: BrowserContext;
  extensionId: string;
  userDataDir: string;
}> {
  if (!fs.existsSync(path.join(extensionPath, "manifest.json"))) {
    throw new Error("dist/ missing. Run npm run build first.");
  }
  if (!fs.existsSync(path.join(extensionPath, "models"))) {
    throw new Error(
      "dist/models missing. Run npm run setup:models && npm run build",
    );
  }

  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "truepixel-eval-"));
  // Extensions need Chromium's new headless; headed only when EVAL_SUITE_HEADED=1.
  const headed = process.env.EVAL_SUITE_HEADED === "1";
  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] = {
    headless: !headed,
    channel: "chromium",
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,UseSkiaRenderer",
      ...(headed ? [] : ["--headless=new"]),
    ],
  };
  if (process.env.CHROME_FOR_TESTING_PATH) {
    delete launchOptions.channel;
    launchOptions.executablePath = process.env.CHROME_FOR_TESTING_PATH;
  }

  const context = await chromium.launchPersistentContext(
    userDataDir,
    launchOptions,
  );
  let serviceWorker = context.serviceWorkers()[0];
  if (!serviceWorker) {
    serviceWorker = await context.waitForEvent("serviceworker", {
      timeout: 60_000,
    });
  }
  const extensionId = serviceWorker.url().split("/")[2] ?? "";
  if (!extensionId) {
    await context.close();
    throw new Error("Could not resolve extension id");
  }
  return { context, extensionId, userDataDir };
}

test.describe("Offline extension eval suite", () => {
  test.setTimeout(30 * 60_000);

  test("run browser ORT providers against OpenRouter corpus", async () => {
    test.skip(
      !fs.existsSync(path.join(corpusDir, "index.json")),
      "OpenRouter corpus missing",
    );

    const corpus = await startCorpusServer();
    const { context, extensionId, userDataDir } =
      await launchExtensionContext();
    const browserResults: unknown[] = [];

    try {
      for (const engine of engines) {
        for (const provider of providers) {
          const page = await context.newPage();
          const corpusBase = `http://127.0.0.1:${corpus.port}`;
          const limit = process.env.EVAL_SUITE_LIMIT ?? "0";
          const url =
            `chrome-extension://${extensionId}/eval.html` +
            `?corpus=${encodeURIComponent(corpusBase)}` +
            `&provider=${encodeURIComponent(provider)}` +
            `&engine=${encodeURIComponent(engine)}` +
            `&autorun=1&threshold=0.65&limit=${encodeURIComponent(limit)}`;

          const tag = `${engine}/${provider}`;
          page.on("console", (msg) => {
            console.log(`[eval:${tag}]`, msg.type(), msg.text());
          });
          page.on("pageerror", (err) => {
            console.log(`[eval:${tag}] pageerror`, err.message);
          });
          await page.goto(url);
          // Surface early status while waiting for full corpus.
          await page.waitForFunction(
            () => {
              const el = document.getElementById("statusLine");
              return Boolean(el && el.textContent && el.textContent !== "Idle");
            },
            null,
            { timeout: 60_000 },
          );
          console.log(
            `[eval:${tag}] status=`,
            await page.locator("#statusLine").innerText(),
          );
          await page.waitForSelector(
            '#done[data-state="done"], #done[data-state="error"]',
            { timeout: 25 * 60_000, state: "attached" },
          );
          const state = await page.getAttribute("#done", "data-state");
          const raw = await page.locator("#suite-result").innerText();
          const parsed = JSON.parse(raw) as Record<string, unknown>;
          if (state === "error") {
            browserResults.push({
              mode: `js-ext-${provider}-cascade`,
              skipped: provider === "webgpu",
              error: parsed.message ?? "browser eval failed",
              providerRequested: provider,
              engineRequested: engine,
            });
            if (provider !== "webgpu") {
              throw new Error(String(parsed.message ?? "browser eval failed"));
            }
          } else {
            const visualEngine =
              typeof parsed.visualEngine === "string" && parsed.visualEngine
                ? parsed.visualEngine
                : "onnxruntime-web";
            const modePrefix =
              visualEngine === "zig-ort-wasm" ? "js-ext-zig" : "js-ext";
            // Prefer EPs from cascade detail tags (CF may differ under #29599).
            const rows = Array.isArray(parsed.rows) ? parsed.rows : [];
            const detail = rows
              .map((r) =>
                r && typeof r === "object" && "tiers" in r
                  ? String((r as { tiers?: unknown }).tiers ?? "")
                  : "",
              )
              .find((t) => t.includes("distilledEp="));
            const epFrom = (key: string, fallback: unknown) => {
              const m = detail?.match(new RegExp(`${key}=([a-z0-9-]+)`));
              return m?.[1] ?? fallback;
            };
            const distilledEp = epFrom("distilledEp", parsed.providerActual);
            const forensicsEp = epFrom("forensicsEp", distilledEp);
            const wasmThreads = epFrom("wasmThreads", undefined);
            const webgpuSkip = epFrom("webgpuSkip", undefined);
            browserResults.push({
              mode: `${modePrefix}-${provider}-cascade`,
              runtime: "extension-chromium",
              engine: visualEngine,
              visualMode: "cascade",
              preferEp: provider,
              distilledEp,
              forensicsEp,
              ...(wasmThreads ? { wasmThreads } : {}),
              ...(webgpuSkip ? { webgpuSkip } : {}),
              gpuAvailable: parsed.gpuAvailable,
              threshold: parsed.threshold,
              balancedAccuracy: parsed.balancedAccuracy,
              confusion: parsed.confusion,
              timing: parsed.timing,
              rows: parsed.rows,
            });
          }
          await page.close();
        }
      }
    } finally {
      await context.close();
      await corpus.close();
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }

    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, "browser-latest.json");
    fs.writeFileSync(
      outPath,
      JSON.stringify(
        {
          generatedAt: new Date().toISOString(),
          providers,
          engines,
          results: browserResults,
        },
        null,
        2,
      ) + "\n",
    );

    const hardFailures = browserResults.filter(
      (r) =>
        typeof r === "object" &&
        r !== null &&
        "error" in r &&
        !(r as { skipped?: boolean }).skipped,
    );
    expect(hardFailures, JSON.stringify(hardFailures)).toEqual([]);
  });
});
