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
  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] = {
    headless: false,
    channel: "chromium",
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,UseSkiaRenderer",
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
      for (const provider of providers) {
        const page = await context.newPage();
        const corpusBase = `http://127.0.0.1:${corpus.port}`;
        const limit = process.env.EVAL_SUITE_LIMIT ?? "0";
        const url =
          `chrome-extension://${extensionId}/eval.html` +
          `?corpus=${encodeURIComponent(corpusBase)}` +
          `&provider=${encodeURIComponent(provider)}` +
          `&autorun=1&threshold=0.65&limit=${encodeURIComponent(limit)}`;

        page.on("console", (msg) => {
          console.log(`[eval:${provider}]`, msg.type(), msg.text());
        });
        page.on("pageerror", (err) => {
          console.log(`[eval:${provider}] pageerror`, err.message);
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
          `[eval:${provider}] status=`,
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
            mode: `js-ext-${provider}-dual`,
            skipped: provider === "webgpu",
            error: parsed.message ?? "browser eval failed",
            providerRequested: provider,
          });
          if (provider !== "webgpu") {
            throw new Error(String(parsed.message ?? "browser eval failed"));
          }
        } else {
          browserResults.push({
            mode: `js-ext-${provider}-dual`,
            runtime: "extension-chromium",
            engine: "onnxruntime-web",
            preferEp: provider,
            distilledEp: parsed.providerActual,
            forensicsEp: parsed.providerActual,
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
