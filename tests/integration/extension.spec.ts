import {
  test,
  expect,
  chromium,
  type BrowserContext,
  type Page,
} from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";

const extensionPath = path.resolve("dist");
const galleryPath = path.resolve("tests/fixtures/pages/gallery.html");

async function launchExtensionContext(): Promise<{
  context: BrowserContext;
  extensionId: string;
  userDataDir: string;
}> {
  if (!fs.existsSync(path.join(extensionPath, "manifest.json"))) {
    throw new Error("dist/ missing. Run npm run build first.");
  }

  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "truepixel-pw-"));

  // Branded Google Chrome removed --load-extension (137+).
  // Playwright's Chromium / Chrome for Testing still supports it.
  const launchOptions: Parameters<typeof chromium.launchPersistentContext>[1] = {
    headless: false,
    channel: "chromium",
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu-sandbox",
    ],
  };

  // Allow overriding with Chrome for Testing binary if provided.
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
      timeout: 30_000,
    });
  }

  const extensionId = serviceWorker.url().split("/")[2] ?? "";
  if (!extensionId) {
    await context.close();
    throw new Error("Could not resolve extension id from service worker URL");
  }

  return { context, extensionId, userDataDir };
}

async function enableStubInference(
  context: BrowserContext,
  extensionId: string,
): Promise<void> {
  const page = await context.newPage();
  await page.goto(`chrome-extension://${extensionId}/popup.html`);
  await page.evaluate(async () => {
    await chrome.storage.local.set({ stubInference: true });
    await chrome.runtime.sendMessage({
      kind: "setup-models",
      requestId: crypto.randomUUID(),
    });
  });
  await page.close();
}

test.describe("TruePixel Chrome extension", () => {
  let context: BrowserContext;
  let extensionId: string;
  let userDataDir: string;

  test.beforeAll(async () => {
    ({ context, extensionId, userDataDir } = await launchExtensionContext());
    await enableStubInference(context, extensionId);
  });

  test.afterAll(async () => {
    await context?.close();
    if (userDataDir && fs.existsSync(userDataDir)) {
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }
  });

  test("popup reports stub models ready", async () => {
    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/popup.html`);
    await expect(page.locator("#modelStatus")).toContainText(/ready/i, {
      timeout: 15_000,
    });
    await expect(page.locator("#thresholdStatus")).toHaveText("65%");
    await page.close();
  });

  test("content script badges images on a local gallery page", async () => {
    await context.route("https://truepixel.test/**", async (route) => {
      const u = new URL(route.request().url());
      if (u.pathname === "/" || u.pathname === "/gallery.html") {
        const html = fs
          .readFileSync(galleryPath, "utf8")
          .replaceAll("../images/", "https://truepixel.test/images/");
        await route.fulfill({
          status: 200,
          contentType: "text/html",
          body: html,
        });
        return;
      }
      if (u.pathname.includes("images/")) {
        const file = path.resolve(
          "tests/fixtures/images",
          path.basename(u.pathname),
        );
        await route.fulfill({
          status: 200,
          contentType: "image/png",
          body: fs.readFileSync(file),
        });
        return;
      }
      await route.fulfill({ status: 404, body: "missing" });
    });

    const gallery: Page = await context.newPage();
    await gallery.goto("https://truepixel.test/gallery.html");
    await expect(gallery.locator("img")).toHaveCount(6);

    await expect
      .poll(async () => gallery.locator(".truepixel-badge").count(), {
        timeout: 60_000,
      })
      .toBeGreaterThanOrEqual(1);

    await expect
      .poll(
        async () => {
          const texts = await gallery
            .locator(".truepixel-badge")
            .allTextContents();
          return texts.some((t) => /AI\s+\d+%/i.test(t));
        },
        { timeout: 60_000 },
      )
      .toBe(true);

    await gallery.close();
  });

  test("options page persists threshold", async () => {
    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/options.html`);
    await page.locator("#threshold").fill("0.7");
    await page.locator("button[type=submit]").click();
    await expect(page.locator("#saved")).toBeVisible();

    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);
    await expect(popup.locator("#thresholdStatus")).toHaveText("70%");
    await popup.close();
    await page.close();
  });
});
