import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    ...devices["Desktop Chrome"],
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "integration",
      testDir: "tests/integration",
      timeout: 120_000,
    },
    {
      name: "eval",
      testDir: "tests/eval",
      // Full OpenRouter corpus + dual ORT can take a while on CPU.
      timeout: 30 * 60_000,
    },
  ],
});
