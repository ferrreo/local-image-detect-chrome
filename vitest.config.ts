import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "extension/src/shared"),
      "@lib": path.resolve(__dirname, "extension/src/lib"),
    },
  },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    setupFiles: ["tests/setup/canvas-polyfill.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json-summary"],
    },
  },
});

