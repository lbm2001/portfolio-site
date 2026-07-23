import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Mirror tsconfig.json's "@/*" -> "./*" so tests can import app/ and
    // component modules the same way the app itself does, not just lib/.
    alias: { "@": path.resolve(import.meta.dirname, ".") },
  },
  test: {
    include: ["tests/unit/**/*.test.{ts,tsx}"],
    environment: "node",
  },
});
