import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API paths the dev server should forward to the backend.
const API_PATHS = ["/ask", "/ask-audio", "/race", "/meta", "/sample-queries", "/health", "/docs"];

// Where the backend is. Port 8000 is the default, but it is a popular port, so
// the run scripts pass an explicit one when they have to move.
const API_TARGET = process.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxying means the browser talks to one origin in development, exactly as
    // it does in production where the API serves the built UI. Without this the
    // frontend has to be told the API's address, and forgetting produces 404s
    // from the dev server rather than anything that names the cause.
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [path, { target: API_TARGET, changeOrigin: true }]),
    ),
  },
});
