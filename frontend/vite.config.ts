import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const frontendPort = Number(env.VITE_FRONTEND_PORT ?? (mode === "production" ? "4173" : "5173"));
  const backendPort = env.VITE_BACKEND_PORT ?? (mode === "production" ? "8001" : "8000");
  const configuredApiBaseUrl = (env.VITE_API_BASE_URL ?? "").trim().replace(/\/$/, "");
  const apiBaseUrl = configuredApiBaseUrl || `http://localhost:${backendPort}`;

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: frontendPort,
      strictPort: true,
      proxy: {
        "/api": {
          target: apiBaseUrl,
          changeOrigin: true,
        },
      },
    },
    preview: {
      host: "0.0.0.0",
      port: frontendPort,
      strictPort: true,
    },
  };
});
