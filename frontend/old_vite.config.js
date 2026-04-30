import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var _b, _c, _d, _e, _f;
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var frontendPort = Number((_b = env.VITE_FRONTEND_PORT) !== null && _b !== void 0 ? _b : (mode === "production" ? "4173" : "5173"));
    var backendPort = (_c = env.VITE_BACKEND_PORT) !== null && _c !== void 0 ? _c : (mode === "production" ? "8001" : "8000");
    var configuredApiBaseUrl = ((_d = env.VITE_API_BASE_URL) !== null && _d !== void 0 ? _d : "").trim().replace(/\/$/, "");
    var apiBaseUrl = configuredApiBaseUrl || "http://localhost:".concat(backendPort);
    var configuredAllowedHosts = ((_f = (_e = env.VITE_ALLOWED_HOSTS) !== null && _e !== void 0 ? _e : env.ALLOWED_HOSTS) !== null && _f !== void 0 ? _f : '')
        .split(',')
        .map(function (host) { return host.trim(); })
        .filter(Boolean);
    var allowedHosts = configuredAllowedHosts.length > 0 ? configuredAllowedHosts : ['faiht.ai', 'www.faiht.ai'];
    return {
        plugins: [react()],
        server: {
            host: "0.0.0.0",
            port: frontendPort,
            strictPort: true,
            allowedHosts: allowedHosts,
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
            allowedHosts: allowedHosts
        },
    };
});
