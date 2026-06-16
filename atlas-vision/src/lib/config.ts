// Client-side public config. Readable from both client and server.
//
// API_BASE is the base URL of the Route Resilience FastAPI backend.
// All backend routes live under `${API_BASE}/api` (e.g. `${API_BASE}/api/health`).
//
// Override at build/dev time by setting VITE_API_BASE in a .env file
// (see .env.example). The VITE_ prefix is required for Vite to expose it
// to the browser bundle. Never put secrets here — they ship to the client.
export const API_BASE =
  (import.meta.env.VITE_API_BASE as string) || "http://localhost:8000";
