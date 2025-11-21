// lib/serverConfig.ts
import "server-only"; // keep this server-side

export const BACKEND_URL = process.env.BACKEND_URL!;
export const BACKEND_TOKEN = process.env.BACKEND_TOKEN!;

if (!BACKEND_URL) {
  throw new Error("BACKEND_URL is not set in .env.local");
}
if (!BACKEND_TOKEN) {
  throw new Error("BACKEND_TOKEN is not set in .env.local");
}
