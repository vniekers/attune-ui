// frontend/lib/devUser.ts
import { v4 as uuid } from "uuid";

const STORAGE_KEY = "codex-dev-user-id";

export function getDevUserId(): string {
  if (typeof window === "undefined") return "dev-server";
  let id = window.localStorage.getItem(STORAGE_KEY);
  if (!id) {
    // For your machine, you can also hardcode e.g. "tiny"
    id = uuid();
    window.localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}
