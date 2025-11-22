// attune-ui/frontend/app/api/chat/route.ts
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL; // e.g. https://your-render-app.onrender.com
const BACKEND_API_TOKEN = process.env.BACKEND_API_TOKEN; // 👈 NEW

if (!BACKEND_URL) {
  throw new Error("BACKEND_URL is not set in environment variables");
}

export async function POST(req: NextRequest) {
  const body = await req.json();

  // We’ll get this from the browser in the next step
  const userId =
    req.headers.get("x-codex-user-id") ??
    body.user_id ??
    "anonymous";

  const backendRes = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(BACKEND_API_TOKEN ? { Authorization: `Bearer ${BACKEND_API_TOKEN}` } : {}),
    },
    body: JSON.stringify({
      ...body,
      user_id: userId, // 👈 this is what FastAPI expects
    }),
  });

  const data = await backendRes.json();
  return NextResponse.json(data, { status: backendRes.status });
}
