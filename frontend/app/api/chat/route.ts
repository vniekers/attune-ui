import { NextRequest, NextResponse } from "next/server";
import "server-only";
import { BACKEND_URL, BACKEND_TOKEN } from "@/lib/serverConfig";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const upstream = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${BACKEND_TOKEN}`,
      },
      body: JSON.stringify(body),
    });

    const text = await upstream.text(); // <-- capture raw text
    if (!upstream.ok) {
      // bubble up exact upstream body to see FastAPI validation/trace
      return new NextResponse(
        JSON.stringify({
          error: `Upstream error ${upstream.status}`,
          upstreamBody: safeJson(text),
        }),
        {
          status: upstream.status,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // return parsed JSON if possible, otherwise the raw text
    try {
      return NextResponse.json(JSON.parse(text));
    } catch {
      return new NextResponse(text, { status: 200, headers: { "Content-Type": "application/json" } });
    }
  } catch (err: any) {
    return NextResponse.json({ error: err?.message ?? "Proxy error" }, { status: 500 });
  }
}

// helper to wrap non-JSON text
function safeJson(s: string) {
  try { return JSON.parse(s); } catch { return s; }
}
