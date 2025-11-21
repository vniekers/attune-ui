// app/api/search/route.ts
import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, BACKEND_TOKEN } from "@/lib/serverConfig";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const q = searchParams.get("q") ?? "";
  const limit = searchParams.get("limit") ?? "10";

  try {
    const upstream = await fetch(`${BACKEND_URL}/search?q=${encodeURIComponent(q)}&limit=${encodeURIComponent(limit)}`, {
      headers: { "Authorization": `Bearer ${BACKEND_TOKEN}` },
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      return NextResponse.json(
        { error: `Upstream error ${upstream.status}: ${text}` },
        { status: upstream.status }
      );
    }

    const data = await upstream.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ error: err?.message ?? "Proxy error" }, { status: 500 });
  }
}
