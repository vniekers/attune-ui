import { NextRequest, NextResponse } from "next/server";

const BASIC_USER = process.env.BASIC_AUTH_USER;
const BASIC_PASS = process.env.BASIC_AUTH_PASS;

export function middleware(req: NextRequest) {
  // If creds are not configured, don't block (fails open).
  // In production you'll have them set via Vercel env.
  if (!BASIC_USER || !BASIC_PASS) {
    return NextResponse.next();
  }

  const authHeader = req.headers.get("authorization");

  if (!authHeader) {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="Codex OS"',
      },
    });
  }

  const [scheme, encoded] = authHeader.split(" ");

  if (scheme !== "Basic" || !encoded) {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="Codex OS"',
      },
    });
  }

  // Basic auth is "username:password" base64 encoded
  let decoded: string;
  try {
    decoded = globalThis.atob(encoded);
  } catch {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="Codex OS"',
      },
    });
  }

  const [user, pass] = decoded.split(":", 2);

  if (user === BASIC_USER && pass === BASIC_PASS) {
    return NextResponse.next();
  }

  return new NextResponse("Authentication required", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="Codex OS"',
    },
  });
}

// Protect everything except static assets and favicon
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
