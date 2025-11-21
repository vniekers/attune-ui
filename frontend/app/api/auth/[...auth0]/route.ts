import { handleAuth } from "@auth0/nextjs-auth0/edge";
//import { handleAuth } from "@auth0/nextjs-auth0";
// Not needed with this version:
export const runtime = "edge";

export const GET = handleAuth();
export const POST = handleAuth();
