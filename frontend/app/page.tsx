"use client";

import { useState, FormEvent } from "react";
import { useUser } from "@auth0/nextjs-auth0/client";

type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export default function Home() {
  const { user, isLoading } = useUser();

  const [message, setMessage] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  const send = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!message.trim()) return;

    // Require login
    if (!user) {
      setOutput("Please log in to use CODEX OS.");
      return;
    }

    setLoading(true);
    setOutput("");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: user.sub, // ✅ Auth0 user ID
          messages: [
            { role: "system", content: "You are CODEX OS." },
            { role: "user", content: message },
          ] as ChatMessage[],
          max_tokens: 512,
          temperature: 0.2,
          top_p: 0.95,
          use_shared_memory: false,
          send_memory_to_llm: true,
        }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }

      const data = await res.json();

      // Robust extraction of reply text
      const msgObj =
        data?.reply ??
        data?.message ??
        data?.choices?.[0]?.message ??
        null;

      let replyText = "";

      if (msgObj && typeof msgObj === "object") {
        if (typeof msgObj.content === "string") {
          replyText = msgObj.content;
        } else if (Array.isArray(msgObj.content)) {
          replyText = msgObj.content
            .map((p: any) =>
              typeof p === "string" ? p : p?.text ?? p?.content ?? ""
            )
            .join("");
        }
      }

      if (!replyText) {
        replyText =
          data?.text ??
          data?.output ??
          (typeof data === "string" ? data : "");
      }

      if (!replyText) {
        replyText = JSON.stringify(data, null, 2);
      }

      setOutput(replyText);
    } catch (err: any) {
      setOutput(`Error: ${err.message || "could not reach API."}`);
    } finally {
      setLoading(false);
    }
  };

  // While Auth0 is checking session
  if (isLoading) {
    return (
      <main className="max-w-2xl mx-auto mt-10">
        <p className="text-gray-600">Checking session…</p>
      </main>
    );
  }

  // If not logged in, show a friendly gate
  if (!user) {
    return (
      <main className="max-w-2xl mx-auto mt-10 text-center">
        <h1 className="text-2xl font-semibold mb-4">CODEX OS</h1>
        <p className="text-gray-600 mb-6">
          You must log in to use CODEX OS.
        </p>
        <a
          href="/api/auth/login"
          className="inline-block rounded border px-4 py-2 text-blue-600 border-blue-600 hover:bg-blue-50"
        >
          Log in
        </a>
      </main>
    );
  }

  // Logged-in view
  return (
    <main className="max-w-2xl mx-auto mt-10 font-[system-ui]">
      <h1 className="text-3xl font-bold mb-2">CODEX OS</h1>
      <p className="text-gray-600 mb-6">
        Talk to your private API as {user.name || user.email || user.sub}.
      </p>

      <form onSubmit={send}>
        <textarea
          rows={5}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message…"
          className="w-full border rounded p-2 mb-3"
        />

        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 border rounded hover:bg-gray-50 disabled:opacity-50"
        >
          {loading ? "Sending…" : "Send"}
        </button>
      </form>

      <pre className="whitespace-pre-wrap mt-6">{output}</pre>
    </main>
  );
}
