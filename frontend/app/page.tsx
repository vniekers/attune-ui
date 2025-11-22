"use client";

import { useState, FormEvent } from "react";

type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

type ActiveUser = "tiny" | "judy";

export default function Home() {
  const [message, setMessage] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  // Who is currently using Codex OS in this browser
  const [activeUser, setActiveUser] = useState<ActiveUser>("tiny");

  // Later we can expose this as a toggle; for now keep false
  const useSharedMemory = false;

  const send = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!message.trim() || loading) return;

    const userId = activeUser; // "tiny" or "judy"

    setLoading(true);
    setOutput("");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // used by the proxy to forward user_id to FastAPI
          "x-codex-user-id": userId,
        },
        body: JSON.stringify({
          user_id: userId,
          messages: [
            { role: "system", content: "You are CODEX OS." },
            { role: "user", content: message },
          ] as ChatMessage[],
          max_tokens: 512,
          temperature: 0.2,
          top_p: 0.95,
          use_shared_memory: useSharedMemory,
          send_memory_to_llm: true,
        }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }

      const data = await res.json();

      // Robust extraction of reply text (same logic you had)
      const msgObj =
        data?.reply ??
        data?.message ??
        data?.choices?.[0]?.message ??
        null;

      let replyText = "";

      if (msgObj && typeof msgObj === "object") {
        if (typeof (msgObj as any).content === "string") {
          replyText = (msgObj as any).content;
        } else if (Array.isArray((msgObj as any).content)) {
          replyText = (msgObj as any).content
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

  return (
    <main className="max-w-2xl mx-auto mt-10 font-[system-ui]">
      <h1 className="text-3xl font-bold mb-2">CODEX OS</h1>

      <p className="text-gray-600 mb-4">
        Talk to your private API as{" "}
        <span className="font-semibold uppercase">{activeUser}</span>.
      </p>

      {/* Tiny / Judy switch */}
      <div className="flex items-center gap-3 mb-4 text-sm">
        <span className="text-gray-500">Active user:</span>
        <button
          type="button"
          onClick={() => setActiveUser("tiny")}
          className={`px-3 py-1 rounded border ${
            activeUser === "tiny"
              ? "border-black bg-black text-white"
              : "border-gray-300"
          }`}
        >
          Tiny
        </button>
        <button
          type="button"
          onClick={() => setActiveUser("judy")}
          className={`px-3 py-1 rounded border ${
            activeUser === "judy"
              ? "border-black bg-black text-white"
              : "border-gray-300"
          }`}
        >
          Judy
        </button>
      </div>

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
