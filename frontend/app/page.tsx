"use client";

import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState<"analysis" | "law" | "creative" | "legal">("analysis");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);

  // Create or load a stable user_id once on the client
  const [userId] = useState(() => {
    try {
      const key = "codex-user-id";
      const existing = typeof window !== "undefined" ? localStorage.getItem(key) : null;
      if (existing) return existing;
      const id =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2);
      if (typeof window !== "undefined") localStorage.setItem(key, id);
      return id;
    } catch {
      return "codex-local-user"; // fallback
    }
  });

  const send = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setOutput("");

    try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        messages: [
          { role: "system", content: "You are CODEX OS." },
          { role: "user", content: message }
        ],
        max_tokens: 512,
        temperature: 0.2,
        top_p: 0.95,
        use_shared_memory: true,
        send_memory_to_llm: true
      }),
    });


      if (!res.ok) {
        const payload = await res.text();
        throw new Error(`HTTP ${res.status}: ${payload}`);
      }

      const data = await res.json();
      // --- new robust reply extraction ---
const msgObj =
  data?.reply ??
  data?.message ??
  data?.choices?.[0]?.message ??
  null;

let replyText = "";

// Case 1: OpenAI-style message object
if (msgObj && typeof msgObj === "object") {
  if (typeof msgObj.content === "string") {
    replyText = msgObj.content;
  } else if (Array.isArray(msgObj.content)) {
    replyText = msgObj.content
      .map((p: any) =>
        typeof p === "string" ? p : (p?.text ?? p?.content ?? "")
      )
      .join("");
  }
}

// Case 2: already a plain string elsewhere
if (!replyText) {
  replyText =
    data?.text ??
    data?.output ??
    (typeof data === "string" ? data : "");
}

// Final fallback: stringify whatever’s left
if (!replyText) {
  replyText = JSON.stringify(data, null, 2);
}

setOutput(replyText);
      //const reply =
      //  data.reply ??
      //  data.text ??
      //  data.choices?.[0]?.message?.content ??
      //  data.choices?.[0]?.text ??
      //  JSON.stringify(data, null, 2);

      // setOutput(reply);
    } catch (err: any) {
      setOutput(`Error: ${err.message || "could not reach API."}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-2xl mx-auto mt-10 font-[system-ui]">
      <h1 className="text-3xl font-bold mb-2">CODEX OS</h1>
      <p className="text-gray-600 mb-6">Talk to your private API.</p>

      <label className="block mb-2">
        Mode:{" "}
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as any)}
          className="border rounded px-2 py-1"
        >
          <option value="analysis">Analysis</option>
          <option value="law">Law</option>
          <option value="creative">Creative</option>
          <option value="legal">Legal</option>
        </select>
      </label>

      <textarea
        rows={5}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message…"
        className="w-full border rounded p-2 mb-3"
      />

      <button
        onClick={send}
        disabled={loading}
        className="px-4 py-2 border rounded hover:bg-gray-50 disabled:opacity-50"
      >
        {loading ? "Sending…" : "Send"}
      </button>

      <pre className="whitespace-pre-wrap mt-6">{output}</pre>
    </main>
  );
}
