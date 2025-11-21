"use client";

import { useState } from "react";

type ChatMessage = { role: "user" | "assistant" | "system"; content: string };

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "system", content: "You are CODEX OS." },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function sendMessage(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;

    const nextMessages = [
      ...messages,
      { role: "user", content: input.trim() } as ChatMessage,
    ];
    setMessages(nextMessages);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages }),
      });

      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }

      // Expecting { reply: "..." } OR OpenAI-style choices?
      const data = await res.json();

      // Flexible response mapping (supports common shapes)
      const assistantText =
        data?.reply ??
        data?.message ??
        data?.choices?.[0]?.message?.content ??
        data?.choices?.[0]?.text ??
        JSON.stringify(data);

      setMessages([...nextMessages, { role: "assistant", content: assistantText }]);
    } catch (err: any) {
      setError(err?.message ?? "Unknown error");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl w-full flex flex-col gap-4">
      <div className="border rounded-2xl p-4 min-h-[320px]">
        {messages
          .filter(m => m.role !== "system")
          .map((m, i) => (
            <div
              key={i}
              className={`mb-3 ${
                m.role === "user" ? "text-right" : "text-left"
              }`}
            >
              <div
                className={`inline-block rounded-2xl px-4 py-2 ${
                  m.role === "user"
                    ? "bg-gray-200"
                    : "bg-gray-100"
                }`}
              >
                <pre className="whitespace-pre-wrap break-words font-sans">
                  {m.content}
                </pre>
              </div>
            </div>
          ))}
        {sending && (
          <div className="text-sm text-gray-500 italic">Thinking…</div>
        )}
        {error && (
          <div className="text-sm text-red-600">
            Error: {error}
          </div>
        )}
      </div>

      <form onSubmit={sendMessage} className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message…"
          className="flex-1 border rounded-xl px-3 py-2"
        />
        <button
          type="submit"
          disabled={sending}
          className="rounded-xl px-4 py-2 border hover:bg-gray-50 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
