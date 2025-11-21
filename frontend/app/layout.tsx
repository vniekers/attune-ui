import { UserProvider } from "@auth0/nextjs-auth0/client";
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CODEX OS",
  description: "Operate your Codex with clarity.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
      >
        <UserProvider>
        <header className="border-b">
          <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
            <h1 className="text-xl font-semibold tracking-tight">CODEX OS</h1>
            <p className="text-sm text-gray-500">
              Talk to your Codex (private API)
            </p>
          </div>
        </header>

        <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-6">
          {children}
        </main>

        <footer className="border-t">
          <div className="max-w-5xl mx-auto px-4 py-4 text-xs text-gray-500">
            © {new Date().getFullYear()} Tiny & Judy — CODEX OS
          </div>
        </footer>
        </UserProvider>
      </body>
    </html>
  );
}
