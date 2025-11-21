"use client";

import { useUser } from "@auth0/nextjs-auth0/client";

export function HeaderUser() {
  const { user, isLoading } = useUser();

  if (isLoading) {
    return <span className="text-xs text-gray-400">Checking session…</span>;
  }

  if (!user) {
    return (
      <a
        href="/api/auth/login"
        className="text-sm text-blue-600 hover:underline"
      >
        Log in
      </a>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-gray-600">
        {user.name || user.email || user.sub}
      </span>
      <a
        href="/api/auth/logout"
        className="text-xs text-blue-600 hover:underline"
      >
        Log out
      </a>
    </div>
  );
}
