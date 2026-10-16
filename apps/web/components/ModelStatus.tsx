"use client";

import { useEffect, useState } from "react";

/** Loud warning when the model server is down or running fake (test) models. */
export default function ModelStatus() {
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        if (h?.models?.fake)
          setMsg("Тестийн (fake) модель ажиллаж байна — үр дүн бодит биш. PETTI_FAKE_MODELS-ийг унтрааж server-ээ дахин асаана уу.");
      })
      .catch(() => setMsg("Model server-тэй холбогдож чадсангүй."));
  }, []);
  if (!msg) return null;
  return (
    <div className="flex items-center justify-center gap-2 bg-zinc-900 px-4 py-2 text-center text-xs text-zinc-300">
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500" />
      {msg}
    </div>
  );
}
