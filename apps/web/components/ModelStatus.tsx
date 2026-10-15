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
          setMsg("⚠ Model server PETTI_FAKE_MODELS=1 (тестийн санамсаргүй модел) горимд ажиллаж байна — үр дүн бодит биш. `unset PETTI_FAKE_MODELS` хийгээд server-ээ дахин асаана уу.");
      })
      .catch(() => setMsg("⚠ Model server (8765) холбогдохгүй байна."));
  }, []);
  if (!msg) return null;
  return <div className="bg-red-600 px-4 py-2 text-center text-sm font-semibold text-white">{msg}</div>;
}
