"use client";

import { useEffect, useRef, useState } from "react";
import VisionLab from "@/components/VisionLab";
import { api, type Profile } from "@/lib/api";

export default function AnalyzePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const req = useRef(0);

  // analyse as soon as photos are added/removed; stale responses are dropped
  useEffect(() => {
    const id = ++req.current;
    setProfile(null);
    setErr("");
    if (!files.length) {
      setBusy(false);
      return;
    }
    setBusy(true);
    const timer = setTimeout(async () => {
      try {
        const p = await api.analyze(files);
        if (id === req.current) setProfile(p);
      } catch (e) {
        if (id === req.current) setErr((e as Error).message);
      } finally {
        if (id === req.current) setBusy(false);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [files]);

  return (
    <div className="space-y-10">
      <header className="max-w-2xl">
        <p className="eyebrow">Petti Vision · v0.5.2</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Амьтныг зургаар нь таних</h1>
        <p className="mt-3 text-[15px] leading-7 text-zinc-500">
          Зураг оруулахад модель амьтныг олж, үүлдэр, нас, хэмжээ, өнгө зүс, онцгой шинжийг тодорхойлно. Ингэхдээ юу хийж байгааг шууд харуулна.
        </p>
      </header>
      <VisionLab files={files} profile={profile} busy={busy} onFiles={setFiles} />
      {err && (
        <p className="fade-in flex items-center gap-2 text-sm text-zinc-600">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-500" />
          {err}
        </p>
      )}
    </div>
  );
}
