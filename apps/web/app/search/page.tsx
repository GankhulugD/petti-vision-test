"use client";

import { useState } from "react";
import ImagePicker from "@/components/ImagePicker";
import MatchList from "@/components/MatchList";
import VisionLab from "@/components/VisionLab";
import { api, type MatchResult, type Profile } from "@/lib/api";

const POOLS = [
  { v: "lost", label: "Алга болсон амьтад (би амьтан олсон)" },
  { v: "found", label: "Олдсон амьтдын мэдээлэл (би амьтнаа хайж байна)" },
  { v: "lost,home", label: "Бүх бүртгэл" },
];

export default function SearchPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [ranFiles, setRanFiles] = useState<File[]>([]);
  const [pool, setPool] = useState("lost");
  const [res, setRes] = useState<{ query: Profile; searched: number; results: MatchResult[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [report, setReport] = useState({ contact: "", location: "" });
  const [reported, setReported] = useState(false);

  const run = async () => {
    setBusy(true);
    setErr("");
    setReported(false);
    setRanFiles(files);
    setRes(null);
    try {
      setRes(await api.search(files, pool, 10));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const reportFound = async () => {
    setBusy(true);
    try {
      await api.createPet(files, { status: "found", ...report, event_at: new Date().toISOString() });
      setReported(true);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-10">
      <header className="max-w-2xl">
        <p className="eyebrow">Lost &amp; Found</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Зургаар хайх</h1>
        <p className="mt-3 text-[15px] leading-7 text-zinc-500">
          Таарсан амьтны зургийг оруулбал бүртгэлтэй амьтдаас хамгийн төстэйг нь төрх болон шинжээр нь эрэмбэлж гаргана.
        </p>
      </header>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] lg:gap-12">
        <section className="space-y-4">
          <div className="card space-y-4">
            <ImagePicker files={files} onChange={setFiles} max={6} />
            <select className="input" value={pool} onChange={(e) => setPool(e.target.value)}>
              {POOLS.map((p) => (
                <option key={p.v} value={p.v}>{p.label}</option>
              ))}
            </select>
            <button className="btn w-full" disabled={!files.length || busy} onClick={run}>
              {busy ? "Хайж байна…" : "Хайх"}
            </button>
            {err && <p className="text-sm text-zinc-600">{err}</p>}
          </div>
          {(busy || res) && <VisionLab files={ranFiles} profile={res?.query ?? null} busy={busy && !res} searched={res?.searched} compact />}
          {res && pool === "lost" && (
            <div className="card space-y-3">
              <h2 className="text-sm font-semibold">Эзэн нь олдохгүй байна уу?</h2>
              <p className="text-sm text-zinc-500">«Олдсон» гэж мэдэгдвэл шинээр алга болсон гэж бүртгэгдэх амьтадтай автоматаар тулгагдана.</p>
              <input className="input" placeholder="Таны утас" value={report.contact} onChange={(e) => setReport({ ...report, contact: e.target.value })} />
              <input className="input" placeholder="Хаана олсон бэ?" value={report.location} onChange={(e) => setReport({ ...report, location: e.target.value })} />
              <button className="btn-ghost w-full" disabled={busy || reported} onClick={reportFound}>
                {reported ? "Мэдэгдлээ ✓" : "Олдсон амьтан гэж мэдэгдэх"}
              </button>
            </div>
          )}
        </section>
        <section>
          {res ? (
            <>
              <div className="mb-4 flex items-baseline justify-between">
                <h2 className="text-lg font-semibold tracking-tight">Боломжит амьтад</h2>
                <span className="eyebrow">{res.searched} бүртгэлээс</span>
              </div>
              <MatchList results={res.results} />
            </>
          ) : (
            <div className="grid min-h-48 place-items-center rounded-2xl border border-dashed border-zinc-200 text-sm text-zinc-400">
              {busy ? <span className="shimmer font-medium">Бүртгэлүүдтэй тулгаж байна</span> : "Хайлтын үр дүн энд гарна"}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
