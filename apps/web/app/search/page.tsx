"use client";

import { useState } from "react";
import ImagePicker from "@/components/ImagePicker";
import MatchList from "@/components/MatchList";
import ProfileView from "@/components/ProfileView";
import { api, type MatchResult, type Profile } from "@/lib/api";

const POOLS = [
  { v: "lost", label: "Алга болсон амьтад (би амьтан олсон)" },
  { v: "found", label: "Олдсон амьтдын мэдээлэл (би амьтнаа хайж байна)" },
  { v: "lost,home", label: "Бүх бүртгэл" },
];

export default function SearchPage() {
  const [files, setFiles] = useState<File[]>([]);
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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Зургаар хайх</h1>
        <p className="mt-1 text-stone-600">Гудамжинд таарсан амьтны зургийг дарж оруулбал алга болсон бүртгэлүүдээс хамгийн төстэйг нь эрэмбэлж гаргана.</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
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
            {err && <p className="text-sm text-red-700">{err}</p>}
          </div>
          {res && (
            <div className="card">
              <h2 className="mb-2 font-semibold">Таны зургийн шинжилгээ</h2>
              <ProfileView p={res.query} compact />
              <p className="mt-3 text-xs text-stone-500">
                {res.searched} бүртгэлээс хайв · {res.query.timing_ms?.total} ms
              </p>
            </div>
          )}
          {res && pool === "lost" && (
            <div className="card space-y-3">
              <h2 className="font-semibold">Эзэн нь олдохгүй байна уу?</h2>
              <p className="text-sm text-stone-600">«Олдсон» гэж мэдэгдвэл шинээр алга болсон гэж бүртгэгдэх амьтадтай автоматаар тулгагдана.</p>
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
              <h2 className="mb-3 text-xl font-bold">Боломжит амьтад</h2>
              <MatchList results={res.results} />
            </>
          ) : (
            <div className="card text-stone-500">Хайлтын үр дүн энд гарна.</div>
          )}
        </section>
      </div>
    </div>
  );
}
