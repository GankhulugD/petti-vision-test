"use client";

import { useState } from "react";
import ImagePicker from "@/components/ImagePicker";
import ProfileView from "@/components/ProfileView";
import { api, type Profile } from "@/lib/api";

export default function AnalyzePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const run = async () => {
    setBusy(true);
    setErr("");
    try {
      setProfile(await api.analyze(files));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="space-y-4">
        <div>
          <h1 className="text-2xl font-bold">Амьтныг зургаар таних</h1>
          <p className="mt-1 text-stone-600">
            Төрөл, ойролцоо үүлдэр (эрлийз бол аль үүлдрүүдтэй төстэй), хэмжээ, нас, өнгө зүс, онцгой шинжийг нэг дор гаргана. Нэг амьтны хэд хэдэн
            зураг оруулбал илүү нарийвчлалтай.
          </p>
        </div>
        <ImagePicker files={files} onChange={setFiles} />
        <div className="flex gap-2">
          <button className="btn" disabled={!files.length || busy} onClick={run}>
            {busy ? "Шинжилж байна…" : "Шинжлэх"}
          </button>
          {files.length > 0 && (
            <button className="btn-ghost" onClick={() => { setFiles([]); setProfile(null); }}>
              Цэвэрлэх
            </button>
          )}
        </div>
        {err && <p className="text-sm text-red-700">{err}</p>}
      </section>
      <section className="card">
        {profile ? <ProfileView p={profile} /> : <p className="text-stone-500">Үр дүн энд гарна.</p>}
      </section>
    </div>
  );
}
