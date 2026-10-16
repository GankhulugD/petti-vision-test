"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ImagePicker from "@/components/ImagePicker";
import MatchList from "@/components/MatchList";
import ProfileView from "@/components/ProfileView";
import { api, getTaxonomy, type MatchResult, type Pet, type Profile, type Taxonomy } from "@/lib/api";

const MIN = 5;

type Labels = { breed: string; size: string; age: string; pattern: string; coat: string; colors: string[]; marks: string[] };

function fromProfile(p: Profile): Labels {
  return {
    breed: p.breed?.top[0]?.key ?? "",
    size: p.size.key,
    age: p.age.key,
    pattern: p.pattern.key,
    coat: p.coat.key,
    colors: p.colors.map((c) => c.key),
    marks: p.marks.map((m) => m.key),
  };
}

export default function RegisterPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [tax, setTax] = useState<Taxonomy | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [labels, setLabels] = useState<Labels | null>(null);
  const [info, setInfo] = useState({ name: "", status: "home", owner_name: "", contact: "", location: "", notes: "" });
  const [busy, setBusy] = useState<"" | "analyze" | "save">("");
  const [err, setErr] = useState("");
  const [done, setDone] = useState<{ pet: Pet; matches?: MatchResult[] } | null>(null);

  useEffect(() => {
    getTaxonomy().then(setTax).catch(() => {});
  }, []);

  const species = profile?.species.key ?? "dog";
  const breeds = useMemo(() => tax?.breeds.filter((b) => b.species === species) ?? [], [tax, species]);
  const marks = useMemo(() => tax?.marks.filter((m) => (m.species as string[]).includes(species)) ?? [], [tax, species]);

  const analyze = async () => {
    setBusy("analyze");
    setErr("");
    try {
      const p = await api.analyze(files);
      setProfile(p);
      setLabels(fromProfile(p));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  const save = async () => {
    if (!labels) return;
    setBusy("save");
    setErr("");
    try {
      const res = await api.createPet(files, {
        ...info,
        owner_labels: JSON.stringify({ species, ...labels }),
        event_at: info.status === "lost" ? new Date().toISOString() : undefined,
      });
      setDone(res);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  if (done) {
    return (
      <div className="space-y-6">
        <div className="card">
          <h1 className="text-3xl font-semibold tracking-tight">Бүртгэл амжилттай ✓</h1>
          <p className="mt-1 text-zinc-600">
            {done.pet.name || "Таны амьтан"} — ангилал: <code>{done.pet.category}</code>
          </p>
          <div className="mt-4 flex gap-2">
            <Link className="btn" href={`/pets/${done.pet.id}`}>Профайл харах</Link>
            <button className="btn-ghost" onClick={() => location.reload()}>Шинээр бүртгэх</button>
          </div>
        </div>
        {done.matches && (
          <section>
            <h2 className="mb-3 text-xl font-semibold">Олдсон амьтдын мэдээллээс тохирох магадлалтай</h2>
            <MatchList results={done.matches} />
          </section>
        )}
      </div>
    );
  }

  const sel = (k: keyof Labels, opts: { key: string; mn: string }[]) => (
    <select className="input" value={labels?.[k] as string} onChange={(e) => setLabels({ ...labels!, [k]: e.target.value })}>
      <option value="">— Мэдэхгүй —</option>
      {opts.map((o) => (
        <option key={o.key} value={o.key}>{o.mn}</option>
      ))}
    </select>
  );
  const multi = (k: "colors" | "marks", opts: { key: string; mn: string }[]) => (
    <div className="flex flex-wrap gap-1.5">
      {opts.map((o) => {
        const on = labels![k].includes(o.key);
        return (
          <button
            type="button"
            key={o.key}
            onClick={() => setLabels({ ...labels!, [k]: on ? labels![k].filter((x) => x !== o.key) : [...labels![k], o.key] })}
            className={`chip cursor-pointer transition ${on ? "border-zinc-900 bg-zinc-900 text-white" : "hover:border-zinc-400"}`}
          >
            {o.mn}
          </button>
        );
      })}
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Тэжээвэр амьтан бүртгүүлэх</h1>
        <p className="mt-1 text-zinc-600">
          Амьтныхаа {MIN}+ зургийг өөр өөр өнцгөөс оруулна уу. Систем автоматаар ангилж, та зөвшөөрөх/засах боломжтой. Таны баталсан мэдээлэл модел
          цааш сурахад ашиглагдана.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card space-y-4">
          <h2 className="font-semibold">1. Зураг</h2>
          <ImagePicker files={files} onChange={(f) => { setFiles(f); setProfile(null); }} min={MIN} hints />
          <button className="btn" disabled={files.length < MIN || busy !== ""} onClick={analyze}>
            {busy === "analyze" ? "Шинжилж байна…" : "Автоматаар ангилах"}
          </button>
          {profile && <ProfileView p={profile} compact />}
        </section>

        <section className={`card space-y-4 ${profile ? "" : "pointer-events-none opacity-50"}`}>
          <h2 className="font-semibold">2. Мэдээлэл шалгах, засах</h2>
          {labels && tax && (
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-sm sm:col-span-2">Үүлдэр {sel("breed", breeds)}</label>
              <label className="space-y-1 text-sm">Хэмжээ {sel("size", tax.size)}</label>
              <label className="space-y-1 text-sm">Нас {sel("age", tax.age)}</label>
              <label className="space-y-1 text-sm">Зүс {sel("pattern", tax.pattern)}</label>
              <label className="space-y-1 text-sm">Үс {sel("coat", tax.coat)}</label>
              <div className="space-y-1 text-sm sm:col-span-2">Өнгө {multi("colors", tax.colors)}</div>
              <div className="space-y-1 text-sm sm:col-span-2">Онцгой шинж {multi("marks", marks)}</div>
            </div>
          )}
          <h2 className="pt-2 font-semibold">3. Холбоо барих</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            <input className="input" placeholder="Амьтны нэр" value={info.name} onChange={(e) => setInfo({ ...info, name: e.target.value })} />
            <select className="input" value={info.status} onChange={(e) => setInfo({ ...info, status: e.target.value })}>
              <option value="home">Гэртээ байгаа (урьдчилан бүртгэх)</option>
              <option value="lost">Алга болсон</option>
            </select>
            <input className="input" placeholder="Эзний нэр" value={info.owner_name} onChange={(e) => setInfo({ ...info, owner_name: e.target.value })} />
            <input className="input" placeholder="Утас" value={info.contact} onChange={(e) => setInfo({ ...info, contact: e.target.value })} />
            <input className="input sm:col-span-2" placeholder="Байршил (дүүрэг, хороо)" value={info.location} onChange={(e) => setInfo({ ...info, location: e.target.value })} />
            <textarea className="input sm:col-span-2" rows={2} placeholder="Нэмэлт тайлбар" value={info.notes} onChange={(e) => setInfo({ ...info, notes: e.target.value })} />
          </div>
          <button className="btn w-full" disabled={!profile || busy !== ""} onClick={save}>
            {busy === "save" ? "Хадгалж байна…" : "Бүртгэх"}
          </button>
          {err && <p className="text-sm text-zinc-600">{err}</p>}
        </section>
      </div>
    </div>
  );
}
