"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, STATUS_MN, type Pet } from "@/lib/api";

const TABS = [
  { v: "lost", label: "Алга болсон" },
  { v: "found", label: "Олдсон" },
  { v: "reunited", label: "Эзэндээ очсон" },
  { v: "home", label: "Бүртгэлтэй" },
];

export default function BoardPage() {
  const [tab, setTab] = useState("lost");
  const [pets, setPets] = useState<Pet[] | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setPets(null);
    api.pets(tab).then(setPets).catch((e) => setErr(e.message));
  }, [tab]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Lost & Found</h1>
          <p className="text-zinc-600">Олон нийтийн алга болсон / олдсон амьтдын самбар</p>
        </div>
        <Link href="/search" className="btn">Зургаар хайх</Link>
      </div>
      <div className="flex flex-wrap gap-2">
        {TABS.map((t) => (
          <button key={t.v} onClick={() => setTab(t.v)} className={tab === t.v ? "btn" : "btn-ghost"}>
            {t.label}
          </button>
        ))}
      </div>
      {err && <p className="text-zinc-600">{err}</p>}
      {pets === null ? (
        <p className="text-zinc-500">Ачаалж байна…</p>
      ) : pets.length === 0 ? (
        <p className="text-zinc-500">Одоогоор бүртгэл алга.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
          {pets.map((p) => (
            <Link key={p.id} href={`/pets/${p.id}`} className="card overflow-hidden p-0 transition hover:shadow-md">
              <div className="aspect-square bg-zinc-100">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={p.photos[0]} alt="" className="h-full w-full object-cover" />
              </div>
              <div className="space-y-1 p-3">
                <p className="font-semibold">{p.name || p.profile.breed?.looks_like[0]?.mn || p.profile.species.mn}</p>
                <p className="text-xs text-zinc-500">
                  {p.profile.species.mn} · {p.profile.size.mn.split(" (")[0]} · {p.profile.colors.map((c) => c.mn).join(", ")}
                </p>
                <p className="text-xs text-zinc-500">
                  {STATUS_MN[p.status]}
                  {p.location ? ` · ${p.location}` : ""}
                </p>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
