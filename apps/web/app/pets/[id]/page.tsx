"use client";

import { use, useEffect, useState } from "react";
import MatchList from "@/components/MatchList";
import ProfileView from "@/components/ProfileView";
import { api, STATUS_MN, type MatchResult, type Pet } from "@/lib/api";

export default function PetPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [pet, setPet] = useState<Pet | null>(null);
  const [matches, setMatches] = useState<MatchResult[] | null>(null);
  const [err, setErr] = useState("");

  const load = () =>
    api
      .matches(id)
      .then((r) => {
        setPet(r.pet);
        setMatches(r.results);
      })
      .catch((e) => setErr(e.message));

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const setStatus = async (s: string) => {
    const r = await api.setStatus(id, s);
    setPet(r.pet);
    if (r.matches) setMatches(r.matches);
  };

  if (err) return <p className="text-zinc-600">{err}</p>;
  if (!pet) return <p className="text-zinc-500">Ачаалж байна…</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">{pet.name || pet.profile.species.mn}</h1>
          <p className="text-zinc-600">
            {STATUS_MN[pet.status]}
            {pet.location ? ` · ${pet.location}` : ""}
            {pet.contact ? ` · ☎ ${pet.contact}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {pet.status === "home" && <button className="btn" onClick={() => setStatus("lost")}>Алга болсон гэж мэдэгдэх</button>}
          {(pet.status === "lost" || pet.status === "found") && (
            <button className="btn" onClick={() => setStatus("reunited")}>Эзэндээ очсон ✓</button>
          )}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        {pet.photos.map((src) => (
          <div key={src} className="aspect-square overflow-hidden rounded-xl bg-zinc-100">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt="" className="h-full w-full object-cover" />
          </div>
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-[1fr_1.3fr]">
        <section className="card">
          <ProfileView p={pet.profile} />
        </section>
        <section>
          <h2 className="mb-3 text-xl font-semibold">
            {pet.status === "found" ? "Алга болсон бүртгэлээс тохирох магадлалтай" : "Олдсон мэдээллээс тохирох магадлалтай"}
          </h2>
          {matches && <MatchList results={matches} />}
        </section>
      </div>
    </div>
  );
}
