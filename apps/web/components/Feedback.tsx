"use client";

// "Wrong answer?" — the user's correction becomes training data, but the photo
// is only kept when they explicitly agree (consent unchecked by default).

import { useEffect, useState } from "react";
import { api, getTaxonomy, type Opt, type Profile } from "@/lib/api";

export default function Feedback({ files, profile }: { files: File[]; profile: Profile }) {
  const [open, setOpen] = useState(false);
  const [breeds, setBreeds] = useState<Opt[]>([]);
  const [breed, setBreed] = useState("");
  const [noMarks, setNoMarks] = useState(false);
  const [consent, setConsent] = useState(false);
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");

  useEffect(() => {
    if (!open || breeds.length) return;
    getTaxonomy()
      .then((t) => setBreeds(t.breeds.filter((b) => !b.species || b.species === profile.species.key || (Array.isArray(b.species) && b.species.includes(profile.species.key)))))
      .catch(() => setBreeds([]));
  }, [open, breeds.length, profile.species.key]);

  if (state === "sent")
    return <p className="fade-in text-sm text-zinc-500">Баярлалаа — засварыг тань хадгаллаа{consent ? ", зургийг дараагийн сургалтад ашиглана" : ""}.</p>;

  if (!open)
    return (
      <button type="button" onClick={() => setOpen(true)} className="text-sm text-zinc-400 underline-offset-4 transition hover:text-zinc-900 hover:underline">
        Хариу буруу байна уу?
      </button>
    );

  const send = async () => {
    setState("sending");
    const corrections: Record<string, unknown> = { species: profile.species.key, wrong: [] as string[] };
    if (breed) {
      corrections.breed = breed;
      (corrections.wrong as string[]).push("breed");
    }
    if (noMarks) {
      corrections.marks = [];
      (corrections.wrong as string[]).push("marks");
    }
    try {
      await api.feedback(files, corrections, consent);
      setState("sent");
    } catch {
      setState("error");
    }
  };

  return (
    <div className="fade-in space-y-3 rounded-2xl border border-zinc-200 bg-white p-4 text-sm">
      <p className="font-medium">Юу буруу байсан бэ?</p>
      <label className="block">
        <span className="text-xs text-zinc-500">Зөв үүлдэр</span>
        <select className="input mt-1" value={breed} onChange={(e) => setBreed(e.target.value)}>
          <option value="">— мэдэхгүй / өөрчлөхгүй —</option>
          {breeds.map((b) => (
            <option key={b.key} value={b.key}>{b.mn}</option>
          ))}
        </select>
      </label>
      {profile.marks.length > 0 && (
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={noMarks} onChange={(e) => setNoMarks(e.target.checked)} />
          Онцгой шинж буруу ({profile.marks.map((m) => m.mn).join(", ")})
        </label>
      )}
      <label className="flex items-start gap-2 text-zinc-600">
        <input type="checkbox" className="mt-1" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        <span>Энэ зургийг загварыг сайжруулахад ашиглахыг зөвшөөрч байна. Хүссэн үедээ устгуулж болно.</span>
      </label>
      <div className="flex items-center gap-2">
        <button className="btn" disabled={state === "sending" || (!breed && !noMarks)} onClick={send}>
          {state === "sending" ? "Илгээж байна…" : "Илгээх"}
        </button>
        <button className="btn-ghost" onClick={() => setOpen(false)}>Болих</button>
        {state === "error" && <span className="text-xs text-zinc-500">Илгээж чадсангүй, дахин оролдоно уу.</span>}
      </div>
    </div>
  );
}
