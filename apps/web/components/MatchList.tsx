import Link from "next/link";
import { STATUS_MN, type MatchResult } from "@/lib/api";

function tone(p: number) {
  if (p >= 75) return "bg-zinc-900 text-white";
  if (p >= 50) return "bg-zinc-200 text-zinc-900";
  return "bg-zinc-100 text-zinc-500";
}

export default function MatchList({ results }: { results: MatchResult[] }) {
  if (results.length === 0) return <p className="text-zinc-500">Тохирох амьтан олдсонгүй.</p>;
  return (
    <ol className="space-y-3">
      {results.map((m, i) => (
        <li key={m.pet.id} className="card fade-in flex flex-col gap-4 sm:flex-row" style={{ animationDelay: `${i * 70}ms` }}>
          <Link href={`/pets/${m.pet.id}`} className="relative h-36 w-full shrink-0 overflow-hidden rounded-xl bg-zinc-100 sm:w-36">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={m.pet.photos[0]} alt="" className="h-full w-full object-cover" />
            <span className="absolute left-2 top-2 rounded-full bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">#{i + 1}</span>
          </Link>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Link href={`/pets/${m.pet.id}`} className="text-lg font-semibold hover:underline">
                {m.pet.name || m.pet.profile.breed?.looks_like[0]?.mn || m.pet.profile.species.mn}
              </Link>
              <span className={`rounded-full px-3 py-1 font-mono text-xs font-medium ${tone(m.percent)}`}>{m.percent}% тохирол</span>
            </div>
            <p className="mt-0.5 text-sm text-zinc-500">
              {STATUS_MN[m.pet.status]}
              {m.pet.location ? ` · ${m.pet.location}` : ""}
              {m.pet.contact ? ` · ☎ ${m.pet.contact}` : ""}
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
              <span className="chip">Төрх {Math.round(m.components.identity * 100)}%</span>
              <span className="chip">Харагдах байдал {Math.round(m.components.appearance * 100)}%</span>
              <span className="chip">Шинж тэмдэг {Math.round(m.components.attributes * 100)}%</span>
            </div>
            <ul className="mt-2 space-y-0.5 text-sm">
              {m.reasons.slice(0, 4).map((r) => (
                <li key={r} className="text-zinc-700">✓ {r}</li>
              ))}
              {m.mismatches.slice(0, 2).map((r) => (
                <li key={r} className="text-zinc-400">✗ {r}</li>
              ))}
            </ul>
          </div>
        </li>
      ))}
    </ol>
  );
}
