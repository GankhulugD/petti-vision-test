import Link from "next/link";
import { STATUS_MN, type MatchResult } from "@/lib/api";

function tone(p: number) {
  if (p >= 75) return "bg-green-600";
  if (p >= 50) return "bg-amber-500";
  return "bg-stone-400";
}

export default function MatchList({ results }: { results: MatchResult[] }) {
  if (results.length === 0) return <p className="text-stone-500">Тохирох амьтан олдсонгүй.</p>;
  return (
    <ol className="space-y-3">
      {results.map((m, i) => (
        <li key={m.pet.id} className="card flex flex-col gap-4 sm:flex-row">
          <Link href={`/pets/${m.pet.id}`} className="relative h-36 w-full shrink-0 overflow-hidden rounded-xl bg-stone-100 sm:w-36">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={m.pet.photos[0]} alt="" className="h-full w-full object-cover" />
            <span className="absolute left-2 top-2 rounded-full bg-black/60 px-2 py-0.5 text-xs font-semibold text-white">#{i + 1}</span>
          </Link>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Link href={`/pets/${m.pet.id}`} className="text-lg font-semibold hover:underline">
                {m.pet.name || m.pet.profile.breed?.looks_like[0]?.mn || m.pet.profile.species.mn}
              </Link>
              <span className={`rounded-full px-3 py-1 text-sm font-bold text-white ${tone(m.percent)}`}>{m.percent}% тохирол</span>
            </div>
            <p className="mt-0.5 text-sm text-stone-500">
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
                <li key={r} className="text-green-800">✓ {r}</li>
              ))}
              {m.mismatches.slice(0, 2).map((r) => (
                <li key={r} className="text-red-700">✗ {r}</li>
              ))}
            </ul>
          </div>
        </li>
      ))}
    </ol>
  );
}
