import { ConfidenceBadge } from "@/components/Confidence";
import type { Level, Profile } from "@/lib/api";

function Bar({ value }: { value: number }) {
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-100">
      <div className="h-full rounded-full bg-zinc-900 transition-[width] duration-700" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

function LevelChip({ level }: { level?: Level }) {
  if (!level) return null;
  return <ConfidenceBadge level={level} />;
}

function Row({ label, value, conf, sub, level }: { label: string; value: string; conf?: number; sub?: string; level?: Level }) {
  return (
    <div className="grid grid-cols-[7.5rem_1fr] items-start gap-3 py-2.5">
      <div className="text-sm text-zinc-500">{label}</div>
      <div>
        <div className="flex items-baseline justify-between gap-2">
          <span className="flex items-center gap-2 font-semibold">
            {value} <LevelChip level={level} />
          </span>
          {conf !== undefined && <span className="text-xs tabular-nums text-zinc-500">{Math.round(conf * 100)}%</span>}
        </div>
        {conf !== undefined && <Bar value={conf} />}
        {sub && <p className="mt-1 text-xs text-zinc-500">{sub}</p>}
      </div>
    </div>
  );
}

export const SWATCH: Record<string, string> = {
  black: "#1f1f1f", white: "#f5f5f0", grey: "#80808a", brown: "#64391f", tan: "#aa7d50",
  golden: "#d2a550", red: "#c0602a", cream: "#e8d4a8", silver: "#b8b8c0",
};

export default function ProfileView({ p, compact = false }: { p: Profile; compact?: boolean }) {
  if (p.quality.no_animal) {
    return (
      <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-sm leading-relaxed">
        <p className="font-medium">{p.description_mn}</p>
        <p className="mt-2 text-zinc-500">{p.description_en}</p>
      </div>
    );
  }
  return (
    <div className="space-y-4">
      <div className="divide-y divide-zinc-100">
        <Row label="Төрөл" value={p.species.mn} conf={p.species.confidence} level={p.species.level} />
        {p.breed && (
          <div className="grid grid-cols-[7.5rem_1fr] items-start gap-3 py-2.5">
            <div className="text-sm text-zinc-500">Үүлдэр</div>
            <div className="space-y-1.5">
              <div className="flex flex-wrap gap-1.5">
                <LevelChip level={p.breed.level} />
                {p.breed.uncertain && <span className="chip">Энэ зургаар тодорхойгүй — нүүр харагдах зураг нэмнэ үү</span>}
                {p.breed.is_mixed && <span className="chip">Эрлийз эсвэл дараахаас аль нэг нь</span>}
              </div>
              {p.breed.top.slice(0, compact ? 3 : 5).map((b) => (
                <div key={b.key}>
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{b.mn}</span>
                    <span className="tabular-nums text-zinc-500">{Math.round(b.confidence * 100)}%</span>
                  </div>
                  <Bar value={b.confidence} />
                </div>
              ))}
            </div>
          </div>
        )}
        <Row label="Хэмжээ" value={p.size.mn} conf={p.size.confidence} level={p.size.level} />
        <Row label="Нас" value={p.age.mn} conf={p.age.confidence} level={p.age.level} />
        <div className="grid grid-cols-[7.5rem_1fr] items-center gap-3 py-2.5">
          <div className="text-sm text-zinc-500">Өнгө</div>
          <div className="flex flex-wrap gap-2">
            {p.colors.map((c) => (
              <span key={c.key} className="chip">
                <span className="h-3 w-3 rounded-full border border-zinc-300" style={{ background: SWATCH[c.key] }} />
                {c.mn} <span className="text-zinc-400">{Math.round(c.share * 100)}%</span>
              </span>
            ))}
          </div>
        </div>
        <Row label="Зүс" value={p.pattern.mn} conf={p.pattern.confidence} />
        <Row label="Үс" value={p.coat.mn} conf={p.coat.confidence} />
        <div className="grid grid-cols-[7.5rem_1fr] items-start gap-3 py-2.5">
          <div className="text-sm text-zinc-500">Онцгой шинж</div>
          <div className="flex flex-wrap gap-2">
            {p.marks.length === 0 && <span className="text-sm text-zinc-400">Илрээгүй</span>}
            {p.marks.map((m) => (
              <span key={m.key} className="chip">
                {m.mn} · {Math.round(m.confidence * 100)}%
              </span>
            ))}
          </div>
        </div>
      </div>

      {!compact && (
        <>
          <div className="rounded-xl bg-zinc-50 p-3 text-sm leading-relaxed">
            <p>{p.description_mn}</p>
            <p className="mt-2 text-zinc-500">{p.description_en}</p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-zinc-500">
            <span className="chip">Ангилал: {p.category}</span>
            {p.timing_ms && <span className="chip">⏱ {p.timing_ms.total} ms ({p.timing_ms.photos} зураг)</span>}
          </div>
        </>
      )}
      {p.quality.warnings.length > 0 && (
        <ul className="space-y-1 rounded-xl border border-zinc-200 bg-white p-3 text-sm text-zinc-600">
          {p.quality.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
