import type { Level } from "@/lib/api";

/** Calibrated certainty bands from the model (not raw softmax %). */
export const LEVEL_META: Record<
  Level,
  { label: string; short: string; hint: string; segments: number; tone: string }
> = {
  high: {
    label: "Итгэлтэй",
    short: "Өндөр",
    hint: "Зургийн нөхцөл сайн, хариу нэг чиглэлд төвлөрсөн.",
    segments: 3,
    tone: "bg-emerald-600 text-emerald-950 ring-emerald-600/20",
  },
  medium: {
    label: "Магадгүй",
    short: "Дунд",
    hint: "Хэд хэдэн сонголт ойролцоо — нэмэлт зураг эсвэл өөр өнцөг тусална.",
    segments: 2,
    tone: "bg-amber-100 text-amber-950 ring-amber-500/25",
  },
  low: {
    label: "Тодорхойгүй",
    short: "Бага",
    hint: "Зураг бүдэг, хэсэгчлэн далдсан эсвэл үүлдэр манай сургалтад байхгүй байж болно.",
    segments: 1,
    tone: "bg-zinc-100 text-zinc-600 ring-zinc-300/40",
  },
};

export function ConfidenceBadge({ level, percent }: { level?: Level; percent?: number }) {
  if (!level) return null;
  const m = LEVEL_META[level];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${m.tone}`}
      title={m.hint}
    >
      <SignalDots segments={m.segments} active className="opacity-90" />
      {m.label}
      {percent !== undefined && <span className="font-mono tabular-nums opacity-80">{Math.round(percent * 100)}%</span>}
    </span>
  );
}

export function SignalDots({
  segments,
  active = true,
  className = "",
}: {
  segments: number;
  active?: boolean;
  className?: string;
}) {
  return (
    <span className={`inline-flex gap-[2px] ${className}`} aria-hidden>
      {[0, 1, 2].map((i) => (
        <i
          key={i}
          className={`h-1.5 w-1 rounded-sm ${i < segments ? (active ? "bg-current" : "bg-zinc-800") : "bg-current/25"}`}
        />
      ))}
    </span>
  );
}

export function AttributeConfidence({ level }: { level?: Level }) {
  if (!level) return null;
  const m = LEVEL_META[level];
  return (
    <p className="mt-1 flex items-start gap-2 text-[11px] leading-4 text-zinc-500">
      <SignalDots segments={m.segments} className="mt-0.5 shrink-0 text-zinc-700" />
      <span>
        <span className="font-medium text-zinc-600">{m.label}</span>
        <span className="text-zinc-400"> · </span>
        {m.hint}
      </span>
    </p>
  );
}

type Vote = { zero_shot: number; head: number | null; final: number };

const SIGNALS: { key: keyof Vote; label: string; desc: string; bar: string; show: (v: Vote) => boolean }[] = [
  {
    key: "zero_shot",
    label: "Тексттэй харьцуулалт",
    desc: "SigLIP — зургийг үүлдрийн тайлбартай тааруулна",
    bar: "bg-zinc-300",
    show: () => true,
  },
  {
    key: "head",
    label: "Сургасан толгой",
    desc: "Манай өгөгдөл дээр сургасан жижиг классификатор",
    bar: "bg-zinc-500",
    show: (v) => v.head !== null,
  },
  {
    key: "final",
    label: "Нэгтгэсэн үр дүн",
    desc: "Дээрх хоёрыг (эсвэл зөвхөн текстийг) нэгтгэсэн эцсийн санал",
    bar: "bg-zinc-900",
    show: () => true,
  },
];

export function BreedVoteSignals({ vote, scale = 1 }: { vote: Vote; scale?: number }) {
  return (
    <ul className="space-y-2.5">
      {SIGNALS.filter((s) => s.show(vote)).map((s) => {
        const raw = vote[s.key];
        const val = typeof raw === "number" ? raw : null;
        return (
          <li key={s.key}>
            <div className="flex items-baseline justify-between gap-2 text-xs">
              <div className="min-w-0">
                <span className="font-medium text-zinc-800">{s.label}</span>
                <span className="mt-0.5 block truncate text-[11px] text-zinc-400">{s.desc}</span>
              </div>
              <span className="shrink-0 font-mono tabular-nums text-zinc-600">
                {val === null ? "—" : `${Math.round(val * 100 * scale)}%`}
              </span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-zinc-100">
              {val !== null && (
                <div
                  className={`h-full rounded-full ${s.bar} transition-[width] duration-700`}
                  style={{ width: `${Math.max(2, val * 100 * scale)}%` }}
                />
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
