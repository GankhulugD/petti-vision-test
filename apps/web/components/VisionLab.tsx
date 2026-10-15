"use client";

// The analyzer: photo stage + an LLM-style live trace of what the model did,
// then the answer streams in. All content is the real server output
// (profile + profile.explain); only its reveal is animated.

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { Level, Profile } from "@/lib/api";
import { SWATCH } from "@/components/ProfileView";
import { CH, easeOut, prog, useClock } from "@/lib/stream";

const SPECIES_MN: Record<string, string> = {
  dog: "нохой", cat: "муур", bird: "шувуу", horse: "морь", sheep: "хонь", cow: "үхэр", rabbit: "туулай", other: "амьтан",
};
const LEVEL_MN: Record<Level, string> = { high: "итгэлтэй", medium: "магадгүй", low: "тодорхойгүй" };
const GAP = 110;
const MAX_T = Number.MAX_SAFE_INTEGER;

const pct = (x: number) => `${Math.round(x * 100)}%`;
const ms = (x?: number) => (x === undefined ? "" : `${Math.round(x)} ms`);
function short(s?: string) {
  if (!s) return "";
  return s
    .split("/")
    .pop()!
    .replace(/\.pt$/, "")
    .replace(/^dinov2-/, "DINOv2-")
    .replace(/^siglip2-base-patch16-224$/, "SigLIP2")
    .replace(/^yolo/, "YOLO");
}

// ------------------------------------------------------------------------- //
// timeline
// ------------------------------------------------------------------------- //
type Mark = "box" | "mask" | "attn" | "votes";
type Line = { text: string; meta?: string; mark?: Mark; at: number; end: number };
type Row = { label: string; value: string; level?: Level; swatches?: string[]; at: number };

function buildPlan(p: Profile, nFiles: number, searched?: number) {
  const ex = p.explain;
  const tm = p.timing_ms ?? {};
  const det = ex && ex.chosen !== null && ex.chosen !== undefined ? ex.detections[ex.chosen] : null;
  const raw: Omit<Line, "at" | "end">[] = [];

  raw.push({
    text: nFiles > 1 && ex ? `${nFiles} зураг уншлаа · ${ex.photo_index + 1}-р зураг хамгийн тод` : `${nFiles} зураг уншлаа`,
    meta: ex ? `${ex.image.w}×${ex.image.h}` : undefined,
  });
  raw.push(
    det
      ? { text: `Амьтан илрүүллээ — итгэл ${pct(det.confidence)}`, meta: `${short(ex?.models.detector)} · ${ms(tm.detect)}`, mark: "box" }
      : { text: "Амьтан тод илрээгүй — бүтэн зургаар үргэлжлүүлэв", meta: ms(tm.detect), mark: "box" },
  );
  if (ex?.polygon) raw.push({ text: "Дүрсийг арын дэвсгэрээс салгалаа", meta: "сегментчлэл", mark: "mask" });
  raw.push({ text: "Төрх, хээ угалзыг нь тоон хэлбэрт оруулав", meta: `${short(ex?.models.identity) || "DINOv2"} · ${ms(tm.embed)}`, mark: "attn" });
  const v = ex?.breed_votes?.[0];
  if (v) {
    raw.push({ text: `Үүлдрийн тайлбаруудтай харьцуулав → ${v.mn} ${pct(v.zero_shot)}`, meta: `${short(ex?.models.vision_language)} · zero-shot`, mark: "votes" });
    if (v.head !== null) raw.push({ text: `Сургасан модель дүгнэв → ${v.mn} ${pct(v.head)}`, meta: "MLP" });
  }
  raw.push({ text: "Нас, хэмжээ, өнгө, онцгой шинжийг тооцоолов", meta: ms(tm.reason) });
  if (searched !== undefined) raw.push({ text: `${searched} бүртгэлтэй төрх болон шинжээр нь тулгалаа`, meta: ms(tm.match) });

  let t = 200;
  const lines: Line[] = raw.map((r) => {
    const at = t;
    const end = at + r.text.length * CH;
    t = end + GAP;
    return { ...r, at, end };
  });
  const marks: Partial<Record<Mark, number>> = {};
  lines.forEach((l) => l.mark && (marks[l.mark] ??= l.at));
  const traceEnd = t;

  // ---- the answer ------------------------------------------------------- //
  let a = traceEnd + 200;
  const answerAt = a;
  const noAnimal = p.quality.no_animal;
  const eyebrow = noAnimal ? "Амьтан илрээгүй" : `${p.species.mn} · ${pct(p.species.confidence)}`;
  a += eyebrow.length * CH + 60;
  const title = noAnimal ? "Зурган дээр нохой, муур алга" : (p.breed?.top[0]?.mn ?? p.species.mn);
  const titleConf = noAnimal ? -1 : (p.breed?.top[0]?.confidence ?? p.species.confidence);
  const titleAt = a;
  a += title.length * 16 + 80;
  const sub = p.breed
    ? p.breed.is_mixed
      ? `Эрлийз байж магадгүй · ${p.breed.looks_like.slice(0, 3).map((b) => `${b.mn} ${pct(b.confidence)}`).join(" · ")}`
      : `Үүлдэр · ${LEVEL_MN[p.breed.level ?? "medium"]}`
    : "";
  const subAt = a;
  a += sub.length * CH + 80;
  const rowsRaw: Omit<Row, "at">[] = noAnimal ? [] : [
    { label: "Хэмжээ", value: p.size.mn, level: p.size.level },
    { label: "Нас", value: p.age.mn, level: p.age.level },
    { label: "Өнгө", value: p.colors.map((c) => c.mn).join(", ") || "—", swatches: p.colors.map((c) => SWATCH[c.key] ?? "#999") },
    { label: "Зүс", value: p.pattern.mn },
    { label: "Үс", value: p.coat.mn },
    { label: "Онцгой шинж", value: p.marks.map((m) => m.mn).join(", ") || "Илрээгүй" },
  ];
  const rows: Row[] = rowsRaw.map((r) => {
    const at = a;
    a += r.value.length * CH + 50;
    return { ...r, at };
  });
  const descAt = a + 80;
  const words = p.description_mn.split(/(\s+)/);
  a = descAt + Math.ceil(words.length / 2) * 22;
  return { lines, marks, traceEnd, answerAt, eyebrow, title, titleConf, titleAt, sub, subAt, rows, descAt, end: a + 400 };
}

// ------------------------------------------------------------------------- //
// small pieces
// ------------------------------------------------------------------------- //
/** Types `text` from time `at`; the untyped rest is reserved (invisible) so nothing reflows. */
function Typed({ text, t, at, speed = CH, className = "" }: { text: string; t: number; at: number; speed?: number; className?: string }) {
  const n = t <= at ? 0 : Math.min(text.length, Math.ceil((t - at) / speed));
  return (
    <span className={className}>
      {text.slice(0, n)}
      {n > 0 && n < text.length && <span className="caret" />}
      <span className="invisible">{text.slice(n)}</span>
    </span>
  );
}

/** Word-chunk streaming, like an LLM answer. */
function Streamed({ text, t, at }: { text: string; t: number; at: number }) {
  const toks = text.split(/(\s+)/);
  const n = t <= at ? 0 : Math.min(toks.length, Math.ceil((t - at) / 11));
  return (
    <>
      {toks.slice(0, n).join("")}
      {n > 0 && n < toks.length && <span className="caret" />}
      <span className="invisible">{toks.slice(n).join("")}</span>
    </>
  );
}

function Dots({ level }: { level?: Level }) {
  if (!level) return null;
  const n = { high: 3, medium: 2, low: 1 }[level];
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400" title={LEVEL_MN[level]}>
      <span className="inline-flex gap-[3px]">
        {[0, 1, 2].map((i) => (
          <i key={i} className={`h-1 w-1 rounded-full ${i < n ? "bg-zinc-800" : "bg-zinc-300"}`} />
        ))}
      </span>
      {LEVEL_MN[level]}
    </span>
  );
}

function Check() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 text-zinc-400" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3.5 8.5l3 3 6-7" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 animate-spin text-zinc-900">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeOpacity={0.15} strokeWidth={2} />
      <path d="M14 8a6 6 0 0 0-6-6" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
}

function Corners({ box }: { box: number[] }) {
  const [x1, y1, x2, y2] = box;
  const L = Math.min(x2 - x1, y2 - y1) * 0.12;
  const d = `M${x1},${y1 + L}V${y1}H${x1 + L}M${x2 - L},${y1}H${x2}V${y1 + L}M${x2},${y2 - L}V${y2}H${x2 - L}M${x1 + L},${y2}H${x1}V${y2 - L}`;
  return <path d={d} fill="none" stroke="white" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />;
}

// ------------------------------------------------------------------------- //
// main
// ------------------------------------------------------------------------- //
type Props = {
  files: File[];
  profile: Profile | null;
  busy: boolean;
  /** enables the drop zone + filmstrip (analyze page) */
  onFiles?: (files: File[]) => void;
  /** search mode: number of registrations compared */
  searched?: number;
  compact?: boolean;
  maxFiles?: number;
};

export default function VisionLab({ files, profile, busy, onFiles, searched, compact = false, maxFiles = 6 }: Props) {
  const ex = profile?.explain;
  const [urls, setUrls] = useState<string[]>([]);
  const [layers, setLayers] = useState({ mask: true });
  const [drag, setDrag] = useState(false);
  const [skipped, setSkipped] = useState<Profile | null>(null);
  const [traceOpen, setTraceOpen] = useState<boolean | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const maskId = "m" + useId().replace(/[^a-zA-Z0-9]/g, "");

  useEffect(() => {
    const u = files.map((f) => URL.createObjectURL(f));
    setUrls(u);
    return () => u.forEach(URL.revokeObjectURL);
  }, [files]);
  useEffect(() => setTraceOpen(null), [profile]);

  const plan = useMemo(() => (profile ? buildPlan(profile, files.length || 1, searched) : null), [profile, files.length, searched]);
  const streaming = !!plan && !busy;
  const clock = useClock(streaming, profile, plan?.end ?? 0);
  const t = skipped === profile && profile ? MAX_T : clock;
  const waitT = useClock(busy, files, 20000);
  const done = !!plan && t >= plan.end;

  const add = (list: FileList | null) => {
    if (!list || !onFiles) return;
    const imgs = Array.from(list).filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
    if (imgs.length) onFiles([...files, ...imgs].slice(0, maxFiles));
  };
  const dropProps = onFiles
    ? {
        onDragOver: (e: React.DragEvent) => {
          e.preventDefault();
          setDrag(true);
        },
        onDragLeave: () => setDrag(false),
        onDrop: (e: React.DragEvent) => {
          e.preventDefault();
          setDrag(false);
          add(e.dataTransfer.files);
        },
      }
    : {};

  const idx = ex && !busy ? ex.photo_index : 0;
  const url = urls[idx] ?? urls[0];
  const W = ex?.image.w ?? 4;
  const H = ex?.image.h ?? 3;
  const [ratio, setRatio] = useState(4 / 3);
  const r = ex && !busy ? W / H : ratio;
  const det = ex && ex.chosen !== null && ex.chosen !== undefined ? ex.detections[ex.chosen] : null;
  const poly = ex?.polygon?.map((p) => p.join(",")).join(" ");
  const on = (m: Mark) => !!plan && streaming && plan.marks[m] !== undefined && t >= plan.marks[m]!;
  const boxOn = on("box");
  const maskOn = on("mask") && layers.mask;

  const waitMsg = waitT < 500 ? "Зураг илгээж байна" : waitT < 1100 ? "Амьтан хайж байна" : waitT < 1800 ? "Дүрсийг салгаж байна" : "Шинжийг тооцоолж байна";

  // ---------------------------------------------------------------- stage
  const stage = (
    <div className="space-y-3">
      <div
        {...dropProps}
        className={`relative overflow-hidden rounded-2xl bg-zinc-100 ring-1 transition ${drag ? "ring-2 ring-zinc-900" : "ring-zinc-200/80"}`}
      >
        {url ? (
          <div className="relative mx-auto" style={{ width: `min(100%, ${Math.round(560 * r)}px)`, aspectRatio: `${r}` }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={url}
              alt=""
              onLoad={(e) => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight || 4 / 3)}
              className="absolute inset-0 h-full w-full object-cover transition-[filter,transform] duration-700"
              style={{ filter: busy ? "grayscale(0.6) brightness(0.92)" : "none", transform: busy ? "scale(1.015)" : "none" }}
            />
            {busy && <div className="sweep" />}
            {ex && streaming && (
              <>
                <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
                  <defs>
                    <mask id={maskId}>
                      <rect width={W} height={H} fill="white" />
                      {poly ? (
                        <polygon points={poly} fill="black" />
                      ) : (
                        det && <rect x={det.box[0]} y={det.box[1]} width={det.box[2] - det.box[0]} height={det.box[3] - det.box[1]} fill="black" />
                      )}
                    </mask>
                  </defs>
                  <rect
                    width={W}
                    height={H}
                    fill="rgb(9 9 11)"
                    mask={`url(#${maskId})`}
                    style={{ opacity: maskOn ? 0.5 : boxOn && det ? 0.18 : 0, transition: "opacity 600ms cubic-bezier(.2,.8,.2,1)" }}
                  />
                  {maskOn && poly && (
                    <polygon
                      key={String(profile?.timing_ms?.total)}
                      points={poly}
                      pathLength={1}
                      fill="none"
                      stroke="white"
                      strokeWidth={1.5}
                      strokeOpacity={0.9}
                      vectorEffect="non-scaling-stroke"
                      className="draw"
                    />
                  )}
                  {boxOn &&
                    ex.detections.map((d, i) =>
                      i === ex.chosen ? (
                        <g key={i} className="fade-in">
                          <Corners box={d.box} />
                        </g>
                      ) : (
                        <rect
                          key={i}
                          x={d.box[0]}
                          y={d.box[1]}
                          width={d.box[2] - d.box[0]}
                          height={d.box[3] - d.box[1]}
                          fill="none"
                          stroke="white"
                          strokeOpacity={0.45}
                          strokeDasharray="4 4"
                          strokeWidth={1}
                          vectorEffect="non-scaling-stroke"
                          className="fade-in"
                        />
                      ),
                    )}
                </svg>
                {boxOn && det && (
                  <div
                    className="fade-in absolute flex items-center gap-1.5 rounded-full bg-white/95 py-0.5 pl-1.5 pr-2 text-[11px] font-medium text-zinc-900 shadow-sm backdrop-blur"
                    style={{
                      left: `${(det.box[0] / W) * 100}%`,
                      top: `${(det.box[1] / H) * 100}%`,
                      transform: det.box[1] / H < 0.07 ? "translate(6px, 6px)" : "translate(0, calc(-100% - 6px))",
                    }}
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-brand-500" />
                    {profile?.species.mn ?? SPECIES_MN[det.species] ?? det.species}
                    <span className="font-mono text-zinc-400">{pct(det.confidence)}</span>
                  </div>
                )}
              </>
            )}
            {busy && (
              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full bg-white/90 px-3 py-1 text-xs shadow-sm backdrop-blur">
                <Spinner />
                <span className="shimmer font-medium">{waitMsg}</span>
              </div>
            )}
            {ex && done && poly && (
              <div className="fade-in absolute right-3 top-3 flex gap-1 rounded-full bg-black/40 p-0.5 text-[11px] text-white backdrop-blur-md">
                {poly && (
                  <button className={`rounded-full px-2.5 py-1 transition ${layers.mask ? "bg-white text-zinc-900" : "hover:bg-white/10"}`} onClick={() => setLayers((l) => ({ ...l, mask: !l.mask }))}>
                    Маск
                  </button>
                )}
              </div>
            )}
          </div>
        ) : (
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="group flex aspect-[4/3] w-full flex-col items-center justify-center gap-4 text-center"
          >
            <span className={`grid h-14 w-14 place-items-center rounded-2xl bg-white shadow-sm ring-1 ring-zinc-200 transition duration-300 group-hover:-translate-y-0.5 group-hover:shadow-md ${drag ? "scale-110" : ""}`}>
              <svg viewBox="0 0 24 24" className="h-6 w-6 text-zinc-700" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 16l4.5-4.5a2 2 0 0 1 2.8 0L16 16m-2-2 1.5-1.5a2 2 0 0 1 2.8 0L20 14" />
                <rect x="3" y="4" width="18" height="16" rx="3" />
                <circle cx="9" cy="9" r="1.5" />
              </svg>
            </span>
            <span>
              <span className="block text-[15px] font-medium text-zinc-900">{drag ? "Энд тавина уу" : "Зургаа чирж оруулах эсвэл сонгох"}</span>
              <span className="mt-1 block text-sm text-zinc-500">Нэг амьтны 1–{maxFiles} зураг, өөр өнцгөөс · JPG, PNG, HEIC</span>
            </span>
          </button>
        )}
        {onFiles && <input ref={input} type="file" accept="image/*" multiple className="hidden" onChange={(e) => { add(e.target.files); e.target.value = ""; }} />}
      </div>

      {onFiles && files.length > 0 && (
        <div className="flex items-center gap-2">
          {urls.map((u, i) => (
            <div
              key={u}
              className={`group relative h-14 w-14 shrink-0 overflow-hidden rounded-xl ring-1 transition ${i === idx && profile ? "ring-2 ring-zinc-900 ring-offset-2 ring-offset-[#fafafa]" : "ring-zinc-200"}`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={u} alt="" className={`h-full w-full object-cover transition ${busy ? "opacity-60" : ""}`} />
              <button
                type="button"
                aria-label="Устгах"
                onClick={() => onFiles(files.filter((_, j) => j !== i))}
                className="absolute inset-0 grid place-items-center bg-black/50 text-lg text-white opacity-0 transition group-hover:opacity-100"
              >
                ×
              </button>
            </div>
          ))}
          {files.length < maxFiles && (
            <button
              type="button"
              onClick={() => input.current?.click()}
              className="grid h-14 w-14 shrink-0 place-items-center rounded-xl border border-dashed border-zinc-300 text-xl text-zinc-400 transition hover:border-zinc-500 hover:text-zinc-700"
              aria-label="Зураг нэмэх"
            >
              +
            </button>
          )}
          <button type="button" onClick={() => onFiles([])} className="ml-auto text-sm text-zinc-400 transition hover:text-zinc-900">
            Цэвэрлэх
          </button>
        </div>
      )}
    </div>
  );

  // ---------------------------------------------------------------- trace
  const open = traceOpen ?? (!plan || t < plan.answerAt);
  const trace = (busy || plan) && (
    <div>
      <button
        type="button"
        onClick={() => (plan && !done && t < plan.answerAt ? setSkipped(profile) : setTraceOpen(!open))}
        className="flex w-full items-center gap-2 text-left text-sm"
      >
        {busy || (plan && t < plan.traceEnd) ? (
          <>
            <Spinner />
            <span className="shimmer font-medium">{busy ? waitMsg : "Модель бодож байна"}</span>
            {plan && <span className="ml-auto text-xs text-zinc-400 hover:text-zinc-700">Алгасах</span>}
          </>
        ) : (
          <>
            <Check />
            <span className="font-medium text-zinc-500">
              Шинжилсэн · <span className="font-mono text-xs">{ms(profile?.timing_ms?.total)}</span>
            </span>
            <svg viewBox="0 0 16 16" className={`ml-1 h-3.5 w-3.5 text-zinc-400 transition ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" strokeWidth={1.8}>
              <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </>
        )}
      </button>
      {plan && (
        <div className="expander" data-open={open}>
          <div>
            <ol className="ml-[7px] mt-3 space-y-2 border-l border-zinc-200 pb-1 pl-5">
              {plan.lines
                .filter((l) => t >= l.at)
                .map((l) => {
                  const active = t < l.end + GAP;
                  return (
                    <li key={l.text} className="fade-in relative flex items-baseline gap-3 text-[13px] leading-5">
                      <span className="absolute -left-[25px] top-[7px] grid h-[9px] w-[9px] place-items-center rounded-full bg-[#fafafa]">
                        <span className={`h-[5px] w-[5px] rounded-full ${active ? "pulse-dot bg-zinc-900" : "bg-zinc-300"}`} />
                      </span>
                      <Typed text={l.text} t={t} at={l.at} className={active ? "text-zinc-900" : "text-zinc-500"} />
                      {l.meta && (
                        <span className="ml-auto shrink-0 font-mono text-[11px] text-zinc-400 transition-opacity duration-300" style={{ opacity: t >= l.end ? 1 : 0 }}>
                          {l.meta}
                        </span>
                      )}
                    </li>
                  );
                })}
            </ol>
          </div>
        </div>
      )}
    </div>
  );

  // ---------------------------------------------------------------- answer
  const answer = plan && profile && t >= plan.answerAt && (
    <div className="space-y-5">
      <div>
        <p className="eyebrow">
          <Typed text={plan.eyebrow} t={t} at={plan.answerAt} />
        </p>
        <div className="mt-1.5 flex items-baseline justify-between gap-4">
          <h2 className={`${compact ? "text-2xl" : "text-[32px]"} font-semibold leading-tight tracking-tight`}>
            <Typed text={plan.title} t={t} at={plan.titleAt} speed={16} />
          </h2>
          {plan.titleConf >= 0 && (
            <span className="font-mono text-sm tabular-nums text-zinc-400" style={{ opacity: t >= plan.titleAt ? 1 : 0 }}>
              {Math.round(plan.titleConf * 100 * easeOut(prog(t, plan.titleAt, 700)))}%
            </span>
          )}
        </div>
        {plan.sub && (
          <p className="mt-1 text-sm text-zinc-500">
            <Typed text={plan.sub} t={t} at={plan.subAt} />
          </p>
        )}
      </div>

      <dl className="grid grid-cols-2 gap-x-6 border-t border-zinc-200/80">
        {plan.rows.map((row) => (
          <div key={row.label} className="border-b border-zinc-200/80 py-3 transition-opacity duration-300" style={{ opacity: t >= row.at ? 1 : 0.25 }}>
            <dt className="text-xs text-zinc-400">{row.label}</dt>
            <dd className="mt-0.5 flex items-center gap-1.5 text-sm font-medium text-zinc-900">
              {row.swatches?.map((c, i) => (
                <span key={i} className="h-2.5 w-2.5 shrink-0 rounded-full ring-1 ring-black/10" style={{ background: c, opacity: t >= row.at ? 1 : 0 }} />
              ))}
              <Typed text={row.value} t={t} at={row.at} />
            </dd>
            <div className="mt-0.5 h-4" style={{ opacity: t >= row.at + row.value.length * CH ? 1 : 0, transition: "opacity .3s" }}>
              <Dots level={row.level} />
            </div>
          </div>
        ))}
      </dl>

      <p className="text-[15px] leading-7 text-zinc-700" style={{ opacity: t >= plan.descAt ? 1 : 0 }}>
        <Streamed text={profile.description_mn} t={t} at={plan.descAt} />
      </p>

      {done && profile.quality.warnings.length > 0 && (
        <ul className="fade-in space-y-1.5 rounded-xl border border-zinc-200 bg-white p-3 text-sm text-zinc-600">
          {profile.quality.warnings.map((w) => (
            <li key={w} className="flex gap-2">
              <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500" />
              {w}
            </li>
          ))}
        </ul>
      )}
    </div>
  );

  // ---------------------------------------------------------------- insights
  const votes = ex?.breed_votes ?? [];
  const vAt = plan?.marks.votes ?? plan?.answerAt ?? 0;
  const fAt = plan?.marks.attn ?? 0;
  // the identity fingerprint only matters where we match against registrations
  const showPrint = searched !== undefined;
  const insights = plan && ex && streaming && (votes.length > 0 || showPrint) && t >= Math.min(vAt, showPrint ? fAt : vAt) && (
    <div className={`grid gap-4 ${!compact && showPrint && votes.length > 0 ? "md:grid-cols-[1.3fr_1fr]" : ""}`}>
      {votes.length > 0 && (
        <section className="card fade-in">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold">Үүлдрийг хэрхэн шийдсэн бэ</h3>
            <span className="eyebrow">санал</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-zinc-500">
            <span className="flex items-center gap-1.5"><i className="h-1.5 w-3 rounded-full bg-zinc-300" /> Zero-shot (зураг ↔ текст)</span>
            <span className="flex items-center gap-1.5"><i className="h-1.5 w-3 rounded-full bg-zinc-500" /> Сургасан модель</span>
            <span className="flex items-center gap-1.5"><i className="h-1.5 w-3 rounded-full bg-zinc-900" /> Эцсийн</span>
          </div>
          <ul className="mt-4 space-y-3.5">
            {votes.map((v, i) => {
              const k = (d: number) => easeOut(prog(t, vAt + i * 90 + d, 700));
              return (
                <li key={v.key}>
                  <div className="mb-1.5 flex items-baseline justify-between text-sm">
                    <span className={i === 0 ? "font-medium text-zinc-900" : "text-zinc-600"}>{v.mn}</span>
                    <span className="font-mono text-xs tabular-nums text-zinc-500">{Math.round(v.final * 100 * k(200))}%</span>
                  </div>
                  <div className="space-y-[3px]">
                    {([
                      [v.zero_shot, "bg-zinc-300", 0],
                      [v.head, "bg-zinc-500", 100],
                      [v.final, "bg-zinc-900", 200],
                    ] as const).map(([val, cls, d], j) => (
                      <div key={j} className="h-[3px] overflow-hidden rounded-full bg-zinc-100">
                        {val !== null && <div className={`h-full rounded-full ${cls}`} style={{ width: `${Math.max(1, val * 100) * k(d)}%` }} />}
                      </div>
                    ))}
                  </div>
                </li>
              );
            })}
          </ul>
          <p className="mt-4 text-xs leading-5 text-zinc-500">
            SigLIP2 зургийг үүлдэр бүрийн хэдэн арван тайлбартай харьцуулна. Манай өгөгдөл дээр сургасан жижиг модель өөрийн саналаа нэмж, бүх зургаар
            дундажлаад эцсийн хариу гарна.
          </p>
        </section>
      )}
      {showPrint && (
      <section className="card fade-in">
        <div className="flex items-baseline justify-between">
          <h3 className="text-sm font-semibold">Төрхийн хээ</h3>
          <span className="eyebrow">{short(ex.models.identity) || "DINOv2"} · 384</span>
        </div>
        <svg viewBox="0 0 480 96" className="mt-4 h-24 w-full" preserveAspectRatio="none">
          <line x1="0" x2="480" y1="48" y2="48" stroke="#e4e4e7" strokeDasharray="2 4" />
          {ex.fingerprint.map((v, i) => {
            const h = Math.max(2, Math.abs(v) * 84 * easeOut(prog(t, fAt + 150 + i * 16, 450)));
            return <rect key={i} x={i * 10 + 2} y={48 - h / 2} width={5} height={h} rx={2.5} fill={v >= 0 ? "#18181b" : "#a1a1aa"} />;
          })}
        </svg>
        <p className="mt-3 text-xs leading-5 text-zinc-500">
          Зургаас гаргасан 384 тоог 48 баганаар харууллаа. Нэг амьтны өөр зургууд бараг ижил хээ өгдөг тул бүртгэлтэй амьтадтай ингэж тулгадаг.
        </p>
      </section>
      )}
    </div>
  );

  // ---------------------------------------------------------------- idle explainer
  const idle = !busy && !plan && (
    <div>
      <p className="eyebrow">Хэрхэн ажилладаг вэ</p>
      <ol className="mt-4 space-y-4">
        {[
          ["Илрүүлэх", "Зурган дээрх амьтныг олж, дүрсийг нь дэвсгэрээс салгана", "YOLO26n-seg"],
          ["Төрхийг кодлох", "Амьтны төрх, хээ угалзыг тоон хэлбэрт оруулна", "DINOv2"],
          ["Шинж чанар", "Үүлдэр, нас, хэмжээ, өнгө, онцгой шинжийг тайлбартай харьцуулж, сургасан модельтой хослуулна", "SigLIP2 + MLP"],
          ["Тулгах", "Алга болсон, олдсон амьтдын бүртгэлтэй төрх болон шинжээр нь тулгана", "Matcher"],
        ].map(([title, text, model], i) => (
          <li key={title} className="flex gap-4">
            <span className="mt-0.5 font-mono text-xs text-zinc-300">0{i + 1}</span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm font-medium">{title}</span>
                <span className="font-mono text-[11px] text-zinc-400">{model}</span>
              </div>
              <p className="mt-0.5 text-sm text-zinc-500">{text}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );

  if (compact) {
    return (
      <div className="space-y-6">
        {stage}
        {trace}
        {answer}
        {insights}
      </div>
    );
  }
  return (
    <div className="space-y-10">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:gap-12">
        <div className="lg:sticky lg:top-20 lg:self-start">{stage}</div>
        <div className="space-y-8 lg:pt-1">
          {idle}
          {trace}
          {answer}
        </div>
      </div>
      {insights}
    </div>
  );
}
