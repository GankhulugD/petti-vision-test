"use client";

import { useEffect, useRef, useState } from "react";

export const ANGLES = ["Урдаас (нүүр)", "Зүүн хажуу", "Баруун хажуу", "Ар талаас", "Дээрээс / бүтэн бие", "Онцгой шинж (сорви, толбо)"];

type Props = {
  files: File[];
  onChange: (files: File[]) => void;
  min?: number;
  max?: number;
  hints?: boolean;
};

export default function ImagePicker({ files, onChange, min = 1, max = 12, hints = false }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [urls, setUrls] = useState<string[]>([]);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    const u = files.map((f) => URL.createObjectURL(f));
    setUrls(u);
    return () => u.forEach(URL.revokeObjectURL);
  }, [files]);

  const add = (list: FileList | null) => {
    if (!list) return;
    const imgs = Array.from(list).filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
    onChange([...files, ...imgs].slice(0, max));
  };

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          add(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-8 text-center transition ${
          drag ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 bg-white hover:border-zinc-400"
        }`}
      >
        <p className="font-semibold">Зураг чирж оруулах эсвэл дарж сонгох</p>
        <p className="mt-1 text-sm text-zinc-500">
          {min > 1 ? `Дор хаяж ${min} зураг, өөр өөр өнцгөөс` : "1 ба түүнээс дээш зураг"} · JPG/PNG/HEIC
        </p>
        <input ref={input} type="file" accept="image/*" multiple capture={undefined} className="hidden" onChange={(e) => add(e.target.files)} />
      </div>

      {hints && (
        <div className="mt-3 flex flex-wrap gap-2">
          {ANGLES.map((a, i) => (
            <span key={a} className={`chip ${i < files.length ? "border-zinc-900 text-zinc-900" : "text-zinc-400"}`}>
              {i < files.length ? "✓" : i + 1}. {a}
            </span>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <div className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-6">
          {urls.map((u, i) => (
            <div key={u} className="group relative aspect-square overflow-hidden rounded-xl bg-zinc-100">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={u} alt="" className="h-full w-full object-cover" />
              <button
                type="button"
                onClick={() => onChange(files.filter((_, j) => j !== i))}
                className="absolute right-1 top-1 rounded-full bg-black/60 px-2 text-sm text-white opacity-80 hover:opacity-100"
                aria-label="Устгах"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      {min > 1 && (
        <p className={`mt-2 text-sm ${files.length >= min ? "text-zinc-900" : "text-zinc-500"}`}>
          {files.length}/{min} зураг {files.length >= min ? "— бэлэн" : ""}
        </p>
      )}
    </div>
  );
}
