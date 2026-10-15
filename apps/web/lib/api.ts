// Typed client for the Petti Vision FastAPI server (proxied under /api).

export type Labeled = { key: string; en: string; mn: string; confidence: number };
export type Dist = Record<string, number>;

export type Profile = {
  species: Labeled;
  species_probs: Dist;
  breed: { top: Labeled[]; is_mixed: boolean; looks_like: Labeled[]; probs: Dist } | null;
  size: Labeled & { probs: Dist };
  age: Labeled & { probs: Dist };
  pattern: Labeled & { probs: Dist };
  coat: Labeled & { probs: Dist };
  colors: { key: string; en: string; mn: string; share: number }[];
  color_probs: Dist;
  marks: Labeled[];
  mark_probs: Dist;
  category: string;
  description_mn: string;
  description_en: string;
  detection: { found: boolean; box: number[] | null; confidence: number };
  quality: { warnings: string[]; photos: number };
  timing_ms?: Record<string, number>;
};

export type Pet = {
  id: string;
  name: string | null;
  status: "home" | "lost" | "found" | "reunited";
  species: string;
  category: string;
  owner_name: string | null;
  contact: string | null;
  location: string | null;
  event_at: string | null;
  notes: string | null;
  profile: Profile;
  created_at: number;
  photos: string[];
};

export type MatchResult = {
  pet: Pet;
  score: number;
  percent: number;
  components: { identity: number; appearance: number; attributes: number };
  reasons: string[];
  mismatches: string[];
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

/** Downscale on the client: phone photos are 3-12 MB, the model needs ~1024px. */
export async function shrink(file: File, maxSide = 1024, quality = 0.88): Promise<Blob> {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (s === 1 && file.size < 1_500_000) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * s);
    canvas.height = Math.round(bmp.height * s);
    canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    return await new Promise((ok) => canvas.toBlob((b) => ok(b ?? file), "image/jpeg", quality));
  } catch {
    return file; // e.g. HEIC on unsupported browsers → let the server handle it
  }
}

async function form(files: File[], fields: Record<string, string | undefined> = {}) {
  const fd = new FormData();
  const blobs = await Promise.all(files.map((f) => shrink(f)));
  blobs.forEach((b, i) => fd.append("images", b, files[i].name.replace(/\.\w+$/, "") + ".jpg"));
  Object.entries(fields).forEach(([k, v]) => v !== undefined && v !== "" && fd.append(k, v));
  return fd;
}

export const api = {
  health: () => call<{ status: string; models: Record<string, unknown>; pets: Record<string, number>; min_photos: number }>("/health"),
  analyze: async (files: File[]) => call<Profile>("/analyze", { method: "POST", body: await form(files) }),
  search: async (files: File[], statuses = "lost", top_k = 10) =>
    call<{ query: Profile; searched: number; results: MatchResult[] }>("/search", {
      method: "POST",
      body: await form(files, { statuses, top_k: String(top_k) }),
    }),
  createPet: async (files: File[], fields: Record<string, string | undefined>) =>
    call<{ pet: Pet; matches?: MatchResult[] }>("/pets", { method: "POST", body: await form(files, fields) }),
  pets: (status?: string) => call<Pet[]>(`/pets${status ? `?status=${status}` : ""}`),
  pet: (id: string) => call<Pet>(`/pets/${id}`),
  matches: (id: string) => call<{ pet: Pet; results: MatchResult[] }>(`/pets/${id}/matches`),
  setStatus: (id: string, status: string, location?: string) =>
    call<{ pet: Pet; matches?: MatchResult[] }>(`/pets/${id}/status`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ status, location, event_at: new Date().toISOString() }),
    }),
};

export const STATUS_MN: Record<Pet["status"], string> = {
  home: "Гэртээ",
  lost: "Алга болсон",
  found: "Олдсон (эзэн хайж байна)",
  reunited: "Эзэндээ очсон",
};

export type Opt = { key: string; mn: string; en: string; species?: string | string[]; size?: string };
export type Taxonomy = {
  species: Opt[]; breeds: Opt[]; size: Opt[]; age: Opt[]; colors: Opt[]; pattern: Opt[]; coat: Opt[]; marks: Opt[];
};
export const getTaxonomy = () => call<Taxonomy>("/taxonomy");
