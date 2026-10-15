// Petti Vision accuracy/latency benchmark (run in the browser console of the site):
//   const m = await import('/test/bench.js'); await m.run(3); m.summary()
// Deterministic image set: dog.ceo (Stanford Dogs) + Wikimedia Commons cat breeds.
const DOGS = {labrador:'labrador_retriever',husky:'siberian_husky',malamute:'alaskan_malamute',samoyed:'samoyed','retriever/golden':'golden_retriever',rottweiler:'rottweiler',doberman:'doberman',pug:'pug',chihuahua:'chihuahua',beagle:'beagle','poodle/toy':'toy_poodle',pomeranian:'pomeranian',shihtzu:'shih_tzu','corgi/cardigan':'welsh_corgi',dalmatian:'dalmatian',chow:'chow_chow',shiba:'shiba_inu','bulldog/french':'french_bulldog',boxer:'boxer',dachshund:'dachshund','collie/border':'border_collie',maltese:'maltese','mastiff/tibetan':'tibetan_mastiff',stbernard:'saint_bernard','terrier/yorkshire':'yorkshire_terrier',akita:'akita',papillon:'papillon','german/shepherd':'german_shepherd','spaniel/cocker':'cocker_spaniel',malinois:'belgian_malinois',pekinese:'pekingese','schnauzer/miniature':'miniature_schnauzer',weimaraner:'weimaraner'};
const CATS = {'Bengal cats':'bengal','Persian cats':'persian','Siamese cats':'siamese','Maine Coon cats':'maine_coon','Ragdoll cats':'ragdoll','Sphynx (cat)':'sphynx','Russian Blue cats':'russian_blue','British Shorthair cats':'british_shorthair','Abyssinian cats':'abyssinian','Scottish Fold cats':'scottish_fold','Bombay cats':'bombay','Birman cats':'birman','Siberian cats':'siberian_cat','Devon Rex':'devon_rex','Exotic Shorthair cats':'exotic_shorthair'};
const BAD = /GCCF|stamp|dna|HCM|radiograph|Ticking|mnhn|Tails|Museum|show|Exposition|Wystawa|chart|map/i;
const SIZE = {labrador_retriever:'large',siberian_husky:'medium',alaskan_malamute:'large',samoyed:'medium',golden_retriever:'large',rottweiler:'large',doberman:'large',pug:'small',chihuahua:'toy',beagle:'small',toy_poodle:'toy',pomeranian:'toy',shih_tzu:'toy',welsh_corgi:'small',dalmatian:'large',chow_chow:'medium',shiba_inu:'small',french_bulldog:'small',boxer:'large',dachshund:'small',border_collie:'medium',maltese:'toy',tibetan_mastiff:'giant',saint_bernard:'giant',yorkshire_terrier:'toy',akita:'large',papillon:'toy',german_shepherd:'large',cocker_spaniel:'small',belgian_malinois:'large',pekingese:'toy',miniature_schnauzer:'small',weimaraner:'large'};
const S = ['toy','small','medium','large','giant'];

export const state = (window.__B = window.__B || {jobs: [], res: [], done: false});

export async function jobs(perBreed = 3) {
  const out = [];
  for (const [k, v] of Object.entries(DOGS)) {
    try {
      const j = await (await fetch(`https://dog.ceo/api/breed/${k}/images`)).json();
      if (!Array.isArray(j.message)) continue;
      const L = j.message.filter(u => u.endsWith('.jpg'));
      for (let i = 0; i < perBreed; i++) out.push({url: L[Math.floor(i * L.length / perBreed)], species: 'dog', breed: v});
    } catch (e) {}
  }
  for (const [c, v] of Object.entries(CATS)) {
    try {
      const u = `https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:${encodeURIComponent(c)}&gcmtype=file&gcmlimit=30&prop=imageinfo&iiprop=url|mime&iiurlwidth=640&format=json&origin=*`;
      const j = await (await fetch(u)).json();
      Object.values(j.query?.pages || {}).sort((a, b) => (a.title < b.title ? -1 : 1))
        .filter(p => p.imageinfo?.[0]?.mime === 'image/jpeg' && !BAD.test(p.title)).slice(0, perBreed)
        .forEach(p => out.push({url: p.imageinfo[0].thumburl, species: 'cat', breed: v, title: p.title}));
    } catch (e) {}
  }
  return out;
}

export async function run(perBreed = 3, api = '/api', concurrency = 2) {
  state.jobs = await jobs(perBreed); state.res = []; state.done = false;
  const q = [...state.jobs];
  await Promise.all([...Array(concurrency)].map(async () => {
    while (q.length) {
      const job = q.shift();
      try {
        const b = await (await fetch(job.url)).blob();
        const fd = new FormData(); fd.append('images', b, 'x.jpg');
        const t0 = performance.now();
        const r = await fetch(`${api}/analyze`, {method: 'POST', body: fd});
        state.res.push({...job, ms: Math.round(performance.now() - t0), status: r.status, p: await r.json()});
      } catch (e) { state.res.push({...job, err: String(e)}); }
    }
  }));
  state.done = true;
  return summary();
}

export function summary() {
  const R = state.res.filter(r => r.status === 200);
  const out = {n: R.length, errors: state.res.length - R.length};
  for (const sp of ['dog', 'cat']) {
    const X = R.filter(r => r.species === sp);
    const a = {n: X.length, species: 0, top1: 0, top3: 0, size: 0, size_pm1: 0, marks_per_img: 0, by_level: {high: [0, 0], medium: [0, 0], low: [0, 0]}, errors: []};
    for (const r of X) {
      const p = r.p; if (p.species.key === sp) a.species++;
      const keys = (p.breed?.top || []).map(x => x.key); const i = keys.indexOf(r.breed);
      if (i === 0) a.top1++; else a.errors.push(`${r.breed}→${keys[0] || p.species.key}`);
      if (i >= 0 && i < 3) a.top3++;
      a.marks_per_img += p.marks.length;
      const L = p.breed?.level; if (L) { a.by_level[L][0]++; if (i === 0) a.by_level[L][1]++; }
      if (sp === 'dog' && SIZE[r.breed]) { if (p.size.key === SIZE[r.breed]) a.size++; if (Math.abs(S.indexOf(p.size.key) - S.indexOf(SIZE[r.breed])) <= 1) a.size_pm1++; }
    }
    for (const k of ['species', 'top1', 'top3', 'size', 'size_pm1', 'marks_per_img']) a[k] = +(a[k] / Math.max(a.n, 1)).toFixed(3);
    out[sp] = a;
  }
  const ms = R.map(r => r.p.timing_ms.total).sort((x, y) => x - y);
  out.server_ms = {p50: ms[ms.length >> 1], p90: ms[Math.floor(ms.length * 0.9)]};
  out.marks = R.flatMap(r => r.p.marks.map(m => m.key)).reduce((m, k) => ((m[k] = (m[k] || 0) + 1), m), {});
  out.dog_patterns = R.filter(r => r.species === 'dog').reduce((m, r) => ((m[r.p.pattern.key] = (m[r.p.pattern.key] || 0) + 1), m), {});
  return out;
}
