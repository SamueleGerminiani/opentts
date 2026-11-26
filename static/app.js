"use strict";

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const viewer = $("viewer");
const pagesEl = $("pages");

const SKIP_KINDS = ["figure", "table", "caption", "formula", "margin"];
const REGION_LABEL = { figure: "Figure · ignored", table: "Table · ignored", formula: "Formula · ignored" };
const SILENCE = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";

// ---------------------------------------------------------------- settings

const defaults = {
  speed: 1,
  voice: "",
  voiceByLang: {},
  skip: { figure: true, table: true, caption: false, formula: true, margin: true },
  cites: true,
  parens: false,
  autoScroll: true,
  zoom: 900,
};
let settings = structuredClone(defaults);
try {
  const saved = JSON.parse(localStorage.getItem("opentts.settings") || "{}");
  settings = { ...defaults, ...saved, skip: { ...defaults.skip, ...(saved.skip || {}) } };
} catch {}
const saveSettings = () => localStorage.setItem("opentts.settings", JSON.stringify(settings));

// ---------------------------------------------------------------- state

let doc = null; // {id, name, pages, sizes, lang}
let layouts = []; // per page: layout object once loaded
let layoutPromises = [];
let els = []; // els[p][i] = array of rect elements of that sentence
let voices = [];

let state = "idle"; // idle | loading | playing | paused
let cur = null; // {p, i} — reading position
let curIds = []; // sentences covered by the utterance being read
let consumed = new Set(); // sentences already read as the tail of a merged utterance
let gen = 0; // bumped on every jump/stop, cancels stale async work
let follow = true; // keep the reading position in view
let unlocked = false;

const key = (pos) => pos.p + ":" + pos.i;

function setState(s) {
  state = s;
  document.body.dataset.state = s;
  if ("mediaSession" in navigator) navigator.mediaSession.playbackState = s === "playing" || s === "loading" ? "playing" : s === "paused" ? "paused" : "none";
}

function toast(msg, kind = "error") {
  const t = $("toast");
  t.textContent = msg;
  t.className = kind;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 5000);
}

function setStatus(msg, warn = false) {
  const s = $("status");
  s.hidden = !msg;
  s.textContent = msg || "";
  s.classList.toggle("warn", warn);
}

// ---------------------------------------------------------------- document loading

async function openFile(file) {
  if (!file) return;
  stop();
  setStatus(`Opening ${file.name}…`);
  const body = new FormData();
  body.append("file", file);
  let meta;
  try {
    const res = await fetch("/api/docs", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    meta = await res.json();
  } catch (e) {
    setStatus("");
    toast("Could not open the PDF: " + e.message);
    return;
  }
  showDoc(meta);
}

function showDoc(meta) {
  stop();
  doc = meta;
  layouts = new Array(doc.pages).fill(null);
  layoutPromises = new Array(doc.pages).fill(null);
  els = Array.from({ length: doc.pages }, () => []);
  cur = null;
  curIds = [];
  consumed.clear();
  localStorage.setItem("opentts.last", doc.id);
  document.title = doc.name + " — OpenTTS";
  $("pageInfo").title = doc.tracked
    ? `${doc.name} — if this file changes on disk, refreshing the page loads the new version`
    : `${doc.name} — the original file was not located on disk, so refreshing will not pick up changes`;

  $("empty").hidden = true;
  pagesEl.textContent = "";
  for (let p = 0; p < doc.pages; p++) {
    const [w, h] = doc.sizes[p];
    const page = document.createElement("div");
    page.className = "page pending";
    page.dataset.p = p;
    page.style.aspectRatio = `${w} / ${h}`;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = `Page ${p + 1}`;
    img.draggable = false;
    img.src = `/api/docs/${doc.id}/pages/${p}/image`;
    const layer = document.createElement("div");
    layer.className = "layer";
    page.append(img, layer);
    pagesEl.append(page);
  }
  $("pageInfo").hidden = false;
  $("pageCount").textContent = doc.pages;
  $("pageNum").max = doc.pages;
  for (const id of ["play", "stop", "prev", "next"]) $(id).disabled = false;
  viewer.scrollTop = 0;

  pickVoiceForLanguage();
  if (!doc.has_text) {
    setStatus("This PDF has no text layer (it looks like a scan), so there is nothing to read. Run OCR on it first, e.g. “ocrmypdf in.pdf out.pdf”.", true);
  }

  // Restore the last reading position for this document.
  try {
    const saved = JSON.parse(localStorage.getItem("opentts.pos." + doc.id));
    if (saved && saved.p < doc.pages) {
      getLayout(saved.p).then((l) => {
        if (doc !== meta || cur || !l.items[saved.i]) return;
        cur = saved;
        curIds = [saved];
        paint();
        scrollToCurrent(true);
      });
    }
  } catch {}

  analyseAll(meta);
}

function getLayout(p) {
  if (!layoutPromises[p]) {
    const d = doc;
    layoutPromises[p] = fetch(`/api/docs/${d.id}/pages/${p}/layout`)
      .then((r) => {
        if (!r.ok) throw new Error("layout " + r.status);
        return r.json();
      })
      .then((layout) => {
        if (doc === d) {
          layouts[p] = layout;
          renderLayer(p, layout);
        }
        return layout;
      })
      .catch((e) => {
        if (doc === d) layoutPromises[p] = null;
        throw e;
      });
  }
  return layoutPromises[p];
}

// Analyse every page in the background, starting from what is on screen.
async function analyseAll(meta) {
  let failures = 0;
  while (doc === meta) {
    const done = layouts.filter(Boolean).length;
    if (done === doc.pages) break;
    if (doc.has_text) setStatus(`Analysing page layout… ${done} / ${doc.pages} pages (you can start listening already)`);
    const start = cur && state !== "idle" ? cur.p : visiblePage();
    let p = -1;
    for (let k = 0; k < doc.pages; k++) {
      const q = (start + k) % doc.pages;
      if (!layouts[q]) { p = q; break; }
    }
    if (p < 0) break;
    try {
      await getLayout(p);
    } catch (e) {
      if (++failures > 3) { toast("Page analysis failed: " + e.message); return; }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  if (doc === meta && doc.has_text) setStatus("");
}

function renderLayer(p, layout) {
  const page = pagesEl.children[p];
  const layer = page.lastElementChild;
  page.classList.remove("pending");
  const frag = document.createDocumentFragment();
  for (const reg of layout.regions) {
    const d = document.createElement("div");
    d.className = "region k-" + reg.k;
    d.dataset.label = REGION_LABEL[reg.k] || "";
    place(d, reg.r, 0);
    frag.append(d);
  }
  els[p] = layout.items.map((it, i) =>
    it.r.map((r) => {
      const d = document.createElement("div");
      d.className = "s k-" + it.k;
      d.dataset.i = i;
      place(d, r, 0.0025);
      frag.append(d);
      return d;
    })
  );
  layer.append(frag);
  paint();
}

function place(el, [x, y, w, h], pad) {
  el.style.left = (x - pad) * 100 + "%";
  el.style.top = (y - pad / 2) * 100 + "%";
  el.style.width = (w + 2 * pad) * 100 + "%";
  el.style.height = (h + pad) * 100 + "%";
}

// ---------------------------------------------------------------- reading order helpers

const item = (pos) => layouts[pos.p]?.items[pos.i];
const speakableText = (t) => /[\p{L}\p{N}]/u.test(t);
const isReadable = (it) => !settings.skip[it.k] && speakableText(it.t);

// Next readable sentence strictly after `pos` (or from the start of page `pos.p` if i === -1).
async function nextPos(pos, alsoSkip) {
  let { p, i } = pos;
  while (p < doc.pages) {
    const layout = await getLayout(p);
    for (i++; i < layout.items.length; i++) {
      const k = p + ":" + i;
      if (isReadable(layout.items[i]) && !consumed.has(k) && !(alsoSkip && alsoSkip.has(k))) return { p, i };
    }
    p++;
    i = -1;
  }
  return null;
}

async function prevPos(pos) {
  let { p, i } = pos;
  while (p >= 0) {
    const layout = await getLayout(p);
    if (i === Infinity) i = layout.items.length;
    for (i--; i >= 0; i--) if (isReadable(layout.items[i])) return { p, i };
    p--;
    i = Infinity;
  }
  return null;
}

const ENDS = /[.!?…:;]["')\]”’»]*\d{0,3}$/;

// A sentence, plus — when a paragraph is cut by a column or page break — its continuation.
async function buildUtterance(pos) {
  const first = item(pos);
  const ids = [pos];
  let text = first.t;
  let last = first;
  let at = pos;
  for (let hop = 0; hop < 2; hop++) {
    if (last.k !== "text" || !last.l || ENDS.test(text)) break;
    // Look for a paragraph start, shortly after, that begins in lower case.
    let found = null;
    let tried = 0;
    let { p, i } = at;
    scan: for (; p < doc.pages && p <= at.p + 1; p++, i = -1) {
      const layout = p === at.p ? layouts[p] : await getLayout(p).catch(() => null);
      if (!layout) break;
      for (i++; i < layout.items.length; i++) {
        const c = layout.items[i];
        if (c.k !== "text" || !c.f) continue;
        if (/^\p{Ll}/u.test(c.t) && !consumed.has(p + ":" + i)) { found = { p, i }; break scan; }
        if (++tried >= 3) break scan;
      }
    }
    if (!found) break;
    const c = item(found);
    text = /\p{L}-$/u.test(text) ? text.slice(0, -1) + c.t : text + " " + c.t;
    ids.push(found);
    last = c;
    at = found;
  }
  const pause = first.k === "heading" ? 0.45 : last.l ? 0.32 : 0.1;
  return { ids, text, pause };
}

// ---------------------------------------------------------------- audio

const audioCache = new Map(); // request key -> {promise, ctrl, url}
const AUDIO_CACHE_MAX = 40;

function fetchAudio(u) {
  const k = [settings.voice, settings.cites ? 1 : 0, settings.parens ? 1 : 0, u.pause, u.text].join("|");
  let entry = audioCache.get(k);
  if (entry) return entry.promise;
  const ctrl = new AbortController();
  entry = { ctrl, url: null };
  entry.promise = fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: u.text, voice: settings.voice, pause: u.pause, strip_citations: settings.cites, strip_parens: settings.parens }),
    signal: ctrl.signal,
  })
    .then(async (r) => {
      if (r.status === 204) return null; // nothing pronounceable
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "speech synthesis failed");
      entry.url = URL.createObjectURL(await r.blob());
      return entry.url;
    })
    .catch((e) => {
      audioCache.delete(k);
      throw e;
    });
  audioCache.set(k, entry);
  while (audioCache.size > AUDIO_CACHE_MAX) {
    const [oldKey, old] = audioCache.entries().next().value;
    if (old.url && old.url !== audio.src) URL.revokeObjectURL(old.url);
    audioCache.delete(oldKey);
  }
  return entry.promise;
}

function dropPending() {
  for (const [k, e] of audioCache) {
    if (!e.url) {
      e.ctrl.abort();
      audioCache.delete(k);
    }
  }
}

function clearAudioCache() {
  for (const e of audioCache.values()) {
    e.ctrl.abort();
    if (e.url && e.url !== audio.src) URL.revokeObjectURL(e.url);
  }
  audioCache.clear();
}

// Browsers only allow sound that starts from a user gesture; claim that right once.
function unlock() {
  if (unlocked) return;
  unlocked = true;
  audio.src = SILENCE;
  audio.play().catch(() => {});
}

async function prefetch(myGen, from, ids) {
  const skip = new Set(ids.map(key));
  let pos = from;
  for (let n = 0; n < 2; n++) {
    pos = await nextPos(pos, skip);
    if (!pos || myGen !== gen) return;
    const u = await buildUtterance(pos);
    if (myGen !== gen) return;
    u.ids.forEach((id) => skip.add(key(id)));
    await fetchAudio(u).catch(() => {});
    if (myGen !== gen) return;
  }
}

// ---------------------------------------------------------------- transport

async function playFrom(pos) {
  const myGen = ++gen;
  audio.pause();
  dropPending();
  setState("loading");
  cur = pos;
  curIds = [pos];
  paint();
  try {
    await getLayout(pos.p);
    if (myGen !== gen) return;
    const u = await buildUtterance(pos);
    if (myGen !== gen) return;
    curIds = u.ids;
    u.ids.slice(1).forEach((id) => consumed.add(key(id)));
    paint();
    scrollToCurrent();
    localStorage.setItem("opentts.pos." + doc.id, JSON.stringify(pos));
    const url = await fetchAudio(u);
    if (myGen !== gen) return;
    if (!url) return advance(myGen);
    audio.src = url;
    audio.playbackRate = settings.speed;
    await audio.play();
    if (myGen !== gen) return;
    setState("playing");
    prefetch(myGen, pos, u.ids).catch(() => {});
  } catch (e) {
    if (myGen !== gen || e.name === "AbortError") return;
    setState("idle");
    paint();
    toast(e instanceof TypeError ? "OpenTTS is not running any more — click its icon to start it again, then reload this page." : e.message || String(e));
  }
}

async function advance(myGen) {
  const next = await nextPos(cur).catch(() => null);
  if (myGen !== gen) return;
  if (next) return playFrom(next);
  // End of the document.
  gen++;
  setState("idle");
  paint();
}

audio.addEventListener("ended", () => {
  if (state === "playing") advance(gen);
});
audio.addEventListener("error", () => {
  if (state === "playing" && audio.src !== SILENCE) advance(gen);
});

async function firstOnScreen() {
  const p = visiblePage();
  const layout = await getLayout(p);
  // First readable sentence whose top is inside the viewport, else the first of the page.
  const page = pagesEl.children[p].getBoundingClientRect();
  const top = viewer.getBoundingClientRect().top;
  for (let i = 0; i < layout.items.length; i++) {
    const it = layout.items[i];
    if (isReadable(it) && page.top + it.r[0][1] * page.height >= top - 4) return { p, i };
  }
  return nextPos({ p, i: -1 });
}

async function togglePlay() {
  if (!doc) return;
  unlock();
  if (state === "playing") {
    audio.pause();
    setState("paused");
  } else if (state === "paused") {
    follow = true;
    $("follow").hidden = true;
    audio.playbackRate = settings.speed;
    await audio.play().catch(() => {});
    setState("playing");
    scrollToCurrent();
  } else if (state === "idle") {
    setState("loading");
    const myGen = ++gen;
    let pos = cur;
    try {
      if (!pos) pos = await firstOnScreen();
    } catch (e) {
      toast(e.message);
    }
    if (myGen !== gen) return;
    if (!pos) {
      setState("idle");
      if (doc.has_text) toast("Nothing to read here.");
      return;
    }
    follow = true;
    $("follow").hidden = true;
    playFrom(pos);
  }
}

function stop() {
  gen++;
  audio.pause();
  dropPending();
  consumed.clear();
  if (cur) curIds = [cur];
  setState("idle");
  paint();
}

async function step(dir) {
  if (!doc || !cur) return togglePlay();
  unlock();
  const wasIdle = state === "idle" || state === "paused";
  const myGen = ++gen;
  audio.pause();
  consumed.clear();
  const pos = await (dir > 0 ? nextPos(cur) : prevPos(cur)).catch(() => null);
  if (myGen !== gen) return;
  if (!pos) {
    if (!wasIdle) playFrom(cur);
    return;
  }
  follow = true;
  $("follow").hidden = true;
  if (wasIdle) {
    cur = pos;
    curIds = [pos];
    setState("idle");
    paint();
    scrollToCurrent();
    localStorage.setItem("opentts.pos." + doc.id, JSON.stringify(pos));
  } else {
    playFrom(pos);
  }
}

// ---------------------------------------------------------------- highlighting & scrolling

let painted = [];
function paint() {
  for (const el of painted) el.classList.remove("active", "cursor");
  painted = [];
  if (!cur) return;
  const cls = state === "idle" ? "cursor" : "active";
  for (const id of curIds) {
    for (const el of els[id.p]?.[id.i] || []) {
      el.classList.add(cls);
      painted.push(el);
    }
  }
}

function scrollToCurrent(force = false) {
  if (!cur || (!(follow && settings.autoScroll) && !force)) return;
  const el = els[cur.p]?.[cur.i]?.[0] || pagesEl.children[cur.p];
  if (!el) return;
  const r = el.getBoundingClientRect();
  const v = viewer.getBoundingClientRect();
  if (force || r.top < v.top + v.height * 0.12 || r.bottom > v.top + v.height * 0.72) {
    programmaticScroll = performance.now();
    viewer.scrollTo({ top: viewer.scrollTop + r.top - v.top - v.height * 0.3, behavior: "smooth" });
  }
}

function visiblePage() {
  const mid = viewer.getBoundingClientRect().top + viewer.clientHeight / 3;
  const kids = pagesEl.children;
  let lo = 0, hi = kids.length - 1;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (kids[m].getBoundingClientRect().bottom < mid) lo = m + 1;
    else hi = m;
  }
  return lo;
}

let programmaticScroll = 0;
let scrollTick = false;
viewer.addEventListener("scroll", () => {
  if (scrollTick) return;
  scrollTick = true;
  requestAnimationFrame(() => {
    scrollTick = false;
    if (doc && document.activeElement !== $("pageNum")) $("pageNum").value = visiblePage() + 1;
  });
});
// The listener scrolled away on purpose: stop pulling the view back.
function userScrolled() {
  if (state === "idle" || !follow || !settings.autoScroll) return;
  follow = false;
  $("follow").hidden = false;
}
viewer.addEventListener("wheel", userScrolled, { passive: true });
viewer.addEventListener("touchmove", userScrolled, { passive: true });
$("follow").onclick = () => {
  follow = true;
  $("follow").hidden = true;
  scrollToCurrent(true);
};

// "Follow" toggle: whether the view scrolls along with the reading at all.
function applyAutoScroll() {
  $("autoScroll").setAttribute("aria-pressed", settings.autoScroll ? "true" : "false");
}
$("autoScroll").onclick = () => {
  settings.autoScroll = !settings.autoScroll;
  saveSettings();
  applyAutoScroll();
  follow = true;
  $("follow").hidden = true;
  if (settings.autoScroll) scrollToCurrent(true);
};

// ---------------------------------------------------------------- page interaction

let hovered = [];
function hover(list) {
  for (const el of hovered) el.classList.remove("hover");
  hovered = list || [];
  for (const el of hovered) el.classList.add("hover");
}
pagesEl.addEventListener("mouseover", (e) => {
  const s = e.target.closest(".s");
  if (!s) return hover(null);
  const p = +s.closest(".page").dataset.p;
  hover(els[p][+s.dataset.i]);
});
pagesEl.addEventListener("mouseleave", () => hover(null));

pagesEl.addEventListener("click", (e) => {
  const s = e.target.closest(".s");
  if (!s) return;
  unlock();
  consumed.clear();
  follow = true;
  $("follow").hidden = true;
  // A click always reads the clicked sentence, even one of an ignored kind.
  playFrom({ p: +s.closest(".page").dataset.p, i: +s.dataset.i });
});

// ---------------------------------------------------------------- controls

$("open").onclick = $("open2").onclick = () => $("file").click();
$("file").onchange = (e) => {
  openFile(e.target.files[0]);
  e.target.value = "";
};
$("play").onclick = togglePlay;
$("stop").onclick = stop;
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);

function setSpeed(v) {
  settings.speed = Math.min(3, Math.max(0.5, Math.round(v * 20) / 20));
  $("speed").value = settings.speed;
  $("speedLabel").textContent = settings.speed.toFixed(2) + "×";
  audio.defaultPlaybackRate = settings.speed;
  audio.playbackRate = settings.speed;
  saveSettings();
}
$("speed").oninput = (e) => setSpeed(+e.target.value);
$("slower").onclick = () => setSpeed(settings.speed - 0.1);
$("faster").onclick = () => setSpeed(settings.speed + 0.1);
$("speedLabel").onclick = () => setSpeed(1);
audio.preservesPitch = true;

function applySkip() {
  let n = 0;
  for (const k of SKIP_KINDS) {
    pagesEl.classList.toggle("skip-" + k, !!settings.skip[k]);
    if (settings.skip[k]) n++;
  }
  n += (settings.cites ? 1 : 0) + (settings.parens ? 1 : 0);
  $("ignoreCount").textContent = n ? `(${n})` : "";
}
for (const box of document.querySelectorAll("[data-skip]")) {
  box.checked = !!settings.skip[box.dataset.skip];
  box.onchange = () => {
    settings.skip[box.dataset.skip] = box.checked;
    saveSettings();
    applySkip();
    dropPending();
  };
}
$("cites").checked = settings.cites;
$("cites").onchange = (e) => {
  settings.cites = e.target.checked;
  saveSettings();
  applySkip();
};
$("parens").checked = settings.parens;
$("parens").onchange = (e) => {
  settings.parens = e.target.checked;
  saveSettings();
  applySkip();
  dropPending();
};
document.addEventListener("click", (e) => {
  if (!$("ignore").contains(e.target)) $("ignore").open = false;
});

function setZoom(w) {
  settings.zoom = Math.min(2400, Math.max(400, Math.round(w)));
  document.documentElement.style.setProperty("--page-w", settings.zoom + "px");
  pagesEl.classList.toggle("zoomed", settings.zoom > viewer.clientWidth - 32);
  saveSettings();
}
$("zoomIn").onclick = () => setZoom(settings.zoom * 1.15);
$("zoomOut").onclick = () => setZoom(settings.zoom / 1.15);
window.addEventListener("resize", () => setZoom(settings.zoom));

$("pageNum").onchange = (e) => {
  const p = Math.min(doc.pages, Math.max(1, +e.target.value || 1)) - 1;
  pagesEl.children[p].scrollIntoView({ block: "start" });
  userScrolled();
  e.target.blur();
};

// voices
async function loadVoices() {
  for (let attempt = 0; attempt < 60 && !voices.length; attempt++) {
    try {
      const r = await fetch("/api/voices");
      if (r.ok) voices = await r.json();
    } catch {}
    if (!voices.length) await new Promise((res) => setTimeout(res, 1000));
  }
  const sel = $("voice");
  sel.textContent = "";
  const groups = new Map();
  for (const v of voices) {
    if (!groups.has(v.group)) {
      const g = document.createElement("optgroup");
      g.label = v.group;
      groups.set(v.group, g);
      sel.append(g);
    }
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = `${v.name} · ${v.gender}`;
    groups.get(v.group).append(o);
  }
  if (!voices.some((v) => v.id === settings.voice)) settings.voice = voices.some((v) => v.id === "af_heart") ? "af_heart" : voices[0]?.id || "";
  sel.value = settings.voice;
  pickVoiceForLanguage();
}

const DEFAULT_VOICE = { en: "af_heart", it: "if_sara", fr: "ff_siwis", es: "ef_dora", pt: "pf_dora" };
// Switch to a voice that speaks the document's language (remembering the user's pick per language).
function pickVoiceForLanguage() {
  if (!doc || !voices.length) return;
  const current = voices.find((v) => v.id === settings.voice);
  if (current && current.lang === doc.lang) return;
  const want = settings.voiceByLang[doc.lang] || DEFAULT_VOICE[doc.lang];
  const v = voices.find((x) => x.id === want) || voices.find((x) => x.lang === doc.lang);
  if (v) {
    settings.voice = v.id;
    $("voice").value = v.id;
    saveSettings();
  }
}
$("voice").onchange = (e) => {
  settings.voice = e.target.value;
  const v = voices.find((x) => x.id === settings.voice);
  if (v) settings.voiceByLang[v.lang] = v.id;
  saveSettings();
  clearAudioCache();
  if ((state === "playing" || state === "loading") && cur) playFrom(cur);
  else if (state === "paused") stop();
  e.target.blur();
};

// keyboard
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = e.target.tagName;
  if (tag === "SELECT" || (tag === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox")) return;
  switch (e.key) {
    case " ":
      if (tag === "BUTTON" || tag === "SUMMARY") return;
      e.preventDefault();
      togglePlay();
      break;
    case "ArrowRight": if (tag === "INPUT") return; e.preventDefault(); step(1); break;
    case "ArrowLeft": if (tag === "INPUT") return; e.preventDefault(); step(-1); break;
    case "Escape": stop(); break;
    case "+": case "=": setSpeed(settings.speed + 0.1); break;
    case "-": case "_": setSpeed(settings.speed - 0.1); break;
  }
});

if ("mediaSession" in navigator) {
  const ms = navigator.mediaSession;
  ms.setActionHandler("play", togglePlay);
  ms.setActionHandler("pause", togglePlay);
  ms.setActionHandler("stop", stop);
  ms.setActionHandler("nexttrack", () => step(1));
  ms.setActionHandler("previoustrack", () => step(-1));
}

// drag & drop anywhere
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const f = [...e.dataTransfer.files].find((x) => x.type === "application/pdf" || /\.pdf$/i.test(x.name));
  if (f) openFile(f);
  else toast("Please drop a PDF file.");
});

// Keep a stream open so the server knows this tab exists: when started from the app icon
// it quits a few seconds after the last tab is closed.
let serverLink = null;
function linkServer() {
  if (serverLink && serverLink.readyState !== EventSource.CLOSED) return;
  serverLink = new EventSource("/api/events");
}
linkServer();
setInterval(linkServer, 3000);

// ---------------------------------------------------------------- start

setState("idle");
setSpeed(settings.speed);
setZoom(settings.zoom);
applySkip();
applyAutoScroll();
loadVoices();
(async () => {
  const last = localStorage.getItem("opentts.last");
  if (!last) return;
  try {
    // "latest" also checks whether the PDF file changed on disk since it was opened.
    const r = await fetch("/api/docs/" + last + "/latest");
    if (!r.ok || doc) return;
    const meta = await r.json();
    if (meta.replaces) {
      // A newer version: keep the reading position, as far as it still exists.
      const pos = localStorage.getItem("opentts.pos." + meta.replaces);
      if (pos) localStorage.setItem("opentts.pos." + meta.id, pos);
    }
    showDoc(meta);
    if (meta.replaces) toast(`Loaded the updated version of ${meta.name}`, "info");
  } catch {}
})();
