"""OpenTTS — local PDF read-aloud server.

Run with:  uv run python app.py   (or ./run.sh)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pdfproc

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("OPENTTS_DATA") or ROOT / "data").resolve()
MODELS = ROOT / "models"
KOKORO_MODEL = MODELS / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS / "voices-v1.0.bin"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# Voice name prefix -> (espeak language code, short language code, label)
LANGS = {
    "a": ("en-us", "en", "English (US)"),
    "b": ("en-gb", "en", "English (UK)"),
    "i": ("it", "it", "Italiano"),
    "f": ("fr-fr", "fr", "Français"),
    "e": ("es", "es", "Español"),
    "p": ("pt-br", "pt", "Português (BR)"),
}



@asynccontextmanager
async def lifespan(_app):
    warm_up()
    watch_idle()
    yield


app = FastAPI(title="OpenTTS", lifespan=lifespan)

# --------------------------------------------------------------------------
# automatic shutdown: when started from the macOS app there is no terminal to stop
# the server from. Every open tab holds an event stream open; the server quits a few
# seconds after the last one goes away (the delay lets a page reload reconnect).

APP_MODE = float(os.environ.get("OPENTTS_IDLE_EXIT", "0")) > 0
NO_TAB_GRACE = 6.0  # seconds without any tab before quitting
NEVER_OPENED_EXIT = float(os.environ.get("OPENTTS_IDLE_EXIT", "0"))  # if no tab ever connects
_tabs = 0
_tabs_seen = False
_tabs_empty_since = time.monotonic()
_last_seen = time.monotonic()


@app.middleware("http")
async def _touch(request: Request, call_next):
    global _last_seen
    _last_seen = time.monotonic()
    return await call_next(request)


@app.get("/api/ping")
def ping():
    return {"ok": True}


@app.get("/api/events")
async def events():
    """Held open by each tab; its closing tells us the tab is gone."""

    async def stream():
        global _tabs, _tabs_seen, _tabs_empty_since
        _tabs += 1
        _tabs_seen = True
        try:
            while True:
                yield ": keep-alive\n\n"
                await asyncio.sleep(2)
        finally:
            _tabs -= 1
            if _tabs <= 0:
                _tabs_empty_since = time.monotonic()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def watch_idle():
    if not APP_MODE:
        return

    def run():
        while True:
            time.sleep(1)
            now = time.monotonic()
            if _tabs_seen:
                gone = _tabs <= 0 and now - _tabs_empty_since > NO_TAB_GRACE
            else:
                gone = now - _last_seen > NEVER_OPENED_EXIT
            if gone:
                print("[opentts] no open tab, shutting down")
                os._exit(0)

    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------
# documents

_pdf_lock = threading.Lock()  # PyMuPDF documents are not thread-safe
_docs: dict[str, fitz.Document] = {}


def _doc_dir(doc_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id):
        raise HTTPException(404, "unknown document")
    d = DATA / doc_id
    if not (d / "doc.pdf").exists():
        raise HTTPException(404, "unknown document")
    return d


def _open(doc_id: str) -> fitz.Document:
    """Must be called with _pdf_lock held."""
    doc = _docs.get(doc_id)
    if doc is None:
        doc = fitz.open(_doc_dir(doc_id) / "doc.pdf")
        _docs[doc_id] = doc
    return doc


def _page(doc: fitz.Document, n: int) -> fitz.Page:
    if not 0 <= n < doc.page_count:
        raise HTTPException(404, "no such page")
    return doc[n]


def _doc_id(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:16]


def _stat(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_mtime_ns, st.st_size]


def _public(meta: dict) -> dict:
    """Metadata as sent to the browser: no file-system paths."""
    out = {k: v for k, v in meta.items() if not k.startswith("source")}
    out["tracked"] = bool(meta.get("source"))
    return out


def _load_meta(doc_id: str) -> dict:
    return json.loads((_doc_dir(doc_id) / "meta.json").read_text())


def _save_meta(meta: dict):
    (DATA / meta["id"] / "meta.json").write_text(json.dumps(meta))


def _opening_words(doc: fitz.Document) -> set[str]:
    text = " ".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
    return set(re.findall(r"\w{3,}", text.lower()))


def find_source(name: str, doc_id: str) -> Path | None:
    """Locate on disk the file a PDF was opened from.

    The browser only hands over a file's name and content, never its path, so we ask
    Spotlight for the files with that name. The one with identical content is the
    source. If none is identical, the file has been rewritten since it was opened:
    then the source is the same-named file modified after the opening whose first
    pages look most like the opened document (names like "main.pdf" are common).
    """
    if not shutil.which("mdfind"):
        return None
    try:
        out = subprocess.run(["mdfind", "-name", name], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    stored = DATA / doc_id / "doc.pdf"
    size, opened_at = stored.stat().st_size, stored.stat().st_mtime_ns
    identical, newer = [], []
    for line in out.splitlines()[:500]:
        path = Path(line)
        try:
            if path.name != name or DATA in path.parents or not path.is_file():
                continue
            st = path.stat()
            if st.st_size == size and _doc_id(path.read_bytes()) == doc_id:
                identical.append((st.st_mtime_ns, path))
            elif st.st_mtime_ns > opened_at and st.st_size > 0:
                newer.append((st.st_mtime_ns, path))
        except OSError:  # unreadable, or macOS denied access to that folder
            continue
    if identical:
        return max(identical)[1]
    if not newer:
        return None
    with _pdf_lock:
        mine = _opening_words(_open(doc_id))
        best, best_score = None, 0.5  # at least half of the opening words in common
        for _, path in sorted(newer, reverse=True)[:20]:
            try:
                with fitz.open(path) as cand:
                    theirs = _opening_words(cand)
            except Exception:
                continue
            score = len(mine & theirs) / max(1, len(mine | theirs))
            if score > best_score:
                best, best_score = path, score
    return best


def _locate(meta: dict) -> Path | None:
    """Find and remember the source file of a document that has none (or lost it)."""
    meta.pop("source", None)
    meta.pop("source_stat", None)
    source = find_source(meta["name"], meta["id"])
    if source is not None:
        meta["source"] = str(source)
        _save_meta(meta)
    return source


def ingest(raw: bytes, name: str) -> dict:
    """Store a PDF (if not already known) and return its metadata."""
    doc_id = _doc_id(raw)
    d = DATA / doc_id
    meta_path = d / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception:
        raise HTTPException(400, "This file could not be opened as a PDF.")
    if doc.needs_pass:
        raise HTTPException(400, "Password-protected PDFs are not supported.")
    with _pdf_lock:
        has_text = any(doc[i].get_text("words") for i in range(min(5, doc.page_count)))
        sizes = [[doc[i].rect.width, doc[i].rect.height] for i in range(doc.page_count)]
        meta = {
            "id": doc_id,
            "name": name,
            "pages": doc.page_count,
            "sizes": sizes,
            "lang": pdfproc.detect_language(doc),
            "has_text": has_text,
        }
    doc.close()
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / "doc.pdf").write_bytes(raw)
    meta_path.write_text(json.dumps(meta))
    return meta


@app.post("/api/docs")
def upload(file: UploadFile = File(...)):
    raw = file.file.read()
    meta = ingest(raw, Path(file.filename or "document.pdf").name)
    # Remember where the file lives so a page refresh can pick up a newer version.
    if not meta.get("source") or not Path(meta["source"]).is_file():
        _locate(meta)
    return _public(meta)


@app.get("/api/docs/{doc_id}")
def doc_meta(doc_id: str):
    return _public(_load_meta(doc_id))


def _read_stable(path: Path) -> tuple[bytes, list[int]] | None:
    """Read a file that may be in the middle of being rewritten (e.g. by LaTeX)."""
    before = _stat(path)
    time.sleep(0.2)
    raw = path.read_bytes()
    if _stat(path) != before or b"%%EOF" not in raw[-2048:]:
        return None
    return raw, before


@app.get("/api/docs/{doc_id}/latest")
def latest(doc_id: str):
    """The newest version of a document: if its file changed on disk, load that."""
    meta = _load_meta(doc_id)
    source = Path(meta["source"]) if meta.get("source") else None
    try:
        if source is None or not source.is_file():
            source = _locate(meta)  # not located yet, or moved
            if source is None:
                return _public(meta)
        # Same path, same modification time and size as when it was loaded: nothing new.
        if _stat(source) == meta.get("source_stat"):
            return _public(meta)
        read = _read_stable(source)
        if read is None:  # still being written; the next refresh will get it
            return _public(meta)
        raw, stat = read
        if _doc_id(raw) == doc_id:  # touched but not changed
            meta["source_stat"] = stat
            _save_meta(meta)
            return _public(meta)
        new = ingest(raw, source.name)
    except (OSError, HTTPException) as exc:
        print(f"[update] could not check {meta.get('source')}: {exc!r}")
        return _public(meta)
    new["source"], new["source_stat"] = str(source), stat
    _save_meta(new)
    return {**_public(new), "replaces": doc_id}


@app.get("/api/docs/{doc_id}/pages/{n}/layout")
def page_layout(doc_id: str, n: int):
    cache = _doc_dir(doc_id) / "pages" / f"{n}.json"
    if cache.exists():
        return Response(cache.read_bytes(), media_type="application/json")
    with _pdf_lock:
        if cache.exists():
            return Response(cache.read_bytes(), media_type="application/json")
        result = pdfproc.analyze_page(_page(_open(doc_id), n))
        body = json.dumps(result, ensure_ascii=False).encode()
        cache.write_bytes(body)
    return Response(body, media_type="application/json")


SVG_MAX_BYTES = 8_000_000  # beyond this a page is too heavy to draw as vectors


@app.get("/api/docs/{doc_id}/pages/{n}/image")
def page_image(doc_id: str, n: int):
    """The page as vector graphics (SVG), so text stays sharp at any zoom."""
    _doc_dir(doc_id)
    headers = {"Cache-Control": "private, max-age=86400"}
    with _pdf_lock:
        page = _page(_open(doc_id), n)
        try:
            svg = page.get_svg_image(text_as_path=True)
            if len(svg) <= SVG_MAX_BYTES and "viewBox" in svg[:1000]:
                return Response(svg, media_type="image/svg+xml", headers=headers)
        except Exception as exc:
            print(f"[render] SVG export failed on page {n + 1}: {exc!r}")
        # Fallback for pathological pages: a high-resolution bitmap.
        png = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).tobytes("png")
    return Response(png, media_type="image/png", headers=headers)


# --------------------------------------------------------------------------
# speech

_kokoro = None
_kokoro_lock = threading.Lock()
_tts_queue = asyncio.Lock()
_audio_cache: OrderedDict[tuple, bytes] = OrderedDict()
AUDIO_CACHE_MAX = 300


def _download(url: str, dest: Path):
    import urllib.request

    print(f"[tts] downloading {dest.name} ...")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def get_kokoro():
    global _kokoro
    with _kokoro_lock:
        if _kokoro is None:
            from kokoro_onnx import Kokoro

            MODELS.mkdir(exist_ok=True)
            for path in (KOKORO_MODEL, KOKORO_VOICES):
                if not path.exists():
                    _download(KOKORO_URL + path.name, path)
            _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        return _kokoro


_EN_EXPAND = [
    (re.compile(r"\bFigs\.\s*", re.I), "Figures "),
    (re.compile(r"\bFig\.\s*", re.I), "Figure "),
    (re.compile(r"\bEqs\.\s*", re.I), "Equations "),
    (re.compile(r"\bEqn?\.\s*", re.I), "Equation "),
    (re.compile(r"\bSect?\.\s*(?=\d)", re.I), "Section "),
    (re.compile(r"\bTab\.\s*(?=\d)", re.I), "Table "),
    (re.compile(r"\be\.g\.,?", re.I), "for example,"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
    (re.compile(r"\bet al\.", re.I), "et al"),
    (re.compile(r"\bw\.r\.t\.", re.I), "with respect to"),
    (re.compile(r"\bvs\.", re.I), "versus"),
]
_CITE = re.compile(r"\s*\[\s*\d+(?:\s*[-–,;]\s*\d+)*\s*\]")
_FOOTNOTE_MARK = re.compile(r"(?<=[a-zà-ÿ]{3})([.,;:!?][\"”’)]*)\d{1,3}(?=\s|$)")
_URL = re.compile(r"(?:https?://|www\.)\S+")


_LABEL = re.compile(r"[A-Za-z0-9]{1,3}")
_OPENERS = " \t\n[\"'“‘«—–-"


def strip_parentheses(text: str) -> str:
    """Drop asides in round brackets, nested ones included.

    Kept on purpose: brackets glued to a word, as in f(x) or word(s), and short
    labels such as (3), (A) or (ii), which are references rather than asides.
    Unbalanced brackets are left alone.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "(" and (i == 0 or text[i - 1] in _OPENERS):
            depth, j = 1, i + 1
            while j < n and depth:
                depth += (text[j] == "(") - (text[j] == ")")
                j += 1
            if depth == 0 and not _LABEL.fullmatch(text[i + 1 : j - 1]):
                out.append(" ")
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def speakable(text: str, lang: str, strip_citations: bool, strip_parens: bool = False) -> str:
    """Turn a sentence as printed into something pleasant to listen to."""
    if strip_parens:
        text = strip_parentheses(text)
    if strip_citations:
        text = _CITE.sub("", text)
        text = _FOOTNOTE_MARK.sub(r"\1", text)
    text = _URL.sub(" link ", text)
    if lang == "en":
        for rx, rep in _EN_EXPAND:
            text = rx.sub(rep, text)
    text = text.replace("∗", "").replace("†", "").replace("‡", "").replace("•", ",")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def synthesize(text: str, voice: str, pause: float) -> bytes:
    espeak_lang = LANGS[voice[0]][0]
    kokoro = get_kokoro()
    with _kokoro_lock:
        samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=espeak_lang)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if pause > 0:
        samples = np.concatenate([samples, np.zeros(int(sr * pause), dtype=np.float32)])
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    pause: float = 0.12  # seconds of silence appended after the sentence
    strip_citations: bool = True
    strip_parens: bool = False  # drop asides in (round brackets)


@app.get("/api/voices")
def voices():
    names = sorted(get_kokoro().get_voices())
    out = []
    for v in names:
        if v[0] in LANGS and len(v) > 3 and v[2] == "_":
            _, lang, label = LANGS[v[0]]
            out.append(
                {
                    "id": v,
                    "name": v[3:].replace("_", " ").title(),
                    "gender": "female" if v[1] == "f" else "male",
                    "lang": lang,
                    "group": label,
                }
            )
    return out


@app.post("/api/tts")
async def tts(req: TTSRequest, request: Request):
    if not req.voice or req.voice[0] not in LANGS:
        raise HTTPException(400, "unknown voice")
    text = speakable(req.text[:2000], LANGS[req.voice[0]][1], req.strip_citations, req.strip_parens)
    if not re.search(r"\w", text):
        return Response(status_code=204)  # nothing pronounceable
    pause = min(max(req.pause, 0.0), 2.0)
    key = (req.voice, text, round(pause, 2))
    audio = _audio_cache.get(key)
    if audio is None:
        async with _tts_queue:
            # The listener may have jumped elsewhere while this request waited.
            if await request.is_disconnected():
                return Response(status_code=204)
            audio = _audio_cache.get(key)
            if audio is None:
                try:
                    audio = await run_in_threadpool(synthesize, text, req.voice, pause)
                except Exception as exc:
                    raise HTTPException(500, f"speech synthesis failed: {exc}")
                _audio_cache[key] = audio
                while len(_audio_cache) > AUDIO_CACHE_MAX:
                    _audio_cache.popitem(last=False)
    else:
        _audio_cache.move_to_end(key)
    return Response(audio, media_type="audio/wav")


@app.get("/api/status")
def status():
    return {"tts_ready": _kokoro is not None, "layout_ready": pdfproc._model is not None}


def warm_up():
    """Load both models in the background so the first click is fast."""

    def run():
        try:
            get_kokoro()
            pdfproc.load_layout_model()
            print("[opentts] models ready")
        except Exception as exc:
            print(f"[opentts] model warm-up failed: {exc!r}")

    threading.Thread(target=run, daemon=True).start()


class FreshStatic(StaticFiles):
    """Static files the browser must revalidate, so an updated app is never served stale."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


@app.get("/")
def index():
    # Stamp the script and stylesheet URLs with their modification time: browsers
    # otherwise keep using a cached copy for hours after the files change.
    static = ROOT / "static"
    html = (static / "index.html").read_text()
    for name in ("app.js", "style.css"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={(static / name).stat().st_mtime_ns}")
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


app.mount("/static", FreshStatic(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import webbrowser

    import uvicorn

    port = int(os.environ.get("OPENTTS_PORT", "8765"))
    if not os.environ.get("OPENTTS_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
