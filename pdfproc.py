"""PDF analysis: layout detection, reading order and sentence segmentation.

For every page we produce a list of "items" (sentences) in reading order. Each
item carries its kind (text, heading, caption, figure, table, formula, margin)
and the rectangles it covers on the page, so the UI can highlight it, start
reading from it on click, and skip the kinds the user wants ignored.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

MODELS_DIR = Path(__file__).parent / "models"
LAYOUT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"

# DocLayout-YOLO class id -> our kind
KIND = {
    0: "heading",  # title
    1: "text",  # plain text
    2: "margin",  # "abandon": running heads, footers, page numbers
    3: "figure",
    4: "caption",  # figure caption
    5: "table",
    6: "caption",  # table caption
    7: "table",  # table footnote
    8: "formula",  # isolated formula
    9: "formula",  # formula number
}
CONTAINER_KINDS = {"figure", "table"}
SPLIT_KINDS = {"text", "caption", "figure", "table", "formula", "margin"}

_model = None
_model_lock = threading.Lock()


def load_layout_model():
    """Load (downloading on first use) the layout detector. Thread-safe."""
    global _model
    with _model_lock:
        if _model is None:
            from doclayout_yolo import YOLOv10
            from doclayout_yolo.hub.utils import events

            # The library (an Ultralytics fork) would otherwise report anonymous usage
            # statistics to Google Analytics on every prediction. Nothing leaves this machine.
            events.enabled = False

            path = MODELS_DIR / LAYOUT_FILE
            if not path.exists():
                from huggingface_hub import hf_hub_download

                hf_hub_download(LAYOUT_REPO, LAYOUT_FILE, local_dir=str(MODELS_DIR))
            _model = YOLOv10(str(path))
        return _model


# --------------------------------------------------------------------------
# data


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block: int
    line: int


@dataclass
class Unit:
    """A group of words that is read as one piece (a paragraph, a caption...)."""

    kind: str
    words: list[Word] = field(default_factory=list)
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0

    def fit(self):
        self.x0 = min(w.x0 for w in self.words)
        self.y0 = min(w.y0 for w in self.words)
        self.x1 = max(w.x1 for w in self.words)
        self.y1 = max(w.y1 for w in self.words)


# --------------------------------------------------------------------------
# layout regions


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def detect_regions(page: fitz.Page) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Run the layout detector on a page; boxes are returned in page points."""
    from PIL import Image

    model = load_layout_model()
    rect = page.rect
    scale = 1024 / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    with _model_lock:
        res = model.predict(img, imgsz=1024, conf=0.25, device="cpu", verbose=False)[0]
    regions = []
    for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
        kind = KIND.get(int(cls), "text")
        regions.append((kind, tuple(v / scale for v in box)))

    # Text detected *inside* a figure or table (axis labels, cell contents...)
    # belongs to that figure/table.
    containers = [r for k, r in regions if k in CONTAINER_KINDS]
    out = []
    for kind, r in regions:
        if kind not in CONTAINER_KINDS and kind != "caption":
            a = _area(r)
            if a > 0 and any(_inter(r, c) / a > 0.8 for c in containers):
                continue
        out.append((kind, r))
    return out


# --------------------------------------------------------------------------
# words -> units


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if ch.isprintable()).strip()


def extract_words(page: fitz.Page) -> list[Word]:
    rot = page.rotation_matrix if page.rotation else None
    words = []
    for x0, y0, x1, y1, text, block, line, _ in page.get_text("words"):
        text = _clean(text)
        if not text:
            continue
        if rot is not None:
            r = fitz.Rect(x0, y0, x1, y1) * rot
            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        words.append(Word(text, x0, y0, x1, y1, block, line))
    return words


def build_units(words: list[Word], regions, page_w: float, page_h: float) -> list[Unit]:
    units: dict[tuple, Unit] = {}
    for w in words:
        cx, cy = (w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2
        best = None
        for idx, (kind, r) in enumerate(regions):
            if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                if best is None or _area(r) < _area(regions[best][1]):
                    best = idx
        if best is not None:
            key, kind = ("r", best), regions[best][0]
        else:
            key, kind = ("b", w.block), "text"
        units.setdefault(key, Unit(kind)).words.append(w)

    out = []
    for key, u in units.items():
        u.fit()
        if key[0] == "b":
            n_chars = sum(len(w.text) for w in u.words)
            # Vertical text (e.g. the arXiv stamp in the left margin).
            vertical = sum(1 for w in u.words if len(w.text) > 3 and (w.y1 - w.y0) > 2 * (w.x1 - w.x0))
            if vertical > len(u.words) / 2:
                u.kind = "margin"
            # Short stray text at the very top/bottom: page numbers, running heads.
            elif n_chars <= 60 and (u.y1 < 0.06 * page_h or u.y0 > 0.94 * page_h):
                u.kind = "margin"
        out.append(u)
    return out


# --------------------------------------------------------------------------
# reading order (recursive XY-cut that keeps columns together)

X_GAP = 5.0  # minimum horizontal white gap (pt) between two columns


def _x_split(units: list[Unit]):
    us = sorted(units, key=lambda u: u.x0)
    groups, cur, right = [], [us[0]], us[0].x1
    for u in us[1:]:
        if u.x0 > right + X_GAP:
            groups.append(cur)
            cur = [u]
        else:
            cur.append(u)
        right = max(right, u.x1)
    groups.append(cur)
    return groups if len(groups) > 1 else None


def _y_segments(units: list[Unit]):
    us = sorted(units, key=lambda u: u.y0)
    segs, cur, bottom = [], [us[0]], us[0].y1
    for u in us[1:]:
        if u.y0 > bottom:
            segs.append(cur)
            cur = [u]
        else:
            cur.append(u)
        bottom = max(bottom, u.y1)
    segs.append(cur)
    return segs


def reading_order(units: list[Unit]) -> list[Unit]:
    def rec(us):
        if len(us) <= 1:
            return list(us)
        cols = _x_split(us)
        if cols:
            return [u for c in cols for u in rec(c)]
        segs = _y_segments(us)
        if len(segs) == 1:
            return sorted(us, key=lambda u: (u.y0, u.x0))
        # Glue consecutive horizontal bands together as long as the result still
        # splits into columns; full-width elements (title, wide figure) end a band.
        groups, cur = [], segs[0]
        for s in segs[1:]:
            if _x_split(cur + s):
                cur = cur + s
            else:
                groups.append(cur)
                cur = s
        groups.append(cur)
        if len(groups) == 1:
            return sorted(us, key=lambda u: (u.y0, u.x0))
        return [u for g in groups for u in rec(g)]

    return rec(units)


# --------------------------------------------------------------------------
# units -> sentences

ABBREV = {
    "e.g.", "i.e.", "etc.", "fig.", "figs.", "eq.", "eqs.", "eqn.", "sec.", "sect.", "tab.", "al.",
    "vs.", "cf.", "dr.", "prof.", "mr.", "mrs.", "ms.", "no.", "nos.", "ref.", "refs.", "approx.",
    "resp.", "ca.", "pp.", "vol.", "ch.", "chap.", "st.", "w.r.t.", "a.k.a.", "viz.", "inc.", "ltd.",
    "jr.", "sr.", "thm.", "def.", "prop.", "alg.", "app.", "ed.", "eds.", "dept.", "univ.",
    # Italian
    "es.", "ecc.", "pag.", "dott.", "ing.", "sig.", "art.", "cap.", "n.", "p.",
}
# Sentence-final punctuation, optionally followed by a footnote mark ("margin.12").
_END = re.compile(r"(?:[.!?…][\"')\]”’»]*|(?<=[A-Za-zÀ-ÿ)\]])[.!?][\"')\]”’»]*\d{1,3})$")
_STARTS_NEW = re.compile(r"^[\"'(\[“‘«]*[A-Z0-9À-ÖØ-Þ]")
MAX_LEN = 450  # characters; longer "sentences" are cut so synthesis stays snappy


@dataclass
class Token:
    text: str
    parts: list[tuple[int, Word]]  # (row index, word) — two parts when dehyphenated


def _rows(words: list[Word]) -> list[list[Word]]:
    """Group a unit's words into visual rows, top to bottom, left to right."""
    lines = defaultdict(list)
    for w in words:
        lines[(w.block, w.line)].append(w)
    boxes = []
    for ws in lines.values():
        boxes.append([min(w.y0 for w in ws), max(w.y1 for w in ws), ws])
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
    rows = []
    for y0, y1, ws in boxes:
        if rows:
            r = rows[-1]
            overlap = min(r[1], y1) - max(r[0], y0)
            if overlap > 0.5 * min(r[1] - r[0], y1 - y0):
                r[0], r[1] = min(r[0], y0), max(r[1], y1)
                r[2].extend(ws)
                continue
        rows.append([y0, y1, list(ws)])
    return [sorted(r[2], key=lambda w: w.x0) for r in rows]


def _tokens(unit: Unit) -> list[Token]:
    toks: list[Token] = []
    join_next = False
    for ri, row in enumerate(_rows(unit.words)):
        for wi, w in enumerate(row):
            if join_next and w.text[:1].islower() and "-" not in w.text:
                toks[-1].text = toks[-1].text[:-1] + w.text
                toks[-1].parts.append((ri, w))
            else:
                toks.append(Token(w.text, [(ri, w)]))
            join_next = False
        last = row[-1].text
        join_next = len(last) > 2 and last.endswith("-") and last[-2].isalpha()
    return toks


def _is_sentence_end(tok: Token, nxt: Token | None, first_in_sentence: bool) -> bool:
    t = tok.text
    if not _END.search(t):
        return False
    if nxt is None:
        return True
    core = t.rstrip("0123456789").rstrip("\"')]”’»")
    low = core.lower()
    if low in ABBREV:
        return False
    if core.endswith("."):
        stem = core[:-1]
        if len(stem) == 1 and stem.isalpha():  # initials: "A. Vaswani"
            return False
        if first_in_sentence and stem.isdigit():  # list numbering: "1. First item"
            return False
    return bool(_STARTS_NEW.match(nxt.text))


def _chunk(toks: list[Token]) -> list[list[Token]]:
    """Cut an over-long token run at commas/semicolons (or bluntly if none)."""
    out, cur, n = [], [], 0
    for t in toks:
        cur.append(t)
        n += len(t.text) + 1
        if (n > 220 and t.text[-1:] in ",;:") or n > MAX_LEN:
            out.append(cur)
            cur, n = [], 0
    if cur:
        out.append(cur)
    return out


def _sentences(unit: Unit) -> list[list[Token]]:
    toks = _tokens(unit)
    if unit.kind == "heading":
        return [toks] if toks else []
    sents, cur = [], []
    for i, t in enumerate(toks):
        cur.append(t)
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if _is_sentence_end(t, nxt, len(cur) == 1):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    out = []
    for s in sents:
        if sum(len(t.text) + 1 for t in s) > MAX_LEN:
            out.extend(_chunk(s))
        else:
            out.append(s)
    return out


def _item(kind: str, toks: list[Token], pw: float, ph: float) -> dict:
    rows: dict[int, list[float]] = {}
    for t in toks:
        for ri, w in t.parts:
            r = rows.get(ri)
            if r is None:
                rows[ri] = [w.x0, w.y0, w.x1, w.y1]
            else:
                r[0], r[1], r[2], r[3] = min(r[0], w.x0), min(r[1], w.y0), max(r[2], w.x1), max(r[3], w.y1)
    rects = [
        [round(r[0] / pw, 5), round(r[1] / ph, 5), round((r[2] - r[0]) / pw, 5), round((r[3] - r[1]) / ph, 5)]
        for _, r in sorted(rows.items())
    ]
    return {"k": kind, "t": " ".join(t.text for t in toks), "r": rects}


# --------------------------------------------------------------------------
# public API


def analyze_page(page: fitz.Page) -> dict:
    pw, ph = page.rect.width, page.rect.height
    try:
        regions = detect_regions(page)
    except Exception as exc:  # the detector failing must not make the page unreadable
        print(f"[layout] detection failed on page {page.number + 1}: {exc!r}")
        regions = []
    units = build_units(extract_words(page), regions, pw, ph)

    body = [u for u in units if u.kind != "margin"]
    margin = sorted((u for u in units if u.kind == "margin"), key=lambda u: (u.y0, u.x0))
    top = [u for u in margin if (u.y0 + u.y1) / 2 < ph / 2]
    bottom = [u for u in margin if (u.y0 + u.y1) / 2 >= ph / 2]
    ordered = top + (reading_order(body) if body else []) + bottom

    items = []
    for u in ordered:
        sents = _sentences(u)
        for i, s in enumerate(sents):
            it = _item(u.kind, s, pw, ph)
            if i == 0:
                it["f"] = 1  # first sentence of its paragraph
            if i == len(sents) - 1:
                it["l"] = 1  # last sentence of its paragraph
            items.append(it)

    # Regions shaded in the UI; a sub-figure nested in a bigger figure is not drawn twice.
    boxes = [(k, r) for k, r in regions if k in ("figure", "table", "formula")]
    shown = [
        {"k": k, "r": [round(r[0] / pw, 5), round(r[1] / ph, 5), round((r[2] - r[0]) / pw, 5), round((r[3] - r[1]) / ph, 5)]}
        for k, r in boxes
        if not any(o is not r and _area(o) > _area(r) > 0 and _inter(r, o) / _area(r) > 0.8 for _, o in boxes)
    ]
    return {"w": pw, "h": ph, "items": items, "regions": shown}


_STOP = {
    "en": "the of and to in is that for with as are this be by on which we from".split(),
    "it": "il di che la e un per è non con sono una del della le dei nel si come".split(),
    "fr": "le de la et les des est un une pour dans que qui sur par nous avec".split(),
    "es": "el de la que y en los las un una por con para es del se como".split(),
    "pt": "o de a que e do da em um uma para com não os as é dos se".split(),
}


def detect_language(doc: fitz.Document) -> str:
    text = " ".join(doc[i].get_text() for i in range(min(4, doc.page_count))).lower()
    counts = defaultdict(int)
    for w in re.findall(r"[a-zà-ÿ]+", text):
        counts[w] += 1
    scores = {lang: sum(counts[w] for w in ws) for lang, ws in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"
