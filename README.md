# OpenTTS — read PDFs aloud, locally

A small web app that reads PDFs with a neural voice. Everything runs locally; nothing is sent to a cloud service.

## Install

```bash
git clone https://github.com/SamueleGerminiani/opentts.git
cd opentts
./install.sh
```

The script installs [uv](https://docs.astral.sh/uv/) if it is missing (it asks first), sets up
Python 3.12 and the dependencies in `.venv`, downloads the models (~370 MB) into `models/`, and on
macOS builds the **OpenTTS** app in `~/Applications`. It is safe to run again.

## Run

On macOS, click **OpenTTS** in `~/Applications` (also in Launchpad and Spotlight; drag it to the Dock
to keep it there). It starts the local server if needed and opens the app in your browser. The server
stops by itself a few seconds after the last OpenTTS tab is closed; its log is `~/Library/Logs/OpenTTS.log`.
If the folder is moved, rebuild the launcher with `./macos/make_app.sh`.

From a terminal (any platform): `./run.sh`, which runs until you press Ctrl-C.

## Use

- **Open PDF** (or drop a file on the window).
- **Click any sentence** to start reading from there. The sentence being read is highlighted and kept in view.
- **Play / pause**, **stop**, previous / next sentence.
- **Follow** button: when on (the default), the PDF scrolls automatically to keep the sentence being read
  in view. Scrolling by hand pauses this until you press "Back to reading position" or click a sentence.
- **Speed** slider, 0.5× to 3×, changes immediately (pitch is preserved).
- **Voice** menu: English (US/UK), Italian, French, Spanish, Portuguese. The voice
  switches to the document's language automatically and remembers your choice per language.
- **Ignore** menu: figures, tables, captions, displayed formulas, headers/footers/footnotes,
  citation marks like `[12]`, text in `(round brackets)`. Ignored areas are shaded on the page; clicking one still reads it.
- The reading position of each document is remembered.

Keyboard: `Space` play/pause · `←` `→` previous/next sentence · `Esc` stop · `-` `+` speed.

## How it works

| Part | Tool |
|---|---|
| Speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (Apache-2.0), on CPU |
| Figure / table / caption / header detection | [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) (AGPL-3.0) |
| PDF text, vector (SVG) page rendering | [PyMuPDF](https://pymupdf.readthedocs.io) (AGPL-3.0) |
| Server / UI | FastAPI + plain HTML/JS |

- `pdfproc.py` — layout detection, reading order (column-aware XY-cut), sentence splitting.
- `app.py` — HTTP API, speech synthesis and caching.
- `static/` — the interface.
- `data/` — uploaded PDFs and cached page analysis (safe to delete).

## Limits

- Scanned PDFs without a text layer need OCR first (e.g. `ocrmypdf in.pdf out.pdf`).
- Inline maths is read as extracted text, which can sound odd.
- Japanese and Chinese Kokoro voices are not enabled.

## License

[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only).
