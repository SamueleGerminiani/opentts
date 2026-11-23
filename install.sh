#!/bin/bash
# Set up OpenTTS after cloning the repository:
#   1. make sure uv (the Python package manager) is available
#   2. install Python 3.12 and the dependencies into .venv
#   3. download the speech and layout models into models/ (~370 MB)
#   4. on macOS, build the clickable OpenTTS app in ~/Applications
#
# Usage: ./install.sh            (set OPENTTS_APP_DIR to put the macOS app elsewhere)
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
die() { printf '\033[31mError:\033[0m %s\n' "$1" >&2; exit 1; }

# ---- 1. uv ---------------------------------------------------------------
step "Checking for uv"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. It is needed to install Python and the dependencies."
  if command -v brew >/dev/null 2>&1; then
    HOW="brew install uv"
  else
    HOW="curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
  if [ -t 0 ]; then
    read -r -p "Install it now with '$HOW'? [Y/n] " answer
    case "$answer" in [nN]*) die "uv is required. See https://docs.astral.sh/uv/ and run this script again." ;; esac
  else
    echo "Installing it with: $HOW"
  fi
  command -v curl >/dev/null 2>&1 || command -v brew >/dev/null 2>&1 || die "curl is required to install uv."
  eval "$HOW"
  command -v uv >/dev/null 2>&1 || die "uv was installed but is not on PATH. Open a new terminal and run this script again."
fi
echo "uv: $(uv --version)"

# ---- 2. dependencies -----------------------------------------------------
step "Installing Python 3.12 and dependencies (this can take a few minutes)"
uv sync --python 3.12

# ---- 3. models -----------------------------------------------------------
step "Downloading models (~370 MB, skipped if already present)"
uv run python - <<'PY'
import warnings

warnings.filterwarnings("ignore")
import app
import pdfproc

app.get_kokoro()
print("speech model ready:", app.KOKORO_MODEL.name)
pdfproc.load_layout_model()
print("layout model ready:", pdfproc.LAYOUT_FILE)
PY

# ---- 4. launcher ---------------------------------------------------------
if [ "$(uname)" = "Darwin" ]; then
  step "Building the macOS app"
  ./macos/make_app.sh "${OPENTTS_APP_DIR:-$HOME/Applications}"
  printf '\n\033[32mDone.\033[0m Open OpenTTS from Launchpad or Spotlight, or run ./run.sh\n'
else
  printf '\n\033[32mDone.\033[0m Start OpenTTS with ./run.sh (it opens http://127.0.0.1:8765)\n'
fi
