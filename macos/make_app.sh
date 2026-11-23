#!/bin/bash
# Build OpenTTS.app — a clickable launcher — into ~/Applications (or the folder given as $1).
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/OpenTTS.app"
UV="$(command -v uv || true)"

cd "$PROJECT"
[ -x .venv/bin/python ] || "$UV" sync

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Icon
TMP="$(mktemp -d)"
.venv/bin/python macos/make_icon.py "$TMP/icon.png"
mkdir "$TMP/icon.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon.png" --out "$TMP/icon.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$TMP/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$TMP/icon.iconset" -o "$APP/Contents/Resources/icon.icns"
rm -rf "$TMP"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>OpenTTS</string>
  <key>CFBundleDisplayName</key><string>OpenTTS</string>
  <key>CFBundleIdentifier</key><string>local.opentts.launcher</string>
  <key>CFBundleExecutable</key><string>OpenTTS</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
PLIST

# The launcher: start the server if it is not running yet, then open the browser.
cat > "$APP/Contents/MacOS/OpenTTS" <<LAUNCH
#!/bin/bash
PROJECT="$PROJECT"
PORT=8765
URL="http://127.0.0.1:\$PORT"
LOG="\$HOME/Library/Logs/OpenTTS.log"

fail() { osascript -e "display alert \"OpenTTS\" message \"\$1\" as critical" >/dev/null; exit 1; }
up() { curl -s -o /dev/null --max-time 1 "\$URL/api/ping"; }

if ! up; then
  cd "\$PROJECT" 2>/dev/null || fail "The OpenTTS folder was not found at \$PROJECT. Run macos/make_app.sh again from its new location."
  [ -x .venv/bin/python ] || fail "The Python environment is missing. Run 'uv sync' in \$PROJECT."
  # App mode: the server stops a few seconds after the last OpenTTS tab is closed
  # (or after 15 minutes if no tab ever opened).
  OPENTTS_NO_BROWSER=1 OPENTTS_IDLE_EXIT=900 OPENTTS_PORT=\$PORT nohup .venv/bin/python app.py >"\$LOG" 2>&1 &
  disown
  for _ in \$(seq 1 120); do up && break; sleep 0.25; done
  up || fail "OpenTTS did not start. See \$LOG for details."
fi
open "\$URL"
LAUNCH
chmod +x "$APP/Contents/MacOS/OpenTTS"
touch "$APP"
echo "Built $APP"
