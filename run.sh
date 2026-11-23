#!/bin/bash
# Start OpenTTS and open it in the browser (http://127.0.0.1:8765).
cd "$(dirname "$0")" && exec uv run python app.py
