"""Draw the OpenTTS app icon (1024x1024 PNG): a page with a play button and sound waves."""
import sys

from PIL import Image, ImageDraw

S = 4  # supersampling
N = 1024 * S
img = Image.new("RGBA", (N, N), (0, 0, 0, 0))

# Rounded-square background with a vertical gradient (macOS icon grid: 824px body).
grad = Image.new("RGBA", (N, N))
gd = ImageDraw.Draw(grad)
top, bottom = (86, 130, 255), (40, 70, 200)
for y in range(N):
    t = y / N
    gd.line([(0, y), (N, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
mask = Image.new("L", (N, N), 0)
ImageDraw.Draw(mask).rounded_rectangle([100 * S, 100 * S, 924 * S, 924 * S], radius=186 * S, fill=255)
img.paste(grad, (0, 0), mask)

d = ImageDraw.Draw(img)
# Page
d.rounded_rectangle([250 * S, 220 * S, 610 * S, 804 * S], radius=36 * S, fill=(255, 255, 255, 255))
for i, w in enumerate([240, 240, 200, 240, 150]):
    y = (300 + i * 62) * S
    d.rounded_rectangle([310 * S, y, (310 + w) * S, y + 26 * S], radius=13 * S, fill=(196, 208, 240, 255))
# Play button
d.ellipse([330 * S, 590 * S, 490 * S, 750 * S], fill=(49, 87, 213, 255))
d.polygon([(388 * S, 630 * S), (388 * S, 710 * S), (452 * S, 670 * S)], fill=(255, 255, 255, 255))
# Sound waves
for r, width in ((90, 34), (170, 34), (250, 34)):
    cx, cy = 610 * S, 512 * S
    d.arc([cx - r * S, cy - r * S, cx + r * S, cy + r * S], start=-42, end=42, fill=(255, 255, 255, 255), width=width * S)

img.resize((1024, 1024), Image.LANCZOS).save(sys.argv[1])
