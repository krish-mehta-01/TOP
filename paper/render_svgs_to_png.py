"""
render_svgs_to_png.py -- rasterizes every figures/*.svg to a matching *.png
(same basename) via headless Microsoft Edge, so the Word-doc builder can embed
a raster image instead of an SVG (python-docx / Word do not support SVG).
"""

import glob
import os
import re
import subprocess
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

WRAPPER = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:#ffffff;}}</style></head>
<body><img src="{name}" style="display:block;width:{w}px;"></body></html>"""


def to_file_url(path: str) -> str:
    p = path.replace("\\", "/")
    return "file:///" + urllib.parse.quote(p, safe=":/")


def viewbox_dims(svg_path: str):
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"', content)
    if not m:
        return 1500.0, 500.0
    return float(m.group(1)), float(m.group(2))


def render(svg_path: str, scale: float = 2.0):
    base = os.path.splitext(os.path.basename(svg_path))[0]
    vw, vh = viewbox_dims(svg_path)
    out_w = int(vw * scale / 1.6)  # 1.6 keeps raster width reasonable (~1900px for a 1500-wide viewBox)
    out_h = int(vh / vw * out_w) + 40  # small margin so the last row isn't clipped

    wrapper_path = os.path.join(FIG_DIR, f"_{base}_wrapper.html")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(WRAPPER.format(name=os.path.basename(svg_path), w=out_w))

    png_path = os.path.join(FIG_DIR, f"{base}.png")
    if os.path.exists(png_path):
        os.remove(png_path)

    url = to_file_url(wrapper_path)
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={out_w + 20},{out_h}",
        f"--screenshot={png_path}",
        "--default-background-color=FFFFFFFF",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    os.remove(wrapper_path)
    if not os.path.exists(png_path):
        print("FAILED:", svg_path, result.stderr[-500:])
    else:
        print("OK:", png_path)


if __name__ == "__main__":
    for svg in sorted(glob.glob(os.path.join(FIG_DIR, "*.svg"))):
        render(svg)
