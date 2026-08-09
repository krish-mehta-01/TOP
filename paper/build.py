"""
build.py — converts TOP_Research_Paper.md into a styled single-column HTML file
and renders it to PDF via headless Microsoft Edge. Re-run after any edit to the
markdown source; prints the final page count.

Usage: python build.py
"""

import os
import re
import subprocess
import sys

import markdown as md
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "TOP_Research_Paper.md")
HTML_PATH = os.path.join(HERE, "TOP_Research_Paper.html")
PDF_PATH = os.path.join(HERE, "TOP_Research_Paper.pdf")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page {
  size: A4;
  margin: 25mm 22mm 25mm 22mm;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #111111;
}
body {
  font-family: "Times New Roman", Times, "Liberation Serif", serif;
  font-size: 12pt;
  line-height: 1.5;
  text-align: justify;
  hyphens: auto;
}
.titleblock {
  text-align: center;
  margin-bottom: 6mm;
}
.titleblock h1 {
  font-size: 17pt;
  line-height: 1.3;
  margin: 0 0 4mm 0;
  font-weight: bold;
}
.titleblock .affil {
  font-size: 10.5pt;
  font-style: italic;
  color: #333;
}
h1.abstract-h { display: none; }
h2 {
  font-size: 12.5pt;
  font-weight: bold;
  text-transform: none;
  letter-spacing: 0.02em;
  margin: 7mm 0 3mm 0;
  page-break-after: avoid;
  border-bottom: 0.4pt solid #999;
  padding-bottom: 1mm;
}
h3 {
  font-size: 11.5pt;
  font-weight: bold;
  font-style: italic;
  margin: 5mm 0 2mm 0;
  page-break-after: avoid;
}
p { margin: 0 0 2.6mm 0; orphans: 3; widows: 3; }
strong { font-weight: bold; }
em { font-style: italic; }
hr { display: none; }
ol, ul { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1mm; }
code, pre {
  font-family: "Courier New", monospace;
  font-size: 9.5pt;
}
pre {
  background: #f4f4f4;
  border: 0.3pt solid #ccc;
  padding: 3mm;
  white-space: pre-wrap;
  margin: 3mm 0;
  page-break-inside: avoid;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 3mm 0 4mm 0;
  font-size: 9.3pt;
  page-break-inside: avoid;
}
caption, .table-caption {
  font-size: 10pt;
  font-weight: bold;
  text-align: left;
  margin-bottom: 1.5mm;
}
th, td {
  border: 0.4pt solid #666;
  padding: 1.3mm 2mm;
  text-align: left;
  vertical-align: top;
}
th { background: #e8e8e8; font-weight: bold; }
blockquote {
  margin: 3mm 6mm;
  font-size: 10.8pt;
  color: #333;
  border-left: 2pt solid #999;
  padding-left: 3mm;
}
.keywords { margin: 4mm 0 6mm 0; font-size: 11pt; }
.keywords b { font-style: normal; }
section.abstract p { font-size: 11pt; }
p img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 4mm auto 1mm auto;
  page-break-inside: avoid;
}
p:has(img) {
  text-align: center;
  page-break-inside: avoid;
  margin-bottom: 1mm;
}
p:has(img) + p {
  margin-top: 0;
}
.gallery {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3mm;
  margin: 3mm 0;
}
.gallery figure {
  margin: 0;
  page-break-inside: avoid;
}
.gallery img {
  width: 100%;
  height: auto;
  display: block;
  border: 0.3pt solid #ccc;
}
.gallery figcaption {
  font-size: 8.8pt;
  color: #333;
  text-align: center;
  margin-top: 1mm;
}
"""

HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TOP: Tactical Optimization During Time-Out Period</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def strip_draft_comment(text: str) -> str:
    return re.sub(r"<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)


def convert():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    text = strip_draft_comment(text)

    body_html = md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )

    full_html = HTML_SHELL.format(css=CSS, body=body_html)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Wrote {HTML_PATH} ({len(full_html):,} chars)")


def render_pdf():
    if os.path.exists(PDF_PATH):
        os.remove(PDF_PATH)
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print("Edge stderr:", result.stderr[-2000:])
        sys.exit(1)


def count_pages():
    for _ in range(20):
        if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
            break
        import time
        time.sleep(0.5)
    reader = PdfReader(PDF_PATH)
    n = len(reader.pages)
    print(f"PDF page count: {n}")
    return n


if __name__ == "__main__":
    convert()
    render_pdf()
    count_pages()
