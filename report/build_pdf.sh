#!/bin/zsh
# report.html (+ body.html, numbers filled by fill.py, inserted at <!--BODY-->) -> report.pdf with headless Chrome. Prints the page count.
cd "$(dirname "$0")"
../.venv/bin/python fill.py || echo "!! fill.py reported unknown tokens"
python3 - <<'E'
s = open('report.html', encoding='utf-8').read()
b = open('_body_filled.html', encoding='utf-8').read()
open('_build.html', 'w', encoding='utf-8').write(s.replace('<!--BODY-->', b))
E
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=15000 --print-to-pdf="$PWD/${1:-report}.pdf" "file://$PWD/_build.html" 2>/dev/null
python3 - "$PWD/${1:-report}.pdf" <<'E'
import re, sys
d = open(sys.argv[1], 'rb').read()
print(sys.argv[1], f"{len(d)/1e6:.2f} MB,", len(re.findall(rb'/Type\s*/Page[^s]', d)), "pages")
E
