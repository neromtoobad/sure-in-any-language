"""Render PDF pages to PNG for a visual check.  usage: python report/pages.py report/draft.pdf outdir [dpi]"""
import sys, fitz
doc = fitz.open(sys.argv[1]); dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 80
for i, p in enumerate(doc, 1):
    p.get_pixmap(dpi=dpi).save(f"{sys.argv[2]}/p{i}.png")
print(len(doc), "pages")
