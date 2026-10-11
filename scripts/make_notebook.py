"""Build the Colab notebook from the percent-cell pipeline script (single source of truth).

aqi_cpcb.py is inlined as a code cell before the pipeline's first code cell, so the
notebook needs no extra upload.
"""
import re
import sys

import nbformat

src, dst = sys.argv[1], sys.argv[2]
AQI_MODULE = "aqi_cpcb.py"
IMPORT_LINE = "from aqi_cpcb import AQI_EDGES, CATEGORIES, daily_aqi\n"
code = open(src, encoding="utf-8").read()
assert IMPORT_LINE in code, f"{src} no longer imports {AQI_MODULE} as expected"
code = code.replace(IMPORT_LINE, "")
cells = []
for chunk in re.split(r"^# %%", code, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if header.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                         for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        if not any(c.cell_type == "code" for c in cells):
            aqi = open(AQI_MODULE, encoding="utf-8").read().strip()
            cells.append(nbformat.v4.new_code_cell(f"# {AQI_MODULE} (inlined)\n{aqi}"))
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "colab": {"provenance": []}})
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")
