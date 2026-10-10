"""Build the Colab notebook from the percent-cell pipeline script (single source of truth)."""
import re
import sys

import nbformat

src, dst = sys.argv[1], sys.argv[2]
cells = []
for chunk in re.split(r"^# %%", open(src, encoding="utf-8").read(), flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if header.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                         for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3"},
                    "colab": {"provenance": []}})
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")
