"""Convert the tracked percent-format lab source into an editable notebook."""
from pathlib import Path
import sys
import nbformat

directory = Path(__file__).resolve().parent
source = (directory / "research.py").read_text(encoding="utf-8")
cells = []
for block in source.split("# %%")[1:]:
    if block.startswith(" [markdown]"):
        lines = block.splitlines()[1:]
        text = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in lines).strip()
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(block.strip()))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3 (heart-disease-eda)", "language": "python"},
    "language_info": {"name": "python", "version": "3.10"},
})
if "--keep-outputs" in sys.argv:
    previous = nbformat.read(directory / "research.ipynb", as_version=4)
    assert len(previous.cells) == len(cells)
    notebook.metadata = previous.metadata
    for current, old in zip(notebook.cells, previous.cells):
        if current.cell_type == "code":
            assert current.source == old.source, "Changed code must be executed again"
            current.outputs = old.outputs
            current.execution_count = old.execution_count
nbformat.write(notebook, directory / "research.ipynb")
print(f"Created research.ipynb: {len(cells)} cells")
