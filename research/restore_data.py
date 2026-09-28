"""Recreate lab-1 pickle with the current NumPy without modifying the EDA notebook."""
from pathlib import Path
import shutil
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / "data" / "clean_dataset.pkl"
backup = root / "data" / "clean_dataset_numpy2.pkl"
if path.exists() and not backup.exists():
    shutil.copy2(path, backup)

import os
os.chdir(root)
notebook = nbformat.read(root / "eda" / "eda.ipynb", as_version=4)
namespace = {}
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    exec(compile(cell.source, "eda/eda.ipynb", "exec"), namespace)
    if "clean_df.to_pickle" in cell.source:
        break
else:
    raise RuntimeError("The EDA notebook has no clean_df.to_pickle cell")

import pandas as pd
data = pd.read_pickle(path)
assert data.shape == (303, 15)
assert not data.isna().any().any()
print("Lab-1 data recreated; original backup:", backup)
