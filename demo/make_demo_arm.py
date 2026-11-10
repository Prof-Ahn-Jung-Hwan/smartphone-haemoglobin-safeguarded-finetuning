"""Build the training roster (xlsx) that code/src/train_bins.py expects, from demo/data/roster.csv.

Columns expected by code/dataset/ajoumc_anemia.py:
  file_path  image path relative to `dataset_root` (the repository root)
  number     participant identifier (cross-validation is grouped by this)
  separator  eye (1 = left, 2 = right); the public images are one eye per participant
  Hb         laboratory haemoglobin, g/dL
Output: demo/work/arm_demo_train.xlsx (the India subset, role == "train")
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / "demo/data/roster.csv")
t = r[r.role == "train"].reset_index(drop=True)
arm = pd.DataFrame({"file_path": t.file, "number": range(1, len(t) + 1), "separator": 1, "Hb": t.Hb})
out = ROOT / "demo/work/arm_demo_train.xlsx"; out.parent.mkdir(parents=True, exist_ok=True)
arm.to_excel(out, index=False)
print(f"wrote {out.relative_to(ROOT)}  ({len(arm)} images)")
