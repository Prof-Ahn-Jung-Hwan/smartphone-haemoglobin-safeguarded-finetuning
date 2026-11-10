"""Local calibration at an external location and evaluation — the study's procedure on the demo subset.

Uses the study's own functions from main/external/260918_external_grid.py:
  finetune()  backbone frozen; the head (S1 LayerNorm affine + S2 adaptation layer + output) is retrained
              30 epochs, learning rate 2e-5, batch = n // 3, best epoch by training MAE
              freeze_extra = "bn" -> S3: backbone batch-normalisation statistics kept at development values
  predict()   prediction for the evaluation participants
Three settings are compared on the same evaluation participants:
  before calibration · calibrated with BN statistics updated (S1 + S2) · calibrated with SS (S1 + S2 + S3)
Output: demo/work/demo_results.csv and a printed summary.
The demo model is trained for 40 epochs on 24 images (fold 0 of 5), so the numbers only show that the pipeline runs.
"""
import sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "main/external/260918_external_grid.py"
src = EXT.read_text(encoding="utf-8").split("ap = argparse.ArgumentParser(")[0]      # functions only, no CLI
X = types.ModuleType("extgrid"); X.__dict__["__file__"] = str(EXT)
exec(compile(src, str(EXT), "exec"), X.__dict__)
X.TRAIN = ROOT / "demo/work/train"
NAME, FOLD = "demo_S1_S2", 0

r = pd.read_csv(ROOT / "demo/data/roster.csv")
r["full_path"] = [str(ROOT / f) for f in r.file]                    # absolute: setup() changes directory
r["Country"] = r.site
cal = r[(r.site == "Italy") & (r.role == "calibration")]
ev = r[(r.site == "Italy") & (r.role == "evaluation")]

X.setup()
args = X.build_args(NAME, FOLD, 0, epochs=30, lr=2e-5)
model = X.load_model(args, X.TRAIN / f"{NAME}-fold{FOLD}" / "best.ckpt")
init = {k: v.detach().clone() for k, v in model.state_dict().items()}

out = []
res = X.predict(args, model, ev); res["setting"] = "before calibration"; out.append(res)
for label, fe in [("calibrated, BN statistics updated (S1 + S2)", ""), ("calibrated with SS (S1 + S2 + S3)", "bn")]:
    X.finetune(args, model, init, cal, SimpleNamespace(freeze_extra=fe), seed=args.seed + FOLD)
    res = X.predict(args, model, ev); res["setting"] = label; out.append(res)
O = pd.concat(out, ignore_index=True)
O.to_csv(ROOT / "demo/work/demo_results.csv", index=False)
print(f"\nItalian demo subset: {len(cal)} calibration and {len(ev)} evaluation participants")
for s, g in O.groupby("setting", sort=False):
    print(f"  {s:46s} MAE {np.abs(g.pred - g.y).mean():.2f} g/dL")
print("wrote demo/work/demo_results.csv   (smoke test only — not study results)")
