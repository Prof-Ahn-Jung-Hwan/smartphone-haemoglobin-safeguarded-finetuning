# -*- coding: utf-8 -*-
"""External evaluation driver used in the study: runs `main/external/external_finetune.py` for the trained
ten-fold models of a configuration, at both locations, for every seed and draw.

Usage
    python main/analysis/external_evaluation.py zero                         # before fine-tuning
    python main/analysis/external_evaluation.py few --few-lr 1e-4            # local fine-tuning (S1, S2 as trained)
    python main/analysis/external_evaluation.py few --few-lr 1e-4 --freeze-extra bn --tag bn   # with S3
    (split the work with --shard <i> --nshard <k>)

Outputs  <experiment>/external/{zero, few…}/<location>/n30/rep<r>/<cell>/fold<f>.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

import os
HERE = Path(__file__).resolve().parent; MAIN = HERE.parent; ROOT = MAIN.parent
# EXT_TRAIN_ROOT: folder of trained development models (default main/results/stage5); EXT_CELL_PREFIX: model-name prefix
S5 = Path(os.environ.get("EXT_TRAIN_ROOT", MAIN / "results/stage5"))
PREFIX = os.environ.get("EXT_CELL_PREFIX", "s5_")
OUT = S5 / "external"; SPLIT = MAIN / "results/external/splits"
FEW_DEFAULT_LR = 2e-5          # results folder "few" = 2e-5; other rates go to few_lr<rate>


def few_root(a):
    """Output folder for fine-tuned predictions: few_lr<rate>[_<tag>] (e.g. few_lr0.0001, few_lr0.0001_bn for S3)."""
    lr = float(getattr(a, "few_lr", FEW_DEFAULT_LR))
    name = "few" if abs(lr - FEW_DEFAULT_LR) < 1e-12 else f"few_lr{lr:g}"
    tag = str(getattr(a, "tag", "") or "")
    return OUT / (name if not tag else f"{name}_{tag}")
MR = MAIN / "results/main_results"; SITES = ["India", "Italy"]; CV = 10
INPUTS = ["auto_scca", "auto_bc", "auto_bcwb", "human_scca", "human_bc"]

import types
src = (MAIN / "external/external_finetune.py").read_text()
src = src.split("if __name__ == \"__main__\":")[0]          # functions only, without the command line
X = types.ModuleType("extgrid"); X.__dict__["__file__"] = str(MAIN / "external/external_finetune.py")
exec(compile(src, str(MAIN / "external/external_finetune.py"), "exec"), X.__dict__)
X.TRAIN = S5 / "train"                                       # trained development models


def inp_of(name):                                            # s5_mnl_ve_K40_human_scca_seed2 → human_scca
    for i in INPUTS:
        if f"_{i}_" in name: return i
    sys.exit(f"cannot read the input region from {name}")


def cells(a=None):
    """Models whose ten folds have all finished training (safe to call while training is still running)."""
    s = {}
    for d in (S5 / "train").glob(f"{PREFIX}*-fold*"):
        if (d / "best.ckpt").exists() and (d / "training_time.txt").exists():
            e, f = d.name.rsplit("-fold", 1); s.setdefault(e, set()).add(int(f))
    out = sorted(e for e, v in s.items() if v == set(range(CV)))
    if a is not None:
        if a.heads:  out = [e for e in out if e.split("_")[2] in a.heads]
        if a.inputs: out = [e for e in out if inp_of(e) in a.inputs]
        if a.seeds:  out = [e for e in out if int(e.rsplit("seed", 1)[1]) in a.seeds]
        out = [x for i, x in enumerate(out) if i % a.nshard == a.shard]
    return out


def cmd_zero(a):
    X.setup(); import torch
    for name in cells(a):
        df = X.read_ext(inp_of(name))
        for f in range(CV):
            dst = OUT / "zero" / name / f"fold{f}.csv"
            if dst.exists(): continue
            args = X.build_args(name, f, a.device)
            m = X.load_model(args, X.TRAIN / f"{name}-fold{f}" / "best.ckpt")
            r = X.predict(args, m, df); dst.parent.mkdir(parents=True, exist_ok=True); r.to_csv(dst, index=False)
            del m; torch.cuda.empty_cache()
        print(f"  [zero] {name}", flush=True)


def cmd_few(a):
    X.setup(); import torch, time
    t0 = time.time(); done = 0
    for name in cells(a):
        df = X.read_ext(inp_of(name)).set_index("key", drop=False)
        for f in range(CV):
            FEW = few_root(a)
            _sites = [s for s in SITES if (not a.sites or s in a.sites)]
            todo = [(s, r, n) for s in _sites for r in range(a.reps) for n in a.ns
                    if not (FEW / s / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv").exists()]
            if not todo: continue
            args = X.build_args(name, f, a.device, epochs=a.epochs, lr=a.few_lr)
            model = X.load_model(args, X.TRAIN / f"{name}-fold{f}" / "best.ckpt")
            init = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for s, r, n in todo:
                d = SPLIT / s / f"rep{r}"
                ev = df.loc[pd.read_csv(d / "eval.csv").key]
                cal = df.loc[pd.read_csv(d / "calib_random.csv").key[:n]]
                X.finetune(args, model, init, cal, a, seed=args.seed + f + 100 * r + n)
                res = X.predict(args, model, ev)
                dst = FEW / s / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv"
                dst.parent.mkdir(parents=True, exist_ok=True); res.to_csv(dst, index=False)
                done += 1
            del model; torch.cuda.empty_cache()
        el = (time.time() - t0) / 60
        print(f"  [few] {name} done · {done} fine-tuning runs · {el:.0f} min · {60*el/max(done,1):.1f} s per run", flush=True)


ap = argparse.ArgumentParser(); sub = ap.add_subparsers(required=True)
for n_, f_ in [("zero", cmd_zero), ("few", cmd_few)]:
    s_ = sub.add_parser(n_); s_.set_defaults(func=f_)
    s_.add_argument("--device", type=int, default=0)
    s_.add_argument("--heads", nargs="*", default=None, help="model heads (cell names), default all")
    s_.add_argument("--inputs", nargs="*", default=None, help="input regions, default all")
    s_.add_argument("--seeds", type=int, nargs="*", default=None)
    s_.add_argument("--ns", type=int, nargs="+", default=[30], help="number of local fine-tuning participants")
    s_.add_argument("--reps", type=int, default=10, help="number of random draws")
    s_.add_argument("--epochs", type=int, default=30)
    s_.add_argument("--few-lr", dest="few_lr", type=float, default=1e-4)
    s_.add_argument("--freeze-extra", dest="freeze_extra", default="",
                    help="bn = keep backbone batch-normalisation statistics fixed during fine-tuning (S3)")
    s_.add_argument("--tag", default="", help="suffix of the output folder (few_lr<rate>_<tag>)")
    s_.add_argument("--sites", nargs="*", default=None, help="locations to run (default both)")
    s_.add_argument("--shard", type=int, default=0)
    s_.add_argument("--nshard", type=int, default=1)
a = ap.parse_args(); a.func(a)
