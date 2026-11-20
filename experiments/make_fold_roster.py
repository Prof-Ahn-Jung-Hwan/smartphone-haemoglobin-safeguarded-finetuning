# -*- coding: utf-8 -*-
"""Fold roster of the development cohort: which participant was in the training, validation or test part of
each fold and seed, saved by study number, and a check that trained models used exactly these splits.

Splitting: participant level (both eyes of a participant in the same fold), stratified by integer haemoglobin;
outer folds fixed by OUTER_SEED; the inner validation set (one ninth of each training portion) drawn per seed.
Input   data/arms/internal_*.xlsx (one roster per input region; same participants and Hb in all)
Output  data/splits/fold_roster_10fold_inner9.csv (ten folds, seeds 1–10) · data/splits/fold_roster_5fold_inner8.csv (screen)
        columns: number · hb_int · seed · fold · part (train / validation / test). No photograph file names.
Usage   PYTHONPATH=code python experiments/make_fold_roster.py make                     # make (or check if present)
        PYTHONPATH=code python experiments/make_fold_roster.py check <train folders>    # compare with trained models
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from dataset.ajoumc_anemia import split_by_participant, OUTER_SEED

ROOT = Path(__file__).resolve().parent.parent
ARMS = ROOT / "data/arms"; OUT = ROOT / "data/splits"
ROSTERS = {"fold_roster_10fold_inner9.csv": (10, 9, range(1, 11)), "fold_roster_5fold_inner8.csv": (5, 8, [1])}
PARTS = (("train", 0), ("validation", 1), ("test", 2))


def participants(f=ARMS / "internal_auto_scca_c224.xlsx"):
    """Same rule as the loader (eye: exclusive): participants with Hb and both eyes -> {number: integer Hb}."""
    d = pd.read_excel(f).dropna(subset=["Hb"])
    both = d.groupby("number").separator.agg(lambda s: {1, 2} <= set(s))
    d = d[d.number.isin(both[both].index)]
    if d.groupby("number").Hb.nunique().max() > 1: sys.exit("a participant has photographs with different Hb values")
    return {int(n): int(h) for n, h in d.groupby("number").Hb.first().items()}


def make(_):
    P = participants()
    for f in sorted(ARMS.glob("internal_*.xlsx")):
        if participants(f) != P: sys.exit(f"{f.name}: participants or Hb differ from the reference roster")
    print(f"  {len(P)} participants · {len(list(ARMS.glob('internal_*.xlsx')))} input rosters agree · outer seed {OUTER_SEED}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (cv, K, seeds) in ROSTERS.items():
        rows = []
        for s in seeds:
            for f, sets in enumerate(split_by_participant(P, cv, s, K)):
                for part, i in PARTS:
                    rows += [(n, P[n], s, f, part) for n in sorted(sets[i])]
        T = pd.DataFrame(rows, columns=["number", "hb_int", "seed", "fold", "part"])
        verify(T, P, cv)
        dst = OUT / name
        if dst.exists():
            old = pd.read_csv(dst)
            same = old.equals(T)
            print(f"  {name} exists → {'identical' if same else 'DIFFERENT — not overwritten'}")
            if not same: sys.exit(1)
        else:
            T.to_csv(dst, index=False); print(f"  saved {dst.relative_to(ROOT)} ({len(T)} rows)")


def verify(T, P, cv):
    """(1) one part per participant per fold; (2) each participant tested once per seed; (3) stratification;
    (4) outer folds identical across seeds; (5) inner validation sets differ between seeds."""
    y = pd.Series(P); strata = np.unique(y); overall = np.array([(y == c).mean() for c in strata])
    test = T[T.part == "test"]
    assert (T.groupby(["seed", "fold", "number"]).size() == 1).all(), "participant in two parts of one fold"
    assert (test.groupby(["seed", "number"]).size() == 1).all() and test.groupby("seed").number.nunique().eq(len(P)).all(), \
        "participant not tested exactly once"
    diff = [100 * np.abs(np.array([(g.hb_int == c).mean() for c in strata]) - overall).max() for _, g in test.groupby(["seed", "fold"])]
    outer = test.groupby("seed").apply(lambda g: frozenset(map(frozenset, g.groupby("fold").number.apply(set))), include_groups=False)
    inner = T[T.part == "validation"].groupby("seed").number.apply(frozenset)
    n = test.groupby(["seed", "fold"]).size()
    print(f"  {cv} folds · {T.seed.nunique()} seeds · test fold {n.min()}–{n.max()} participants · largest stratum difference "
          f"{np.mean(diff):.1f} %-points on average (max {max(diff):.1f}) · outer folds identical across seeds {outer.nunique() == 1} · "
          f"inner validation sets differ between seeds {inner.nunique() == len(inner)}")
    assert outer.nunique() == 1, "outer folds differ between seeds"


def check(a):
    """Compare trained-model folders (<cell>-fold<f>/{train,val,test}.txt and the seed in config.yaml) with the roster."""
    d = pd.concat([pd.read_excel(f) for f in ARMS.glob("internal_*.xlsx")])
    stem2n = {Path(str(p)).stem: int(n) for p, n in zip(d.file_path, d.number)}
    M = {k: pd.read_csv(OUT / k) for k in ROSTERS}
    good = bad = 0
    for top in a.dirs:
        for D in sorted(Path(top).glob("*-fold*")):
            if not (D / "test.txt").exists(): continue
            cfg = (D / "config.yaml").read_text()
            g = lambda k: re.search(rf"^{k}: (\S+)", cfg, re.M).group(1)
            cv, K, s, f = int(g("cv_fold")), int(g("val_inner")), int(g("seed")), int(g("fold"))  # config.yaml is written before seed += fold
            T = M[f"fold_roster_{cv}fold_inner{K}.csv"]; T = T[(T.seed == s) & (T.fold == f)]
            ok = True
            for part, fn in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
                actual = {stem2n[Path(l.split("\t")[0]).stem] for l in (D / fn).read_text().splitlines() if l.strip()} if (D / fn).exists() else set()
                ok &= actual == set(T[T.part == part].number)
            good += ok; bad += not ok
            if not ok: print(f"  DIFFERENT {D}")
    print(f"same as roster {good} · different {bad}"); sys.exit(1 if bad else 0)


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
sp.add_parser("make").set_defaults(func=make)
c = sp.add_parser("check"); c.add_argument("dirs", nargs="+"); c.set_defaults(func=check)
a = ap.parse_args(); a.func(a)
