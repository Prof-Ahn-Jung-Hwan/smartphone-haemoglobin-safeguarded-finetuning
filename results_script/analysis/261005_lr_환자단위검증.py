# -*- coding: utf-8 -*-
"""**학습률 선택을 환자 단위 검증 MAE 로 다시 낸다** — 백본 10종 · 가장 단순한 머리(경로 A).  2026-10-05

왜  학습률 격자(`260913_grid.py lr-pick`)는 학습 로그의 *Best validation MAE* — **사진 단위** — 로 골랐다.
    본문은 모두 환자 단위(두 눈 예측의 평균)로 보고하므로 선별도 같은 단위로 맞춘다 (교수님 10-05).
    검증 겹의 사진별 예측은 저장되지 않았으나 **체크포인트(best.ckpt)와 검증 명단(val.txt)이 모두 있다**
    → 재학습 없이 순전파만 한다. 결과는 체크포인트가 같으므로 학습을 다시 하는 것과 같다.

무엇  main/results/grid/train/g2_<백본>_A_lr<값>_<입력>-fold<겹>  (백본 10 × lr 5–6 × 입력 2 × 겹 3)
      검증 사진 → 예측 → 명부의 number 로 묶어 **|두 눈 예측의 평균 − Hb|** 의 평균 = 환자 단위 검증 MAE
      (백본 · lr) 마다 6칸(입력 2 × 겹 3) 평균 → 가장 낮은 lr

내는 것  main/results/grid/val_pred/<칸>.csv          (사진별 예측 · 다시 계산하지 않게)
         main/results/grid/lr_선택_환자단위.csv     (백본 · lr · 사진단위 · 환자단위 · 고른 lr · 바뀌었나)
쓰는 법  OMP_NUM_THREADS=4 python paperwork/results_script/analysis/261005_lr_환자단위검증.py
"""
import importlib.util, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]; MAIN = ROOT / "main"; HERE = MAIN / "analysis"   # 10-20 : paperwork/results_script/analysis/ 로 옮김
GRID = MAIN / "results/grid"; TRAIN = GRID / "train"; VP = GRID / "val_pred"; VP.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("ext", MAIN / "external/260918_external_grid.py")
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
PAT = re.compile(r"g2_(\w+?)_A_lr([\de\-.]+)_(auto_\w+)-fold(\d+)$")


def main():
    import torch
    ext.setup(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    명 = pd.read_csv(ROOT / "data/internal/명부.csv", usecols=["stem", "number"]); 번호 = dict(zip(명.stem.astype(str), 명.number))
    칸 = sorted(d for d in TRAIN.glob("g2_*_A_*-fold*") if (d / "best.ckpt").exists())
    rows = []
    for i, d in enumerate(칸):
        m = PAT.match(d.name); bb, lr, inp, f = m.group(1), float(m.group(2)), m.group(3), int(m.group(4))
        out = VP / f"{d.name}.csv"
        if not out.exists():
            v = pd.read_csv(d / "val.txt", sep="\t", header=None, names=["full_path", "Hb"])
            v["key"] = v.full_path.map(lambda p: Path(p).stem); v["Country"] = "internal"
            name, fold = d.name.rsplit("-fold", 1)
            args = ext.build_args(name, int(fold), dev)
            model = ext.load_model(args, d / "best.ckpt")
            ext.predict(args, model, v).to_csv(out, index=False)
            del model; torch.cuda.empty_cache()
        p = pd.read_csv(out); p["n"] = p.key.astype(str).map(번호)
        if p.n.isna().any(): sys.exit(f"🔴 {d.name} : 명부에 없는 사진 {int(p.n.isna().sum())}장")
        사진 = float((p.pred - p.y).abs().mean())
        g = p.groupby("n").agg(pred=("pred", "mean"), y=("y", "first")); 환자 = float((g.pred - g.y).abs().mean())
        rows.append(dict(백본=bb, lr=lr, 입력=inp, 겹=f, 사진단위=사진, 환자단위=환자, 환자수=len(g)))
        if (i + 1) % 20 == 0: print(f"  {i+1}/{len(칸)}", flush=True)
    T = pd.DataFrame(rows)
    S = T.groupby(["백본", "lr"]).agg(사진단위=("사진단위", "mean"), 환자단위=("환자단위", "mean"), 칸수=("겹", "size")).reset_index()
    옛 = pd.read_csv(GRID / "lr_선택.csv"); 옛.columns = [c.lstrip("﻿") for c in 옛.columns]
    옛 = 옛[옛["경로"] == "A"].set_index("백본")
    고름 = []
    for bb, g in S.groupby("백본"):
        b = g.loc[g.환자단위.idxmin()]
        고름.append(dict(백본=bb, 고른lr_환자단위=b.lr, 환자단위MAE=round(b.환자단위, 4), 사진단위MAE=round(b.사진단위, 4),
                       옛lr=옛.loc[bb, "lr"], 옛_사진단위MAE=round(float(옛.loc[bb, "검증MAE"]), 4),
                       바뀜=bool(abs(b.lr - 옛.loc[bb, "lr"]) > 1e-12), 칸수=int(b.칸수)))
    G = pd.DataFrame(고름).sort_values("환자단위MAE")
    S.to_csv(GRID / "lr_격자_환자단위.csv", index=False, encoding="utf-8-sig")
    G.to_csv(GRID / "lr_선택_환자단위.csv", index=False, encoding="utf-8-sig")
    print(G.to_string(index=False))
    print(f"→ {GRID / 'lr_선택_환자단위.csv'}  · 바뀐 백본 {int(G.바뀜.sum())}개")


if __name__ == "__main__":
    main()
