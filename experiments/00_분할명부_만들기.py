# -*- coding: utf-8 -*-
"""**분할 명부** — 개발 코호트를 누가 어느 겹의 학습 · 검증 · 시험에 들었는지 연구 번호로 저장하고 확인한다  2026-10-28

왜    학습 로더 (code/dataset/ajoumc_anemia.py 의 사람분할) 는 분할을 그때그때 계산한다. 명부를 파일로 남겨
      ① 심사 · 재현 요청에 그대로 내고 ② 학습이 실제로 그 명부대로 나눴는지 대조한다.
      분할 방식 : 사람 단위 층화 무작위 (정수 Hb) · 바깥 겹은 OUTER_SEED 로 고정 · 안쪽 검증은 seed 마다
원자료 data/arms/internal_auto_scca_c224.xlsx (사람 · Hb 는 모든 입력 명부가 같다 — 만들기가 확인한다)
내는 것 data/splits/분할명부_10겹_안쪽9.csv  (E03 – E06 · seed 1–10)
        data/splits/분할명부_5겹_안쪽8.csv   (E02 백본 선별 · seed 1)
        열 : number · Hb정수 · seed · fold · 몫 (학습 · 검증 · 시험) — 🔴 사진 파일 이름 (끝 7자리 = 차트번호) 은 넣지 않는다
        이미 있으면 덮어쓰지 않고 같은지만 확인한다
쓰는 법 PYTHONPATH=code python experiments/00_분할명부_만들기.py 만들기
        PYTHONPATH=code python experiments/00_분할명부_만들기.py 대조 main/results/*/train   # 학습 폴더의 train · val · test.txt 와 명부 대조
"""
import argparse, re, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="The least populated class")   # 정수 Hb 양 끝 층이 1명 — 층화에는 영향 없음 (log 를 덮지 않게)
import numpy as np, pandas as pd
from dataset.ajoumc_anemia import 사람분할, OUTER_SEED

뿌리 = Path(__file__).resolve().parent.parent
ARMS = 뿌리 / "data/arms"; OUT = 뿌리 / "data/splits"
명부 = {"분할명부_10겹_안쪽9.csv": (10, 9, range(1, 11)), "분할명부_5겹_안쪽8.csv": (5, 8, [1])}


def 사람표(f=ARMS / "internal_auto_scca_c224.xlsx"):
    """로더 (eye: exclusive) 와 같은 규칙 — Hb 가 있고 두 눈이 모두 있는 사람 · {number: 정수 Hb}"""
    d = pd.read_excel(f).dropna(subset=["Hb"])
    두눈 = d.groupby("number").separator.agg(lambda s: {1, 2} <= set(s))
    d = d[d.number.isin(두눈[두눈].index)]
    if d.groupby("number").Hb.nunique().max() > 1: sys.exit("🔴 사진마다 Hb 가 다른 사람이 있다")
    return {int(n): int(h) for n, h in d.groupby("number").Hb.first().items()}


def 만들기(_):
    P = 사람표()
    for f in sorted(ARMS.glob("internal_*.xlsx")):
        if 사람표(f) != P: sys.exit(f"🔴 {f.name} 의 사람 · Hb 가 기준 명부와 다르다")
    print(f"  사람 {len(P)} · 입력 명부 {len(list(ARMS.glob('internal_*.xlsx')))}개 모두 같은 사람 · Hb · 바깥 고정값 {OUTER_SEED}")
    OUT.mkdir(parents=True, exist_ok=True)
    for 이름, (cv, K, seeds) in 명부.items():
        rows = []
        for s in seeds:
            for f, (tr, va, te) in enumerate(사람분할(P, cv, s, K)):
                for 몫, S in (("학습", tr), ("검증", va), ("시험", te)):
                    rows += [(n, P[n], s, f, 몫) for n in sorted(S)]
        T = pd.DataFrame(rows, columns=["number", "Hb정수", "seed", "fold", "몫"])
        확인(T, P, cv)
        dst = OUT / 이름
        if dst.exists():
            old = pd.read_csv(dst, encoding="utf-8-sig")
            print(f"  {이름} 이미 있음 → {'같음 ✅' if old.equals(T) else '🔴 다름 — 덮어쓰지 않았다 (새 이름으로 만들 것)'}")
            if not old.equals(T): sys.exit(1)
        else:
            T.to_csv(dst, index=False, encoding="utf-8-sig"); print(f"  저장 {dst.relative_to(뿌리)} ({len(T)}줄)")
    print("[확인] 분할 명부 끝")


def 확인(T, P, cv):
    """① 겹마다 사람 수 ② 사람마다 seed 안에서 한 몫 · 시험은 한 번 ③ 층 최대차이 ④ 바깥 겹이 seed 와 무관 ⑤ 안쪽은 seed 마다 다른가"""
    y = pd.Series(P); 층 = np.unique(y); 전체 = np.array([(y == c).mean() for c in 층])
    시험 = T[T.몫 == "시험"]
    assert (T.groupby(["seed", "fold", "number"]).size() == 1).all(), "한 겹에서 두 몫에 든 사람"
    assert (시험.groupby(["seed", "number"]).size() == 1).all() and 시험.groupby("seed").number.nunique().eq(len(P)).all(), "시험에 한 번씩이 아님"
    차 = [100 * np.abs(np.array([(g.Hb정수 == c).mean() for c in 층]) - 전체).max() for _, g in 시험.groupby(["seed", "fold"])]
    바깥 = 시험.groupby("seed").apply(lambda g: frozenset(map(frozenset, g.groupby("fold").number.apply(set))), include_groups=False)
    검증 = T[T.몫 == "검증"].groupby("seed").number.apply(frozenset)
    n = 시험.groupby(["seed", "fold"]).size()
    print(f"  {cv}겹 · seed {T.seed.nunique()} · 시험 겹 사람 {n.min()}–{n.max()} (사진 ×2) · 층 최대차이 평균 {np.mean(차):.1f} %p (최악 {max(차):.1f})"
          f" · 바깥 겹 seed 사이 같음 {바깥.nunique() == 1} · 안쪽 검증 seed 마다 다름 {검증.nunique() == len(검증)}")
    assert 바깥.nunique() == 1, "바깥 겹이 seed 마다 다르다"


def 대조(a):
    """학습 폴더 (<칸>-fold<f>/{train,val,test}.txt · config.yaml 의 seed) 가 명부와 같은지"""
    d = pd.concat([pd.read_excel(f) for f in ARMS.glob("internal_*.xlsx")])
    stem2n = {Path(str(p)).stem: int(n) for p, n in zip(d.file_path, d.number)}
    M = {k: pd.read_csv(OUT / k, encoding="utf-8-sig") for k in 명부}
    좋음 = 나쁨 = 0
    for top in a.dirs:
        for D in sorted(Path(top).glob("*-fold*")):
            if not (D / "test.txt").exists(): continue
            cfg = (D / "config.yaml").read_text()
            g = lambda k: re.search(rf"^{k}: (\S+)", cfg, re.M).group(1)
            cv, K, s, f = int(g("cv_fold")), int(g("val_inner")), int(g("seed")), int(g("fold"))  # config.yaml 은 seed += fold 전에 저장된다
            T = M[f"분할명부_{cv}겹_안쪽{K}.csv"]; T = T[(T.seed == s) & (T.fold == f)]
            ok = True
            for 몫, fn in (("학습", "train.txt"), ("검증", "val.txt"), ("시험", "test.txt")):
                실제 = {stem2n[Path(l.split("\t")[0]).stem] for l in (D / fn).read_text().splitlines() if l.strip()} if (D / fn).exists() else set()
                ok &= 실제 == set(T[T.몫 == 몫].number)
            좋음 += ok; 나쁨 += not ok
            if not ok: print(f"  🔴 다름 {D}")
    print(f"[확인] 명부와 같음 {좋음} · 다름 {나쁨}"); sys.exit(1 if 나쁨 else 0)


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
sp.add_parser("만들기").set_defaults(func=만들기)
c = sp.add_parser("대조"); c.add_argument("dirs", nargs="+"); c.set_defaults(func=대조)
a = ap.parse_args(); a.func(a)
