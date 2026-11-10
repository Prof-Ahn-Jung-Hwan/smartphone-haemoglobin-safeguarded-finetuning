# -*- coding: utf-8 -*-
"""5단계 외부 — 10겹(8:1:1) 최종 모델을 인도·이탈리아에서 평가한다.  2026-09-22

왜   지금까지의 외부 수치는 3단계(5겹) · 4단계(5겹) 모델의 것이다.
     본문 모델과 본문 외부 수치가 **같은 모델**에서 나와야 읽는 사람이 헷갈리지 않는다.
무엇 09-18 외부 절차 그대로 — 같은 추첨 · 같은 평가 환자(main/results/external/splits 재사용)
     무보정(zero) : 기본 **전부**(머리 4 × 입력 5 × seed) — 싸다
     현지 보정(few) : 기본 **확정 머리 ve × {auto_scca, human_scca}** (n=30 · 추첨 10 · 사이트별) — 비싸다
              보정 1칸(10겹 × 2사이트 × 10추첨) ≈ 20분 → ve 2입력 × seed 10 = 20칸 ≈ 7시간

사용
    python main/analysis/260922_stage5_ext.py zero                          # 끝난 칸 전부
    python main/analysis/260922_stage5_ext.py few                           # ve · auto_scca/human_scca
    python main/analysis/260922_stage5_ext.py few --heads ve veattn --inputs auto_scca human_scca
    python main/analysis/260922_stage5_ext.py score
    (여러 개로 나눠 돌릴 때 : --shard 0 --nshard 3 …)
결과 main/results/stage5/external/{zero,few}/… · main/results/main_results/B4외부_*.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import os
HERE = Path(__file__).resolve().parent; MAIN = HERE.parent; ROOT = MAIN.parent
# ★ 09-24 : 음성 대조(results/negctrl) 등 다른 폴더도 같은 절차로 볼 수 있게 — 기본값은 5단계
S5 = Path(os.environ.get("EXT_TRAIN_ROOT", MAIN / "results/stage5"))
PREFIX = os.environ.get("EXT_CELL_PREFIX", "s5_")
OUT = S5 / "external"; SPLIT = MAIN / "results/external/splits"
FEW_DEFAULT_LR = 2e-5          # 09-17 사전 지정값 (옛 규약 승계)


def few_root(a):
    """★ 09-25 : 보정 lr 이 기본값과 다르면 **다른 폴더**에 쌓는다 (기존 결과 보존·혼동 방지).
    ★ 09-27 : --tag 를 주면 그 이름으로 또 갈라 쌓는다 (보정 규약 대조 : few_enc · few_bn · few_encbn · few_ln)."""
    lr = float(getattr(a, "few_lr", FEW_DEFAULT_LR))
    name = "few" if abs(lr - FEW_DEFAULT_LR) < 1e-12 else f"few_lr{lr:g}"
    tag = str(getattr(a, "tag", "") or "")
    # ★ 10-04 : --eval-all 이면 보정 30명을 뺀 그 사이트 전원을 평가 → 폴더 꼬리 _all (기존 결과와 섞이지 않게)
    if getattr(a, "eval_all", False): tag = f"{tag}_all" if tag else "all"
    return OUT / (name if not tag else f"{name}_{tag}")
MR = MAIN / "results/main_results"; SITES = ["India", "Italy"]; CV = 10
INPUTS = ["auto_scca", "auto_bc", "auto_bcwb", "human_scca", "human_bc"]

import types
src = (MAIN / "external/260918_external_grid.py").read_text()
src = src.split("ap = argparse.ArgumentParser(")[0]          # 맨 아래 CLI 는 뺀다
X = types.ModuleType("extgrid"); X.__dict__["__file__"] = str(MAIN / "external/260918_external_grid.py")
exec(compile(src, str(MAIN / "external/260918_external_grid.py"), "exec"), X.__dict__)
X.TRAIN = S5 / "train"                                       # ★ 5단계 모델을 쓴다


def inp_of(name):                                            # s5_mnl_ve_K40_human_scca_seed2 → human_scca
    for i in INPUTS:
        if f"_{i}_" in name: return i
    sys.exit(f"🔴 입력을 못 읽었다: {name}")


def cells(a=None):
    """10겹이 **모두** 끝난 칸만. (학습이 도는 중에도 안전하게 쓸 수 있다)"""
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
                if getattr(a, "eval_all", False):          # ★ 10-04 : 보정 n명을 뺀 그 사이트 전원
                    ev = df[(df.Country == s) & ~df.index.isin(cal.index)]
                X.finetune(args, model, init, cal, a, seed=args.seed + f + 100 * r + n)
                res = X.predict(args, model, ev)
                dst = FEW / s / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv"
                dst.parent.mkdir(parents=True, exist_ok=True); res.to_csv(dst, index=False)
                done += 1
            del model; torch.cuda.empty_cache()
        el = (time.time() - t0) / 60
        print(f"  [few] {name} 끝 · 누적 {done}회 · {el:.0f}분 · 회당 {60*el/max(done,1):.1f}초", flush=True)


def _읽기(path):
    d = pd.read_csv(path).set_index("key").sort_index()
    if d.empty or d.index.has_duplicates or not np.isfinite(d[["y", "pred"]].to_numpy()).all():
        raise ValueError(f"잘못된 결과 파일: {path}")
    return d


def 평가환자(site, r):
    return pd.read_csv(SPLIT / site / f"rep{r}" / "eval.csv").key.tolist()


def 완료추첨(name, n, reps, FEW=None):
    """양쪽 사이트 × 10겹 파일이 모두 있는 추첨 번호."""
    out = set()
    for r in range(reps):
        FEW_ = FEW or (OUT / "few")
        if all((FEW_ / s_ / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv").is_file()
               and (FEW_ / s_ / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv").stat().st_size > 0
               for s_ in SITES for f in range(CV)):
            out.add(r)
    return out


def 지표(겹들, 이름=""):
    """겹별 데이터프레임 목록(같은 환자 집합)에서 두 단위를 낸다.
       · 단일 : 겹마다 MAE 를 내고 평균 (겹별 변동 = SD) ← 주 지표 · 내부 보고와 같은 단위
       · 앙상블 : 겹 예측을 평균한 뒤 MAE                ← 참고
       합침은 **사이트 환자를 합친 표**를 넘겨받으므로 자연히 환자 수로 가중된다."""
    ms = [float((d.pred - d.y).abs().mean()) for d in 겹들]
    E = pd.DataFrame({"y": 겹들[0].y, "p": pd.concat([d.pred for d in 겹들], axis=1).mean(axis=1)})
    e = (E.p - E.y).abs()
    au = roc_auc_score(E.y < 12, -E.p) if (E.y < 12).nunique() > 1 else np.nan
    return dict(단일MAE=float(np.mean(ms)), 단일SD=float(np.std(ms, ddof=1)) if len(ms) > 1 else np.nan,
                앙상블MAE=float(e.mean()), 편향=float((E.p - E.y).mean()), AUROC=float(au), 환자수=len(E))


def cmd_score(a):
    MR.mkdir(parents=True, exist_ok=True)
    names = cells(a)
    if not names: sys.exit("끝난 칸이 없다")
    n30 = a.ns[0] if len(a.ns) == 1 else 30

    # ★ 지적 4 — 비교하는 칸들이 **같은 추첨**을 쓰게 한다
    FEW = few_root(a)
    if FEW.name != "few": print(f"■ 보정 lr {a.few_lr:g} — 결과 폴더 {FEW.name}\n")
    완료 = {name: 완료추첨(name, n30, a.reps, FEW) for name in names}
    공통 = set.intersection(*완료.values()) if 완료 else set()
    if getattr(a, "require_complete", False) and 공통 != set(range(a.reps)):
        빠짐 = {k: sorted(set(range(a.reps)) - v) for k, v in 완료.items() if v != set(range(a.reps))}
        sys.exit(f"🔴 외부 평가 미완료 — 빠진 추첨: {빠짐}")
    if any(v != 공통 for v in 완료.values()):
        print(f"⚠ 칸마다 끝난 추첨이 다르다 → **공통 추첨 {len(공통)}개 {sorted(공통)}** 만 쓴다")
        for k, v in 완료.items():
            if v != 공통: print(f"   {k} : {sorted(v)}")
    print(f"■ 5단계 외부 — 칸 {len(names)}개 · 공통 추첨 {len(공통)}개 · n={n30}\n")

    rows = []
    for name in names:
        head = name.split("_")[2]; inp = inp_of(name); seed = int(name.rsplit("seed", 1)[1])
        zp = [OUT / "zero" / name / f"fold{f}.csv" for f in range(CV)]
        zero = [_읽기(x) for x in zp] if all(x.exists() for x in zp) else None
        if zero is not None:                                   # 무보정 (외부 202명 전원)
            rows.append(dict(머리=head, 입력=inp, seed=seed, 단계="무보정", 무리="전체202", n=0, 추첨=1,
                             **지표(zero)))
        for r in sorted(공통):
            keys = {s_: 평가환자(s_, r) for s_ in SITES}
            few = {s_: [_읽기(FEW / s_ / f"n{n30}" / f"rep{r}" / name / f"fold{f}.csv") for f in range(CV)]
                   for s_ in SITES}
            for 무리 in SITES + ["합침"]:
                사이트 = SITES if 무리 == "합침" else [무리]
                겹 = [pd.concat([few[s_][f] for s_ in 사이트]) for f in range(CV)]   # ★ 지적 2 — 환자를 합친다
                rows.append(dict(머리=head, 입력=inp, seed=seed, 단계="보정", 무리=무리, n=n30, 추첨=r, **지표(겹)))
                if zero is not None:                            # ★ 지적 3 — **같은 환자**의 무보정 기준선
                    k = [x for s_ in 사이트 for x in keys[s_]]
                    겹0 = [d.loc[[x for x in k if x in d.index]] for d in zero]
                    rows.append(dict(머리=head, 입력=inp, seed=seed, 단계="무보정(같은환자)", 무리=무리,
                                     n=n30, 추첨=r, **지표(겹0)))
    T = pd.DataFrame(rows)
    T.to_csv(MR / "B4외부_원자료.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)

    # 표 — 주 지표는 단일 모델
    G = (T[T.단계.isin(["보정", "무보정(같은환자)"])]
         .groupby(["단계", "머리", "입력", "무리"])
         .agg(seed수=("seed", "nunique"), 추첨수=("추첨", "nunique"), 단일=("단일MAE", "mean"),
              겹변동SD=("단일SD", "mean"), 앙상블=("앙상블MAE", "mean"), 편향=("편향", "mean"),
              AUROC=("AUROC", "mean"), 환자=("환자수", "mean")).round(3))
    print("■ 보정 전후 — **같은 평가 환자**에서 (주 지표 = 단일 모델 · 앙상블은 참고)")
    print(G.to_string()); print()
    Z = T[T.단계 == "무보정"].groupby(["머리", "입력"]).agg(단일=("단일MAE", "mean"), 앙상블=("앙상블MAE", "mean"),
                                                      편향=("편향", "mean"), AUROC=("AUROC", "mean")).round(3)
    print("■ 무보정 (외부 202명 전원 · 참고)"); print(Z.to_string()); print()

    # ★ 지적 5·6 — 자동 vs 사람은 **머리마다 따로**, seed 를 짝으로, 단일 모델 단위로
    P = T[(T.단계 == "보정") & (T.무리 == "합침")].groupby(["머리", "입력", "seed"]).단일MAE.mean().unstack("입력")
    쌍 = [("auto_scca", "human_scca"), ("auto_bc", "human_bc")]
    out = []
    for (자동, 사람) in 쌍:
        if not {자동, 사람} <= set(P.columns): continue
        for head in sorted(P.index.get_level_values("머리").unique()):
            d = (P.loc[head, 자동] - P.loc[head, 사람]).dropna()
            if len(d) < 2: continue
            se = d.std(ddof=1) / np.sqrt(len(d)); tcrit = float(stats.t.ppf(0.975, len(d) - 1))
            out.append(dict(머리=head, 갈래=자동.split("_")[1].upper(), seed수=len(d), 차이=d.mean(),
                            하한=d.mean() - tcrit * se, 상한=d.mean() + tcrit * se,
                            자동이좋은seed=int((d < 0).sum()),
                            p=float(stats.ttest_1samp(d, 0).pvalue)))
    if out:
        A = pd.DataFrame(out)
        A.to_csv(MR / "B4외부_자동vs사람.csv", index=False, encoding="utf-8-sig")
        print("■ 자동 ROI − 사람 ROI (합침 · 보정 · **단일 모델** · 머리마다 따로 · seed 짝)")
        print("   구간은 **seed 변동 구간**이다 (환자 표본의 신뢰구간이 아니다)")
        print(A.round(4).to_string(index=False))
    print(f"\n저장 : {MR/'B4외부_원자료.csv'}" + (f" · {MR/'B4외부_자동vs사람.csv'}" if out else ""))


ap = argparse.ArgumentParser(); sub = ap.add_subparsers(required=True)
DEF = dict(zero=(None, None), few=(["ve"], ["auto_scca", "human_scca"]), score=(None, None))
for n_, f_ in [("zero", cmd_zero), ("few", cmd_few), ("score", cmd_score)]:
    s_ = sub.add_parser(n_); s_.set_defaults(func=f_)
    s_.add_argument("--device", type=int, default=0)
    s_.add_argument("--heads", nargs="*", default=DEF[n_][0])
    s_.add_argument("--inputs", nargs="*", default=DEF[n_][1])
    s_.add_argument("--seeds", type=int, nargs="*", default=None)
    s_.add_argument("--ns", type=int, nargs="+", default=[30])
    s_.add_argument("--reps", type=int, default=10)
    s_.add_argument("--epochs", type=int, default=30)
    s_.add_argument("--few-lr", dest="few_lr", type=float, default=2e-5)
    s_.add_argument("--freeze-extra", dest="freeze_extra", default="",
                    help="보정 때 추가로 얼릴 것 : encoder · norm · bn (쉼표로 여러 개) — 기본값 없음 = 기존 규약")
    s_.add_argument("--tag", default="", help="결과 폴더 꼬리표 (few_<tag>) — 보정 규약 대조용")
    s_.add_argument("--sites", nargs="*", default=None,
                    help="계산할 사이트 (기본 : 둘 다). ★ 09-28 사전 약정 — 새 백본 분석은 India 만 먼저 돌린다")
    s_.add_argument("--require-complete", action="store_true",
                    help="score : 모든 칸이 추첨 0~9 를 마쳤을 때만 표를 낸다")
    s_.add_argument("--eval-all", dest="eval_all", action="store_true",
                    help="★ 10-04 : 보정 n명을 뺀 그 사이트 전원을 평가한다 (결과는 few…_all 폴더)")
    s_.add_argument("--shard", type=int, default=0)
    s_.add_argument("--nshard", type=int, default=1)
a = ap.parse_args(); a.func(a)
