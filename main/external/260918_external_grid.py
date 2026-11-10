# -*- coding: utf-8 -*-
"""★ 확증 실험 5·6단계 — 외부 검증 (인도 95 · 이탈리아 107)   2026-09-18

계획 : main/strategy/260912_end2end_exp_plan.md §4 · §0-3 · D2·D3·D10
옛 run_external.py 에서 검증된 부분(모델 적재 · 예측 · 백본을 얼린 미세조정)을 그대로 옮기고,
확증 계획에 맞게 **사이트별 보정 곡선 · 세 무리(인도/이탈리아/합침) · 외부 자격 판정**을 새로 짰다.

하위 명령
  split     사이트마다 추첨 R 회 : 평가 환자(인도 45 · 이탈리아 57)를 고정하고, 나머지 50명을
            보정 후보로 둔다. 보정 인원 n ∈ {10,20,30,50} 은 **같은 순서의 앞 n 명** (중첩 : C10⊂C20⊂C30⊂C50)
            순서 전략 : random(주) · hbstrat(Hb 5층을 번갈아 — 보조). GPU 불필요
  pick-k    백본마다 공통 K — K 쓰는 머리 전체의 내부 **검증** MAE 평균 최소, ±0.010 안이면 작은 K
  zero      무보정 예측 — 5겹 체크포인트 각각으로 외부 202명 전원 (끝난 3단계 칸 전부 · 싸다)
  few       사이트별 보정 후 예측 — 칸 × 겹 × 사이트 × 추첨 × n
  analyze   표 만들기 (GPU 불필요) — 겹 앙상블 예측으로 세 무리 지표, 선형 재보정 대조, 자격 판정, p 값

미세조정 규약 (옛 규약 그대로 · 모든 칸 동일)
  백본을 얼리고 머리만 · 30 epoch · batch = n//3 · lr --few-lr (기본 2e-5, 모든 칸 통일)
  보정 표본에서 학습 MAE 가 가장 낮은 epoch 의 가중치 · 예측은 5겹 평균(겹 앙상블)
  ⚠️ 09-13 에 감쇠를 epoch 비율(80·90 %)로 바꿨으므로 30 epoch 에서는 25·28 회에 감쇠한다 (옛 탐색 때는 감쇠 없음)
  ⚠️ model.train() 이라 얼린 백본의 BatchNorm 통계는 보정 표본으로 갱신된다 (옛 규약과 동일)

결과  main/results/external/
  splits/{site}/rep{r}/eval.csv · calib_{random,hbstrat}.csv · split_summary.csv · k_선택.csv
  zero/{칸}/fold{f}.csv                                  (202명 전원)
  few/{전략}/{site}/n{n}/rep{r}/{칸}/fold{f}.csv          (그 추첨의 평가 환자)
  analysis/  zero_all.csv(+95% CI) · zero_eval.csv · few.csv · linear.csv · 자격.csv · scca_vs_bc.csv

외부 자격 (09-19) : 사이트별 n=30 에서 인도·이탈리아 모두 ① 보정 후 MAE ≤ 무보정 ② 보정 후 MAE ≤ 1.6 g/dL
  (선행 결막 영상 연구 보고 범위 1.19–1.95 · 기준값 Hu 2023 의 1.6 · |편향| ≤ MAE 이므로 편향도 함께 묶인다. 편향은 보고만,
   옛 기준 |편향| ≤ 1.5 판정은 민감도 분석 열 '옛편향판정' 으로 남긴다)
  + 합침 무보정 AUROC 가 MSE 보다 0.05 이상 낮지 않음. 표준 보정기울기(y~pred)·보정절편은 보고만 한다
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parent; ROOT = MAIN.parent
CODE = ROOT / "code"; GRID = MAIN / "results/grid"; TRAIN = GRID / "train"
DATA = HERE / "data"; OUT = MAIN / "results/external"
SITES = ["India", "Italy"]; NS = [10, 20, 30, 50]; POOL = 50
HEADS = {"mse": (0, 0, 0), "mseattn": (1, 0, 0), "ve": (0, 1, 0), "veattn": (1, 1, 0),
         "or": (0, 0, 1), "orattn": (1, 0, 1), "veor": (0, 1, 1), "full": (1, 1, 1)}
BACKBONES = ["mnl", "mns", "mnv2", "shf", "eb0", "mvxx", "mvx", "enxs", "fvt8", "ef2s0"]
PAT = re.compile(r"g3_(\w+?)_(\w+?)_K(\d+)_(auto_\w+)$")


# ─────────────────────────── 공통 ───────────────────────────
def setup():
    # ★ 09-20 속도 — 일꾼 여러 개가 각자 16 스레드를 쓰면 32 코어에서 서로 빼앗아 회당 88초까지 느려졌다.
    #   환경변수로 준 값이 있으면 그것을, 없으면 4를 쓴다 (계산 결과는 스레드 수와 무관하다).
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    os.chdir(CODE)
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))


def cell(bb, head, K, inp):
    return f"g3_{bb}_{head}_K{K}_{inp}"


def done_cells():
    s = {}
    for d in TRAIN.glob("g3_*-fold*"):
        if (d / "training_time.txt").exists() and (d / "best.ckpt").exists():
            e, f = d.name.rsplit("-fold", 1); s.setdefault(e, set()).add(int(f))
    return sorted(e for e, v in s.items() if v == {0, 1, 2, 3, 4})


def read_ext(inp):
    d = pd.read_csv(DATA / f"arm_ext_{inp}.csv"); d.columns = [c.lstrip("﻿") for c in d.columns]
    miss = [p for p in d.full_path if not Path(p).exists()]
    if miss:
        sys.exit(f"🔴 {inp} 외부 그림 {len(miss)}장 없음: {miss[:3]}")
    return d


def build_args(name, fold, device, epochs=None, lr=None):
    import yaml
    from types import SimpleNamespace
    from ordinal_regression import create_ordinal_lookup
    ns = SimpleNamespace(**yaml.safe_load(open(TRAIN / f"{name}-fold{fold}" / "config.yaml")))
    ns.device = device
    if epochs is not None: ns.epochs = epochs
    if lr is not None: ns.lr = float(lr)
    ns.lookup = create_ordinal_lookup(ns)
    return ns


def load_model(args, ck):
    import torch
    from model import HbNet
    m = HbNet(args).to(args.device)
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval()


def to_samples(df):
    return [(Path(p), {"Hb": float(h), "Hybrid": []}) for p, h in zip(df.full_path, df.Hb)]


_CACHE = {}


def make_dataset(args, samples, mode):
    """★ 09-18 속도 — 원 데이터셋과 **결과가 완전히 같다.**
    변환의 첫 단계는 고정 크기 조정(Resize 224)이고 무작위 변형은 그 뒤에 온다. 2,500×2,000 안팎 원본을
    매 epoch 다시 풀고 줄이는 데 미세조정 시간의 절반 가까이가 들었으므로, 줄인 그림을 한 번만 만들어 두고
    나머지 변형만 적용한다 (Resize 는 난수를 쓰지 않으므로 난수 흐름도 같다)."""
    import torch, torchvision.transforms as T
    from PIL import Image
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset

    class _Cached(AjouMC_AnemiaDataset):
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            ts = s.transform.transforms
            assert isinstance(ts[0], T.Resize), "첫 변환이 Resize 가 아니다 — 캐시를 쓰면 안 된다"
            s._resize, s._rest = ts[0], T.Compose(ts[1:])

        def __getitem__(s, i):
            path, data = s.data[i]
            key = (str(path), tuple(s._resize.size))
            im = _CACHE.get(key)
            if im is None:
                im = s._resize(Image.open(path).convert("RGB")); _CACHE[key] = im
            img = s._rest(im)
            hb = torch.tensor([min(data["Hb"], s.args.Hb_maxima)], dtype=torch.float)
            hyb = torch.FloatTensor(data["Hybrid"]) if s.hybrid_mode else torch.zeros(1)
            return (img, hb, hyb, path.name) if s.w_filename else (img, hb, hyb)

    return _Cached(args, samples, mode=mode, w_filename=True)


def predict(args, model, df, batch=64):
    import torch
    from torch.utils.data import DataLoader
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
    from ordinal_regression import decode_ordinal
    dl = DataLoader(make_dataset(args, to_samples(df), "test"),
                    batch_size=batch, shuffle=False, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for img, _, anno, _ in dl:
            o = model(img.to(args.device), anno.to(args.device))
            p = decode_ordinal(args, o["logits"]) if args.use_ordinal_regression else o["logits"]
            out += p.squeeze(-1).cpu().tolist()
    return pd.DataFrame(dict(key=df.key.to_numpy(), y=df.Hb.to_numpy(), pred=out, Country=df.Country.to_numpy()))


def select(a):
    names = done_cells()
    keep = []
    for n in names:
        m = PAT.match(n); bb, head, K, inp = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        if a.backbones and bb not in a.backbones: continue
        if a.heads and head not in a.heads: continue
        if a.inputs and inp not in a.inputs: continue
        if a.ks and K not in a.ks and K != 0: continue
        keep.append(n)
    if getattr(a, "use_k_rule", False):
        ks = pd.read_csv(OUT / "k_선택.csv").set_index("백본").K.to_dict()
        keep = [n for n in keep if int(PAT.match(n).group(3)) in (0, ks[PAT.match(n).group(1)])]
    if a.nshard > 1:
        keep = [n for i, n in enumerate(keep) if i % a.nshard == a.shard]
    return keep


# ─────────────────────────── split · pick-k ───────────────────────────
def hb_order(pool, rng):
    q = pd.qcut(pool.Hb.rank(method="first"), 5, labels=False)
    bins = [list(rng.permutation(pool.index[q == b].to_numpy())) for b in range(5)]
    order = []
    while any(bins):
        for b in bins:
            if b: order.append(b.pop(0))
    return order


def cmd_split(a):
    base = read_ext("auto_scca")[["key", "Hb", "Country"]]
    rows = []
    for si, site in enumerate(SITES):
        S = base[base.Country == site].reset_index(drop=True)
        for r in range(a.reps):
            rng = np.random.default_rng(20260918 + 1000 * r + si)
            perm = rng.permutation(len(S))
            pool, ev = S.loc[perm[:POOL]], S.loc[perm[POOL:]]
            d = OUT / "splits" / site / f"rep{r}"; d.mkdir(parents=True, exist_ok=True)
            ev.to_csv(d / "eval.csv", index=False)
            orders = {"random": list(pool.index), "hbstrat": hb_order(pool, rng)}
            for st, o in orders.items():
                c = S.loc[o].reset_index(drop=True); c.insert(0, "순서", range(1, len(c) + 1))
                c.to_csv(d / f"calib_{st}.csv", index=False)
                for n in NS:
                    rows.append(dict(site=site, rep=r, 전략=st, n=n, 평가=len(ev), 보정Hb=c.Hb[:n].mean(),
                                     평가Hb=ev.Hb.mean(), 어긋남=c.Hb[:n].mean() - ev.Hb.mean()))
    R = pd.DataFrame(rows); R.to_csv(OUT / "splits/split_summary.csv", index=False, encoding="utf-8-sig")
    print(R.groupby(["site", "전략", "n"]).agg(평가=("평가", "first"), 어긋남=("어긋남", "mean"),
          어긋남_SD=("어긋남", "std")).round(2).to_string())
    print(f"\n저장: {OUT/'splits'}")


def cmd_pick_k(a):
    rows = []
    for n in done_cells():
        m = PAT.match(n); K = int(m.group(3))
        if K == 0 or m.group(4) not in ("auto_scca", "auto_bc"): continue
        v = [float(re.search(r"Best validation MAE:\s*([\d.]+)", (TRAIN / f"{n}-fold{f}" / "training_time.txt").read_text()).group(1)) for f in range(5)]
        rows.append(dict(백본=m.group(1), 머리=m.group(2), K=K, 입력=m.group(4), 검증=np.mean(v)))
    T = pd.DataFrame(rows); out = []
    for bb, g in T.groupby("백본"):
        if g.groupby("입력").size().to_dict() != {"auto_bc": 24, "auto_scca": 24}:
            print(f"  {bb}: SCCA·BC 가 아직 다 끝나지 않았다 — 건너뜀"); continue
        v = g.groupby("K").검증.mean(); best = v.min(); tie = sorted(v[v <= best + 0.010].index)
        out.append(dict(백본=bb, K=tie[0], 최소K=int(v.idxmin()), 동률K=" ".join(map(str, tie)),
                        **{f"검증_K{k}": round(x, 4) for k, x in v.items()}))
    P = pd.DataFrame(out); OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT / "k_선택.csv", index=False, encoding="utf-8-sig"); print(P.to_string(index=False))


# ─────────────────────────── zero ───────────────────────────
def cmd_zero(a):
    setup(); import torch
    cells = select(a); t0 = time.time(); n = 0
    print(f"zero : {len(cells)}칸 × 5겹", flush=True)
    for name in cells:
        inp = PAT.match(name).group(4); df = read_ext(inp)
        for f in range(5):
            dst = OUT / "zero" / name / f"fold{f}.csv"
            if dst.exists(): continue
            args = build_args(name, f, a.device)
            m = load_model(args, TRAIN / f"{name}-fold{f}" / "best.ckpt")
            r = predict(args, m, df); dst.parent.mkdir(parents=True, exist_ok=True); r.to_csv(dst, index=False)
            del m; torch.cuda.empty_cache()
        n += 1
        if n % 20 == 0: print(f"  [zero {n}/{len(cells)}] {(time.time()-t0)/60:.0f}분", flush=True)
    print(f"zero 끝 {(time.time()-t0)/60:.0f}분", flush=True)


# ─────────────────────────── few ───────────────────────────
def finetune(args, model, init_sd, calib, a, seed):
    import torch
    from torch.utils.data import DataLoader
    from dataset.ajoumc_anemia import AjouMC_AnemiaDataset, calc_imbalance
    from misc import MAE_error, adjust_learning_rate, setup_seed
    from model import create_optimizer
    from model.loss import AnemiaLoss
    from ordinal_regression import decode_ordinal, encode_ordinal
    setup_seed(seed)
    model.load_state_dict(init_sd); model.train()
    for p in model.backbone.parameters(): p.requires_grad = False
    # ★ 09-27 보정 규약 대조 — 기본값은 "" 이라 **기존 동작 그대로**다.
    #   freeze_extra 에 들어갈 수 있는 말 (쉼표로 여러 개)
    #     encoder : 인코더(선형 투영)까지 얼린다   → 보정에서 LayerNorm + 출력층만 움직인다
    #     norm    : LayerNorm 의 affine(γ·β)을 얼린다 → 표준화는 그대로, 손잡이만 잠근다
    #     bn      : 백본 BatchNorm 을 **평가 모드로 고정**한다 (현지 사진으로 통계가 갱신되지 않는다)
    #   왜 bn 이 필요한가 : 지금 규약은 백본 가중치만 얼리고 model.train() 을 쓰므로
    #   백본 BatchNorm 의 평균·분산은 보정 사진으로 갱신된다 (09-27 실측 : 한 번 통과에 running_mean 2.7e-2 변화).
    _fe = str(getattr(a, "freeze_extra", "") or getattr(args, "freeze_extra", "") or "")
    if "encoder" in _fe and hasattr(model, "encoder"):
        for p in model.encoder.parameters(): p.requires_grad = False
    if "norm" in _fe and isinstance(getattr(model, "norm", None), torch.nn.LayerNorm):
        for p in model.norm.parameters(): p.requires_grad = False
    _bn_fix = "bn" in _fe
    def _bn_eval():
        if not _bn_fix: return
        for _m in model.backbone.modules():
            if isinstance(_m, torch.nn.modules.batchnorm._BatchNorm): _m.eval()
    _bn_eval()
    data = to_samples(calib); batch = max(2, len(data) // 3)
    # ★ 09-24 음성 대조 B — 보정 단계에서도 **보정용 표본 안에서만** 정렬용 라벨을 섞는다.
    #   이걸 안 하면 보정 때 올바른 정렬이 되살아나 대조가 새어버린다 (외부는 한 사람당 한 장이라 환자 단위 = 표본 단위).
    #   회귀 정답(Hb)은 그대로 두고, 한 번 정한 배정은 이 보정 실행이 끝날 때까지 고정한다.
    align_map = None
    if getattr(args, "shuffle_align", False):
        import numpy as _np
        _rng = _np.random.RandomState(int(getattr(args, "align_seed", 20260924)) + int(seed))
        _names = [pth.name for pth, _ in data]
        _hbs = [float(an["Hb"]) for _, an in data]
        _perm = _rng.permutation(len(_hbs))
        align_map = {n: _hbs[_perm[i]] for i, n in enumerate(_names)}
    dl = DataLoader(make_dataset(args, data, "train"),
                    batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    _, cls_num = calc_imbalance(args, data)
    opt, sch = create_optimizer(args, model, len(dl)); loss_fn = AnemiaLoss(args, cls_num)
    best, step = float("inf"), 0
    유효갱신 = 0                      # ★ 09-29 : 실제로 optimizer 가 움직인 횟수
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(args.epochs):
        adjust_learning_rate(opt, ep, args); model.train(); _bn_eval(); seen, tot = 0, 0.0
        for img, target, anno, fname in dl:
            model.zero_grad()
            img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
            al = None
            if align_map is not None:                                   # ★ 음성 대조 B
                al = torch.tensor([[align_map[f]] for f in fname], dtype=torch.float, device=args.device)
            o = model(img, anno, encode_ordinal(args, target if al is None else al).long())
            loss = loss_fn(o, target, ep >= args.start_ib_epoch, align=al)
            if not torch.isfinite(loss): opt.zero_grad(); continue      # ★ NaN 뿐 아니라 inf 도 거른다
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            유효갱신 += 1
            if sch is not None: sch.step_update(step)
            step += 1
            p = decode_ordinal(args, o["logits"]) if args.use_ordinal_regression else o["logits"]
            tot += MAE_error(p, target).sum().item(); seen += len(target)
        if seen and tot / seen < best:
            best = tot / seen; best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # ★ 09-29 : "보정이 한 번도 유효하게 되지 않았는데 파일만 남는" 경우를 막는다
    if 유효갱신 == 0 or not np.isfinite(best):
        raise RuntimeError(f"보정 실패 — 유효 갱신 {유효갱신}회 · 최적 점수 {best} (seed {seed})")
    model.load_state_dict(best_sd)
    for p in model.parameters(): p.requires_grad = True   # ★ 09-27 : 백본뿐 아니라 **전부** 되돌린다
    #   (freeze_extra 로 얼린 인코더·LayerNorm 이 다음 칸까지 얼어 있지 않도록)
    return model


def cmd_few(a):
    setup(); import torch
    cells = select(a); t0 = time.time(); done = 0
    total = len(cells) * 5 * len(SITES) * a.reps * len(a.ns) * len(a.strategies)
    print(f"few : {len(cells)}칸 · 전략 {a.strategies} · 추첨 {a.reps} · n {a.ns} · 미세조정 lr {a.few_lr} → 최대 {total}회", flush=True)
    for name in cells:
        inp = PAT.match(name).group(4); df = read_ext(inp).set_index("key", drop=False)
        for f in range(5):
            todo = [(st, s, r, n) for st in a.strategies for s in SITES for r in range(a.reps) for n in a.ns
                    if not (OUT / "few" / st / s / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv").exists()]
            if not todo: continue
            args = build_args(name, f, a.device, epochs=a.epochs, lr=a.few_lr)
            model = load_model(args, TRAIN / f"{name}-fold{f}" / "best.ckpt")
            init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for st, s, r, n in todo:
                d = OUT / "splits" / s / f"rep{r}"
                ev = df.loc[pd.read_csv(d / "eval.csv").key]
                cal = df.loc[pd.read_csv(d / f"calib_{st}.csv").key[:n]]
                finetune(args, model, init_sd, cal, a, seed=args.seed + f + 100 * r + n)
                res = predict(args, model, ev)
                dst = OUT / "few" / st / s / f"n{n}" / f"rep{r}" / name / f"fold{f}.csv"
                dst.parent.mkdir(parents=True, exist_ok=True); res.to_csv(dst, index=False)
                done += 1
            del model; torch.cuda.empty_cache()
        el = (time.time() - t0) / 60
        print(f"  [few] {name} 끝 · 누적 {done}회 · {el:.0f}분 · 회당 {60*el/max(done,1):.1f}초", flush=True)
    print(f"few 끝 · {done}회 · {(time.time()-t0)/60:.0f}분", flush=True)


# ─────────────────────────── analyze ───────────────────────────
def auroc(y, p):
    yb = (np.asarray(y) < 12).astype(int); s = -np.asarray(p)
    pos, neg = s[yb == 1], s[yb == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


FEW_MAE_MAX = 1.6   # 09-19 교수님 결정 — 보정 후 MAE 상한 (Hu et al. Front Med 2023). 선행 범위 1.19–1.95

def metrics(y, p):
    """★ 09-18 A안 — 보정 지표를 표준 정의로.
      보정기울기 = 실제를 예측에 회귀(y ~ pred)한 기울기 · 이상값 1 (<1 과적합, >1 과소적합)
      보정절편   = 기울기를 1 로 둔 절편 (calibration-in-the-large) = 평균(실제 − 예측) · 이상값 0
      폭기울기   = 옛 정의(pred ~ y). 눈금이 완벽해도 R² 와 같아져 자격 기준에서 뺐다 — 비교용으로만 남긴다
    """
    y, p = np.asarray(y, float), np.asarray(p, float); e = np.abs(p - y)
    return dict(환자수=len(y), MAE=e.mean(), AUROC=auroc(y, p),
                보정기울기=np.polyfit(p, y, 1)[0], 보정절편=(y - p).mean(), 폭기울기=np.polyfit(y, p, 1)[0],
                편향=(p - y).mean(), 이내1=(e <= 1).mean(), 넘음2=(e > 2).mean())


def boot_ci(y, p, B=1000, seed=0):
    """환자 단위 부트스트랩 95 % CI — MAE · AUROC · 보정기울기 · 보정절편."""
    y, p = np.asarray(y, float), np.asarray(p, float); rng = np.random.default_rng(seed); n = len(y)
    S = {k: [] for k in ["MAE", "AUROC", "보정기울기", "보정절편"]}
    for _ in range(B):
        i = rng.integers(0, n, n); yy, pp = y[i], p[i]
        S["MAE"].append(np.abs(pp - yy).mean()); S["AUROC"].append(auroc(yy, pp))
        S["보정기울기"].append(np.polyfit(pp, yy, 1)[0]); S["보정절편"].append((yy - pp).mean())
    out = {}
    for k, v in S.items():
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        out[f"{k}_하한"], out[f"{k}_상한"] = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return out


def ens(files):
    X = pd.concat([pd.read_csv(f) for f in files])
    return X.groupby("key").agg(y=("y", "first"), pred=("pred", "mean"), Country=("Country", "first"))


def meta(name):
    m = PAT.match(name); return dict(칸=name, 백본=m.group(1), 머리=m.group(2), K=int(m.group(3)), 입력=m.group(4))


def cmd_analyze(a):
    from scipy import stats
    A = OUT / "analysis"; A.mkdir(parents=True, exist_ok=True)
    reps = sorted(int(p.name[3:]) for p in (OUT / "splits/India").glob("rep*"))
    EV = {(s, r): pd.read_csv(OUT / "splits" / s / f"rep{r}" / "eval.csv").key for s in SITES for r in reps}
    CAL = {(st, s, r): pd.read_csv(OUT / "splits" / s / f"rep{r}" / f"calib_{st}.csv").key
           for st in ["random", "hbstrat"] for s in SITES for r in reps}
    Z = {}
    for d in sorted((OUT / "zero").glob("g3_*")):
        fs = sorted(d.glob("fold*.csv"))
        if len(fs) == 5: Z[d.name] = ens(fs)
    # ① 무보정 — 전원 202 (세 무리)
    rows = []
    for name, E in Z.items():
        for g, sub in [("India", E[E.Country == "India"]), ("Italy", E[E.Country == "Italy"]), ("합침", E)]:
            rows.append(dict(**meta(name), 무리=g, **metrics(sub.y, sub.pred), **(boot_ci(sub.y, sub.pred) if a.ci else {})))
    ZA = pd.DataFrame(rows); ZA.to_csv(A / "zero_all.csv", index=False, encoding="utf-8-sig")
    # ② 무보정 · 선형 재보정 · 보정 — 추첨마다 같은 평가 환자
    zrows, lrows, frows = [], [], []
    for name, E in Z.items():
        for r in reps:
            ev = {s: E.loc[EV[(s, r)]] for s in SITES}
            for g in ["India", "Italy", "합침"]:
                sub = pd.concat(ev.values()) if g == "합침" else ev[g]
                zrows.append(dict(**meta(name), rep=r, 무리=g, **metrics(sub.y, sub.pred)))
            for st in ["random", "hbstrat"]:
                for n in NS:
                    lin = {}
                    for s in SITES:
                        c = E.loc[CAL[(st, s, r)][:n]]; b1, b0 = np.polyfit(c.pred, c.y, 1)
                        lin[s] = ev[s].assign(pred=b0 + b1 * ev[s].pred)
                    for g in ["India", "Italy", "합침"]:
                        sub = pd.concat(lin.values()) if g == "합침" else lin[g]
                        lrows.append(dict(**meta(name), 전략=st, n=n, rep=r, 무리=g, **metrics(sub.y, sub.pred)))
                    few = {}
                    for s in SITES:
                        fs = sorted((OUT / "few" / st / s / f"n{n}" / f"rep{r}" / name).glob("fold*.csv"))
                        if len(fs) == 5: few[s] = ens(fs)
                    if len(few) == 2:
                        for g in ["India", "Italy", "합침"]:
                            sub = pd.concat(few.values()) if g == "합침" else few[g]
                            frows.append(dict(**meta(name), 전략=st, n=n, rep=r, 무리=g, **metrics(sub.y, sub.pred)))
    keys = ["칸", "백본", "머리", "K", "입력", "무리"]
    agg = lambda D, k: D.groupby(k).agg(추첨=("rep", "nunique"), MAE=("MAE", "mean"), MAE_SD=("MAE", "std"),
                                        AUROC=("AUROC", "mean"), 보정기울기=("보정기울기", "mean"), 보정기울기_SD=("보정기울기", "std"),
                                        보정절편=("보정절편", "mean"), 폭기울기=("폭기울기", "mean"), 편향=("편향", "mean"),
                                        편향_SD=("편향", "std"), 이내1=("이내1", "mean"), 넘음2=("넘음2", "mean")).reset_index()
    ZE = agg(pd.DataFrame(zrows), keys); ZE.to_csv(A / "zero_eval.csv", index=False, encoding="utf-8-sig")
    LI = agg(pd.DataFrame(lrows), keys + ["전략", "n"]) if lrows else pd.DataFrame()
    LI.to_csv(A / "linear.csv", index=False, encoding="utf-8-sig")
    FE = agg(pd.DataFrame(frows), keys + ["전략", "n"]) if frows else pd.DataFrame()
    FE.to_csv(A / "few.csv", index=False, encoding="utf-8-sig")
    print(f"■ 무보정 {len(Z)}칸 · 보정 결과가 있는 칸 {FE.칸.nunique() if len(FE) else 0}")
    # ③ 자격 판정 (D3 · 09-18 A안 · random · n = a.elig_n · 사이트별로 모두 통과)
    #   ① 보정 후 MAE ≤ 무보정 MAE  ② 보정 후 MAE ≤ FEW_MAE_MAX (09-19, 1.6 = Hu 2023) + 합침 무보정 AUROC 가 MSE 보다 0.05 이상 낮지 않음
    #   09-18 까지의 ② |보정 후 편향| ≤ 1.5 는 탐색 단계 값이라 교체 — '옛편향판정' 열로 민감도 보고
    #   기울기는 자격에서 뺐다 (근거 없는 문턱 · 옛 정의는 눈금이 맞아도 R² 가 된다) — 표준 보정기울기·절편은 보고만 한다
    if len(FE):
        f = FE[(FE.전략 == "random") & (FE.n == a.elig_n)]
        z = ZE.set_index(["칸", "무리"])
        mse_auc = ZA[(ZA.무리 == "합침") & (ZA.머리 == "mse")].set_index(["백본", "입력"]).AUROC
        out = []
        for name, g in f.groupby("칸"):
            row = dict(**meta(name))
            ok = True
            for s in SITES:
                x = g[g.무리 == s].iloc[0]; zm = z.loc[(name, s)].MAE
                row[f"{s}_zero"] = zm; row[f"{s}_few"] = x.MAE; row[f"{s}_편향"] = x.편향
                row[f"{s}_보정기울기"] = x.보정기울기; row[f"{s}_보정절편"] = x.보정절편
                c1, c2 = x.MAE <= zm, x.MAE <= FEW_MAE_MAX
                row[f"{s}_옛편향판정"] = "✓" if abs(x.편향) <= 1.5 else "✗"
                row[f"{s}_판정"] = "".join(["✓" if c else "✗" for c in (c1, c2)]); ok &= c1 and c2
            au = ZA[(ZA.칸 == name) & (ZA.무리 == "합침")].AUROC.iloc[0]
            ref = mse_auc.get((row["백본"], row["입력"]), np.nan)
            row["AUROC_합침"] = au; row["AUROC_MSE"] = ref; c4 = not (au < ref - 0.05); row["AUROC_판정"] = "✓" if c4 else "✗"
            row["자격"] = "🟢" if ok and c4 else "🔴"; out.append(row)
        EL = pd.DataFrame(out); EL.to_csv(A / "자격.csv", index=False, encoding="utf-8-sig")
        print(f"\n■ 외부 자격 (random · 사이트별 n={a.elig_n} · 판정 = 보정악화없음/보정후MAE≤{FEW_MAE_MAX} · AUROC) 통과 {int((EL.자격=='🟢').sum())}/{len(EL)}")
        print(EL[["백본", "머리", "K", "입력", "India_zero", "India_few", "India_판정", "India_보정기울기", "Italy_zero", "Italy_few",
                  "Italy_판정", "Italy_보정기울기", "AUROC_합침", "AUROC_판정", "자격"]].round(3).to_string(index=False))
    # ④ SCCA vs BC · BC_wb — 무보정, 같은 백본·머리·K 짝, 202명 대응 (세 무리)
    rows = []
    for x in ["auto_bc", "auto_bcwb"]:
        pairs = [n for n in Z if PAT.match(n).group(4) == "auto_scca" and n.replace("auto_scca", x) in Z]
        for g in ["India", "Italy", "합침"]:
            es, eb, ds = [], [], []
            for n in pairs:
                A1, B1 = Z[n], Z[n.replace("auto_scca", x)].loc[Z[n].index]
                sel = A1.index if g == "합침" else A1.index[A1.Country == g]
                es.append(np.abs(A1.loc[sel].pred - A1.loc[sel].y).to_numpy())
                eb.append(np.abs(B1.loc[sel].pred - B1.loc[sel].y).to_numpy()); ds.append(es[-1].mean() - eb[-1].mean())
            if not pairs: continue
            es, eb = np.mean(es, axis=0), np.mean(eb, axis=0)
            rows.append(dict(비교=f"SCCA vs {x}", 무리=g, 짝=len(pairs), SCCA우세=int((np.array(ds) < 0).sum()),
                             SCCA=es.mean(), 상대=eb.mean(), 차이=es.mean() - eb.mean(),
                             p_t=stats.ttest_rel(es, eb).pvalue, p_W=stats.wilcoxon(es, eb).pvalue))
    if rows:
        SB = pd.DataFrame(rows); SB.to_csv(A / "scca_vs_bc.csv", index=False, encoding="utf-8-sig")
        print("\n■ 무보정 외부 — SCCA vs BC / BC_wb (짝 칸들의 환자별 오차 평균 · 대응 검정)")
        print(SB.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n저장: {A}")


if __name__ == "__main__":
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--backbones", nargs="*"); c.add_argument("--heads", nargs="*")
    c.add_argument("--inputs", nargs="*"); c.add_argument("--ks", nargs="*", type=int)
    c.add_argument("--use-k-rule", dest="use_k_rule", action="store_true", help="k_선택.csv 의 백본별 K 만 (+K 없는 머리)")
    c.add_argument("--shard", type=int, default=0); c.add_argument("--nshard", type=int, default=1)
    c.add_argument("--reps", type=int, default=10); c.add_argument("--ns", type=int, nargs="+", default=NS)
    c.add_argument("--strategies", nargs="+", default=["random"])
    c.add_argument("--epochs", type=int, default=30); c.add_argument("--few-lr", dest="few_lr", type=float, default=2e-5)
    c.add_argument("--elig-n", dest="elig_n", type=int, default=30)
    c.add_argument("--no-ci", dest="ci", action="store_false", help="무보정 전원 표의 부트스트랩 CI 를 끈다 (빠르게 볼 때)")
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for k, fn in [("split", cmd_split), ("pick-k", cmd_pick_k), ("zero", cmd_zero), ("few", cmd_few), ("analyze", cmd_analyze)]:
        sp.add_parser(k, parents=[c]).set_defaults(func=fn)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); a.func(a)
