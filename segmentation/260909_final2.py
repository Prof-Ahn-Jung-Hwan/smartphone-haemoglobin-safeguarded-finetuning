# -*- coding: utf-8 -*-
"""★ 확정 2판 — 홍채 1단계 1.4배 · 2단계 1.8배 · 새 가르는 자.  2026-09-09

교수님 결정 (09-09)
  · **1단계 홍채 1.4배** — "1단계는 배수에 따라 큰 차이 없지만 1.4배가 오히려 더 깨끗하다.
                            공막은 정말 일부만 잘려 나간다."
  · **2단계 홍채 1.8배** — "2단계는 오히려 더 크게 해야 한다."
  · **새 가르는 자**  공막몫 = |1단계 공막 − BC| / |BC| < 0.30  ->  2단계
  · 판정 화면은 2판(세 줄)으로, 단계를 항상 표기

🔴 09-09 에 잡은 것 — **홍채를 reg 에서만 빼면 소용이 없다.**
   빈 띠를 메우는 닫기(closing)가 홍채 자리를 도로 채운다.
   실측(바깥쪽 `1`) reg1 ∩ 홍채1.4배 = 0 인데 **최종 공막 ∩ 홍채1.4배 = 39,075 화소**.
   그래서 **마지막 SCCA 에서도 홍채를 뺀다.**

빠르게 만든 두 가지 (12 MP 를 그대로 주무르면 한 장에 25초가 걸렸다)
  ① 홍채 키우기를 **팽창 대신 볼록껍질 크기조절**로 (1.7초 -> 0.001초, 결과는 더 정확하다)
  ② 모든 형태연산을 **눈 언저리로 잘라낸 판**에서 한다 (닫기 1.7초 -> 0.1초대)

SAM 은 한 번도 안 돌린다. `reg1`·`reg2`·`bc`·`iris` 는 09-08 전수판에, `pupil` 은 v3 에 있다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OLD  = Path("PATH/TO/PROJECT/2608_eClinicalMedicine_SCCA_imageOnly_filtered_2")
from render_bc_overlay import polygon_mask, read_raw
from metrics_seg import bottom_line, below_frac
from importlib import import_module
PF = import_module("260903_pipeline_final")
R4 = import_module("260904_bc_final_review")
FA = import_module("260908_final_all")
RV = import_module("260909_review2")
tint, outline, _font = R4.tint, R4.outline, R4._font
C_BC, C_SC = FA.C_BC, FA.C_SC

# 🔴 09-09 저녁 되돌림 — 홍채를 키우면 **공막을 그보다 더 먹는다.**
#    안쪽 120장 실측 : 1.1 -> 1.4/1.8 로 공막덮음 0.9491 -> 0.8332 (**−0.116**).
#    2단계에 남는 각막은 1.1 에서 0.065, 1.8 에서 0.036 으로 거의 같은데
#    2단계 공막덮음은 0.941 -> 0.360 으로 무너진다. **키울 값이 없다.**
#    (색칠 문제도 drop_small 문제도 아니었다 — 각각 +0.0001, 200장 중 2장)
GROW1, GROW2 = 1.1, 1.1        # 1단계 홍채 · 2단계 홍채
GATE  = 0.30                   # 새 자 — |1단계 공막 − BC| / |BC|
# ── 속눈썹을 어떻게 걷을까 — 09-09 저녁에 재고 눈으로 고르는 것 셋 ──────────────
#  V1   속눈썹을 **덩어리 무게중심**이 아니라 **화소**로 자른다 (홍채 세로중심 위)
#       🔴 지금 규칙(`upper_only`)은 덩어리 무게중심을 본다. 속눈썹은 길쭉해서
#          위에 걸쳐 있어도 무게중심이 아래로 와, **위 속눈썹의 96.6 %(중앙값)를 놓친다.**
#       안쪽 158장 실측 — 진짜공막 0.0004 · 진짜결막 0.0000 · SCCA 2.07 % 줄임 ·
#       지운 것의 **99.9 % 가 사람 눈 밖**. 공막 5 % 넘게 먹는 장 0/158
#  어둠 SCCA 안에서 밝기 < 동공밝기 + α(공막밝기 − 동공밝기) 인 화소를 버린다
#       안쪽 54장 실측 α=0.35 — 남은 속눈썹의 24 % 만 걷고 공막 0.0026 · 결막 0.0042
#       ※ 교수님 : 남은 각막은 **빛반사(밝은 부분)** 라 어둠으로는 안 걸린다
#  위자름 홍채 윗끝에서 홍채높이 × 0.15 만큼 더 내려 자른다
#       안쪽 158장 실측 — 자른 것 중 **사람 눈은 0.000** (공짜) · 다만 속눈썹 적중은 낮다
ALPHA, TOPCUT = 0.35, 0.15
OPTS = [("② 지금 (덩어리 규칙)", ()),
        ("③ ★V1 속눈썹 화소별", ("v1",)),
        ("④ V1 + 어둠", ("v1", "dark")),
        ("⑤ V1 + 위자름", ("v1", "cut")),
        ("⑥ V1 + 어둠 + 위자름", ("v1", "dark", "cut"))]
# ★ 본판 (교수님 결정 09-09) — 셋 다 쓴다.
#   "v1 위자름 어둠 모두 사용한 것이 더 나은 것 같다. v1 위자름은 깨끗한데 위 속눈썹이 많이 잡힌다."
#   재 보니 셋 다 공짜다 — 바깥쪽 70장에서 BC 를 깎은 몫 평균 0.00000 · 5 % 넘게 깎인 장 0
MAIN = ("v1", "cut", "dark")
GCMP = [1.1, 1.4, 1.8, 2.0]     # 홍채 배수 비교 (따로 그린다)
OUT   = ROOT / "results/260909_final3"
PREV  = ROOT / "results/260908_final_all"
DARK  = 60


def unpack(z, k, H, W):
    return np.unpackbits(z[k])[:H*W].reshape(H, W).astype(bool)


def iris_at(pu, ir0, g):
    """볼록껍질을 무게중심 기준으로 g 배 — 팽창보다 1,000배 빠르고 모양이 정확하다."""
    if pu.sum() < 100 or g <= 1.1:
        return ir0
    ct, _ = cv2.findContours(pu.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ct:
        return ir0
    h = cv2.convexHull(np.vstack(ct)).astype(np.float32); c = h.mean(0, keepdims=True)
    o = np.zeros(ir0.shape, np.uint8)
    cv2.fillConvexPoly(o, ((h - c) * g + c).astype(np.int32), 1)
    return o.astype(bool) | ir0


def cut_iris(m, ir, seed, W):
    """홍채를 빼고 부스러기를 털고 닿은 것만 남긴다. 무너지면 되돌린다.

    🔴 09-09 에 잡은 버그 — `keep_touching` **뒤**의 안전망을 빠뜨렸었다.
       09-08 판에는 `if reg1.sum() < 0.05*w1.sum(): reg1 = drop_small(w1 & fl)` 이 있었고,
       바깥쪽 34장에서 그 안전망이 **실제로 걸려 있었다**(그래서 저장된 reg1 은
       `keep_touching` 을 통과한 적이 없다). 여기서 다시 걸면서 안전망을 안 두니
       **공막이 통째로 0 이 되었다** (`110` : 3,392,967 화소 -> 0, 사람 공막 0.953 을 덮던 것).
       공막이 홍채·BC 와 떨어져 보이는 것은 결막을 절반밖에 못 잡은 장에서 흔하다."""
    if m.sum() < 100:
        return m
    q = m & ~ir
    if q.sum() < 0.05 * m.sum():
        return m
    q = PF.drop_small(q)
    out = FA.keep_touching(q, ir | seed, W)
    return out if out.sum() >= 0.05 * m.sum() else q      # ★ 무너지면 되돌린다


def build(zf, z3, H, W, g1=GROW1, g2=GROW2, gate=GATE, gray=None, opts=MAIN, noiris=False):
    """잘라낸 판에서 다 만들고 제자리에 돌려 놓는다."""
    U0 = unpack(zf, "bc", H, W)
    r1 = unpack(zf, "reg1", H, W); r2 = unpack(zf, "reg2", H, W)
    ir0 = unpack(zf, "iris", H, W); pu = unpack(z3, "pupil", H, W)
    el = unpack(z3, "el_all", H, W)

    # ── 눈 언저리로 자른다 (형태연산이 12 MP 를 다 훑지 않도록)
    any_ = U0 | r1 | r2 | ir0 | pu
    if not any_.any():
        z = np.zeros((H, W), bool)
        return dict(BC=z, SCCA=z, SCLERA=z, 띠=z, IRIS=z, 단계=1, 공막몫=0.0,
                    슬라이스=None, 홍채뺌=False, 띠몫=0.0)
    ys, xs = np.nonzero(any_)
    m_ = int(0.12 * max(ys.max()-ys.min(), xs.max()-xs.min())) + 40
    y0, y1 = max(0, ys.min()-m_), min(H, ys.max()+m_+1)
    x0, x1 = max(0, xs.min()-m_), min(W, xs.max()+m_+1)
    S = (slice(y0, y1), slice(x0, x1))
    h, w = y1-y0, x1-x0
    U0raw = int(U0.sum())
    U0, r1, r2, ir0, pu, el = U0[S], r1[S], r2[S], ir0[S], pu[S], el[S]
    gy = gray[S] if gray is not None else None

    # 🔴 자는 **홍채를 키우기 전**(저장된 1.1배) 값으로 잰다.
    #    키운 뒤에 재면 1단계 공막이 줄어 멀쩡한 장이 자에 걸린다
    #    (바깥쪽 30장 맛보기에서 2단계가 23.3 % -> 43.3 % 로 늘었다).
    #    자가 묻는 것은 "1단계가 공막을 찾았나" 이지 "홍채를 얼마나 깎았나" 가 아니다.
    공막몫 = float((r1 & ~U0).sum() / max(U0.sum(), 1))       # ★ 새 자 — BC 를 먼저 뺀다
    단계 = 2 if 공막몫 < gate else 1
    if noiris:                      # ★ 되돌리기 — 홍채를 아예 안 쓴다 (아래 안전망에서 부른다)
        ir0 = np.zeros_like(ir0); pu = np.zeros_like(pu)
    i1, i2 = iris_at(pu, ir0, g1), iris_at(pu, ir0, g2)
    reg, ir = ((cut_iris(r2, i2, U0, w), i2) if 단계 == 2
               else (cut_iris(r1, i1, U0, w), i1))

    E = FA.fit_ellipse_all(PF.drop_small(U0 | reg), h, w, FA.GROW)
    if (U0 & E).sum() < 0.8 * max(U0.sum(), 1):
        E = np.ones((h, w), bool)
    U, R = U0 & E, reg & E
    s0 = PF.drop_small(U | R)
    ys_, xs_ = np.nonzero(s0) if s0.any() else (np.array([0]), np.array([0]))
    br = FA.bridge_columns(U, PF.drop_small(R & ~U), h, w) & ~ir
    scca, 띠 = FA.fill_gap(s0, pu, max(20, xs_.max()-xs_.min()+1), extra=br)
    scca = scca & ~ir; U = U & ~ir; 띠 = 띠 & ~ir       # 🔴 마지막에도 홍채를 뺀다

    # ── 속눈썹 걷어내기 — 이것도 **마지막에** 해야 한다 (닫기가 도로 채우므로)
    T = np.zeros((h, w), bool)
    yy = np.arange(h)[:, None]
    if ir0.sum() > 20:
        yi = np.nonzero(ir0.any(1))[0]; itop, imid = yi.min(), yi.mean(); ih = max(1, yi.max()-itop)
    else:
        itop, imid, ih = 0, h/2.0, h
    if "v1" in opts and el.sum() > 20:
        T |= el & (yy < imid)                      # ★ 덩어리가 아니라 화소로
    if "cut" in opts:
        T |= (yy < itop + TOPCUT * ih)
    if "dark" in opts and gy is not None and scca.any():
        body = scca & ~ir
        if pu.sum() > 50 and body.sum() > 50:
            b_pu = float(np.median(gy[pu])); b_sc = float(np.median(gy[body]))
            # 🔴 어둠은 **홍채 세로중심 위에만** 건다.
            #    결막의 창백함이 바로 밝기다 — 건강한 사람의 결막은 붉고 어둡다.
            #    밝기로 자르면 건강한 쪽을 더 깎아 **우리가 재려는 차이를 스스로 누른다.**
            #    속눈썹은 위에만 있고 결막은 홍채보다 아래이므로, 위에만 걸면
            #    결막을 한 화소도 안 건드리면서 속눈썹만 걷는다.
            #    (아래까지 걸면 바깥쪽 BC덮음 −1.5 %p · 5 % 넘게 깎인 장 18/202)
            T |= (gy < b_pu + ALPHA * (b_sc - b_pu)) & (yy < imid)
    if T.any():
        scca = scca & ~T; U = U & ~T; 띠 = 띠 & ~T

    # ★ 안전망 C (09-10) — **동공이 폭주하면 홍채가 눈 전체를 덮어 다 지운다.**
    #   `56` : 동공 7,684,218 화소 · 홍채 9,052,377 (사람 결막이 2,500,937) -> SCCA 0 화소
    #   사진 자체는 멀쩡하다. 홍채만 안 쓰면 BC 합만으로 눈을 97~100 % 덮는다
    #   (`56` SCCA덮음 0.973 · `i_89` 0.998). 그래서 **결과가 무너지면 홍채를 빼고 다시 짓는다.**
    #   결과를 보고 되돌리는 것이라 멀쩡한 장은 건드리지 않는다 (안전망 A·B 와 같은 방식).
    if not noiris and scca.sum() < 0.05 * max(U0raw, 1):
        return build(zf, z3, H, W, g1, g2, gate, gray, opts, noiris=True)

    return dict(BC=U, SCCA=scca, SCLERA=PF.drop_small(scca & ~U), 띠=띠, IRIS=ir,
                단계=단계, 공막몫=공막몫, 슬라이스=S, 홍채뺌=bool(noiris),
                띠몫=float(띠.sum()/max(scca.sum(), 1)))


def paste(m, S, H, W):
    o = np.zeros((H, W), bool)
    if S is not None: o[S] = m
    return o


def strip(caps, tiles, ww, hh):
    th = 34
    cv_ = Image.new("RGB", (ww*len(tiles), hh+th), (18, 18, 18)); dr = ImageDraw.Draw(cv_)
    f2 = _font(max(14, ww//26))
    for k, (cap, im) in enumerate(zip(caps, tiles)):
        cv_.paste(Image.fromarray(im), (k*ww, th))
        dr.text((k*ww+8, 6), cap, font=f2, fill=(255, 230, 140))
    return cv_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["internal", "external"], default="internal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="이 stem 들만 다시 만들고 장별 CSV 에 덮어쓴다 (쉼표로 구분)")
    ap.add_argument("--nocmp", action="store_true", help="배수 비교 그림을 안 그린다")
    ap.add_argument("--pageonly", action="store_true")
    ap.add_argument("--outdir", default="", help="결과 폴더를 바꿔 쓴다 (예전 판의 화면만 다시 만들 때)")
    a = ap.parse_args()
    EXT = a.set == "external"; NM = "바깥쪽" if EXT else "안쪽"
    global OUT
    if a.outdir: OUT = ROOT / a.outdir
    v3 = ROOT / ("results/260903_final_v3_ext" if EXT else "results/260903_final_v3")
    pj = json.load(open(OLD / "autosegmentation/external/results/external_index.json" if EXT
                        else "PATH/TO/PROJECT/260910_eClinicalMedicine_filtered_2_autosegmentation/results/extra42_index.json"))
    exif = pj["apply_exif"]
    if EXT:
        poly = {Path(k).stem: v["polygons"] for k, v in pj["items"].items()}
        raws = {Path(k).stem: v["raw"] for k, v in pj["items"].items()}
    else:
        full = json.load(open(ROOT / "results/polygon_index_full.json"))
        poly = {s: v["polygons"] for s, v in full.items()}
        raws = {s: pj["items"][s+".jpg"]["raw"] for s in full if (s+".jpg") in pj["items"]}
    D0 = pd.read_csv(PREV / f"장별_{NM}.csv"); D0.stem = D0.stem.astype(str)
    옛단계 = dict(zip(D0.stem, D0.쓴단계))
    keys = [s for s in D0.stem if (PREV/"mask"/f"{s}.npz").exists() and s in raws]
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        keys = [s for s in keys if s in want]
    if a.limit: keys = keys[:a.limit]
    for d in ("panel", "cmp", "grow", "mask", "review2"):
        (OUT/d).mkdir(parents=True, exist_ok=True)

    if not a.pageonly:
        print(f"[{NM}] {len(keys)}장 · 1단계 홍채 {GROW1}배 · 2단계 {GROW2}배 · 문턱 {GATE} · SAM 안 씀",
              flush=True)
        rows = []
        for i, s in enumerate(keys):
            zf = np.load(PREV/"mask"/f"{s}.npz"); z3 = np.load(v3/"mask"/f"{s}.npz")
            H, W = zf["shape"]
            img = read_raw(raws[s], exif)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            M = build(zf, z3, H, W, gray=gray, opts=MAIN)
            S = M["슬라이스"]
            BC = paste(M["BC"], S, H, W); SC = paste(M["SCLERA"], S, H, W)
            SA = paste(M["SCCA"], S, H, W); TI = paste(M["띠"], S, H, W)
            IR = paste(M["IRIS"], S, H, W)
            P = poly.get(s) or {}
            r = dict(stem=s, 무리=NM, 단계=M["단계"], 공막몫=M["공막몫"],
                     옛단계=int(옛단계.get(s, 0)), 띠몫=M["띠몫"], 홍채뺌=M["홍채뺌"],
                     SCCA화소=int(SA.sum()), BC화소=int(BC.sum()), 공막화소=int(SC.sum()),
                     어두운몫=float((gray[SA] < DARK).mean()) if SA.any() else np.nan,
                     밝기=float(gray[SA].mean()) if SA.any() else np.nan)
            if P.get("WCBC") and P.get("BC"):
                gw = polygon_mask(P["WCBC"], (W, H)); gb = polygon_mask(P["BC"], (W, H))
                gs = gw & ~gb
                if gb.sum() > 100 and gs.sum() > 100:
                    lw = bottom_line(gw, W)
                    r.update(SCCA덮음=float((SA & gw).sum()/gw.sum()),
                             SCCA크기비=float(SA.sum()/gw.sum()),
                             SCCA아래삐짐=below_frac(SA, lw, H),
                             BC덮음=float((BC & gb).sum()/gb.sum()),
                             공막덮음=float((SC & gs).sum()/gs.sum()))
            if P.get("CO"):
                co = polygon_mask(P["CO"], (W, H))
                if co.sum() > 100: r["각막침범"] = float((SA & co).sum()/co.sum())
            rows.append(r)
            np.savez_compressed(OUT/"mask"/f"{s}.npz", shape=np.array([H, W]),
                                bc=np.packbits(BC), sclera=np.packbits(SC),
                                scca=np.packbits(SA), iris=np.packbits(IR))

            base = SA | IR
            if base.sum() < 100: base = np.ones((H, W), bool)
            ys, xs = np.nonzero(base)
            ph, pw = int((ys.max()-ys.min())*0.9)+20, int((xs.max()-xs.min())*0.35)+20
            cy0, cy1 = max(0, ys.min()-ph), min(H, ys.max()+ph)
            cx0, cx1 = max(0, xs.min()-pw), min(W, xs.max()+pw)
            bs = img[cy0:cy1, cx0:cx1]; cm = lambda m: m[cy0:cy1, cx0:cx1]
            caps = ["① 원본", "② 결막 BC", "③ SCCA — 결막 민트 · 공막 하늘", "④ 테두리만",
                    f"⑤ 이어 메운 띠(노랑) · SCCA 의 {M['띠몫']:.1%}"]
            tiles = [bs.copy(),
                     np.ascontiguousarray(tint(bs, cm(BC), C_BC, 0.45)),
                     np.ascontiguousarray(tint(tint(bs, cm(BC), C_BC, 0.45), cm(SC), C_SC, 0.45)),
                     outline(outline(bs.copy(), cm(BC), C_BC), cm(SC), C_SC),
                     np.ascontiguousarray(tint(tint(tint(bs, cm(BC), C_BC, 0.35), cm(SC), C_SC, 0.35),
                                               cm(TI), (255, 210, 0), 0.75))]
            hh, ww = tiles[0].shape[:2]
            strip(caps, tiles, ww, hh).save(OUT/"panel"/f"{s}.jpg", quality=84)

            # ── 배수 비교 그림 — **모든 사진**에. 그 장이 쓰는 단계에 배수를 건다.
            #    교수님: "수치로 보는 것과 내 눈으로 보는 것이 많이 다르다. 다 보여 달라."
            if not a.nocmp:
                cc, tt = ["① 원본"], [bs.copy()]
                for cap, op in OPTS:
                    Mg = build(zf, z3, H, W, gray=gray, opts=op)
                    Sg = Mg["슬라이스"]
                    bcg = paste(Mg["BC"], Sg, H, W); scg = paste(Mg["SCLERA"], Sg, H, W)
                    cc.append(cap)
                    tt.append(np.ascontiguousarray(
                        tint(tint(bs, cm(bcg), C_BC, 0.42), cm(scg), C_SC, 0.42)))
                strip(cc, tt, ww, hh).save(OUT/"cmp"/f"{s}.jpg", quality=84)
                # ── 홍채 배수 비교 (본판 속눈썹 규칙 V1 을 건 채로 배수만 바꾼다)
                cc2, tt2 = ["① 원본"], [bs.copy()]
                for g in GCMP:
                    Mg = build(zf, z3, H, W, g1=g, g2=g, gray=gray, opts=MAIN)
                    Sg = Mg["슬라이스"]
                    bcg = paste(Mg["BC"], Sg, H, W); scg = paste(Mg["SCLERA"], Sg, H, W)
                    cc2.append(f"홍채 {g:.1f}배 · {Mg['단계']}단계")
                    tt2.append(np.ascontiguousarray(
                        tint(tint(bs, cm(bcg), C_BC, 0.42), cm(scg), C_SC, 0.42)))
                strip(cc2, tt2, ww, hh).save(OUT/"grow"/f"{s}.jpg", quality=84)
            if (i+1) % 50 == 0: print(f"  [{i+1}/{len(keys)}]", flush=True)
        D = pd.DataFrame(rows)
        old_csv = OUT/f"장별_{NM}.csv"
        if a.only and old_csv.exists():          # ★ 고친 장만 갈아 끼운다
            O = pd.read_csv(old_csv); O.stem = O.stem.astype(str)
            for c in D.columns:
                if c not in O.columns: O[c] = np.nan
            O = O[~O.stem.isin(D.stem.astype(str))]
            D = pd.concat([O, D], ignore_index=True)
        D.to_csv(old_csv, index=False, encoding="utf-8-sig")
    else:
        D = pd.read_csv(OUT/f"장별_{NM}.csv"); D.stem = D.stem.astype(str)

    print(f"\n===== [{NM}] {len(D)}장 · 1단계 {GROW1}배 · 2단계 {GROW2}배 · 문턱 {GATE} =====")
    print(f"  2단계를 쓴 장 {int((D.단계==2).sum())} ({100*(D.단계==2).mean():.1f} %) · "
          f"09-08 판 {int((D.옛단계==2).sum())} ({100*(D.옛단계==2).mean():.1f} %)")
    print(f"   2 -> 1 로 돌아온 장 {int(((D.옛단계==2)&(D.단계==1)).sum())} · "
          f"1 -> 2 로 넘어간 장 {int(((D.옛단계==1)&(D.단계==2)).sum())}")
    for c in ("어두운몫", "밝기", "각막침범", "SCCA덮음", "SCCA크기비", "SCCA아래삐짐", "BC덮음", "공막덮음"):
        if c in D: print(f"  {c:<10}{D[c].mean():.4f}")

    recs = []
    for r in D.itertuples():
        s = str(r.stem)
        if not (OUT/"panel"/f"{s}.jpg").exists(): continue
        chg = "그대로" if (not r.옛단계 or r.옛단계 == r.단계) else f"{int(r.옛단계)}→{int(r.단계)}"
        recs.append(dict(f=s, img=f"../panel/{s}.jpg", co=NM, stg2=int(r.단계), chg=chg,
                         stg=f"{int(r.단계)}단계" + ("" if chg == "그대로" else f" ({chg})")))
    그룹 = ([("인도", [x for x in recs if x["f"].startswith("i_")]),
           ("이탈리아", [x for x in recs if not x["f"].startswith("i_")])] if EXT else [(NM, recs)])
    for 이름, rs in 그룹:
        if not rs: continue
        for x in rs: x["co"] = 이름
        dst = OUT/"review2"/f"판정2_{이름}.html"
        RV.page(rs, f"확정 2판 · 홍채 {GROW1}/{GROW2} · 속눈썹 화소별(V1) · {이름}",
                f"final2_260909_{이름}", dst, note="정렬 그대로")
        print(f"  {dst}  ({len(rs)}장 · 저장키 final2_260909_{이름})")
    for 폴더, 제목, 열쇠, 파일 in [("cmp", "속눈썹 걷는 법 고르기", "lash2_260909", "속눈썹확인"),
                                ("grow", "홍채 배수 고르기", "grow2_260909", "배수확인")]:
        got = [x for x in recs if (OUT/폴더/f"{x['f']}.jpg").exists()]
        for 이름, rs in ([("인도", [x for x in got if x["f"].startswith("i_")]),
                         ("이탈리아", [x for x in got if not x["f"].startswith("i_")])] if EXT
                        else [(NM, got)]):
            if not rs: continue
            rs = [dict(x, img=f"../{폴더}/{x['f']}.jpg", co=이름) for x in rs]
            dst = OUT/"review2"/f"{파일}_{이름}.html"
            RV.page(rs, f"{제목} · {이름}", f"{열쇠}_{이름}", dst, note="눈으로 고르십시오")
            print(f"  {dst}  ({len(rs)}장 · 저장키 {열쇠}_{이름})")


if __name__ == "__main__":
    main()
