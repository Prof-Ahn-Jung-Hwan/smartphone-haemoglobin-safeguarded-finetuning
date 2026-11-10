# -*- coding: utf-8 -*-
"""★ 최종 자동분할 파이프라인 — 낱말 다섯 · 손질 넷.  2026-09-03

  한 파일로 **내부 1,479장**과 **외부(India·Italy) 202장**을 모두 돌린다.
  사람 손은 0 이다. 사람 주석은 **채점에만** 쓴다.

낱말 다섯 (전부 일상어. 의학용어는 하나도 쓰지 않는다)
  BC   "lower eyelid inner surface"   점수 1등만        -> 결막.  ★ 손대지 않는다
  WC   "white part of the eye"        점수 2등까지      -> 공막 기준선
  ALT  "the bright area in the eye"   점수 2등까지      -> 공막 대체 (방아쇠가 당길 때만)
  PUP  "pupil"                        점수 1등만        -> 홍채의 씨앗
  EL   "eyelashes"                    전부              -> 속눈썹 후보

방아쇠
  WC 1등 점수 < 0.75 이면 ALT 로 갈아탄다        (08-31 실측 AUC 0.894 · 정답을 쓰지 않는다)

손질 넷 — ①②③ 은 공막에만. ④ 부스러기는 **공막과 BC 둘 다**에 건다
  ① 바닥     BC 의 열별 아래끝보다 아래를 자른다       공막은 결막 아래에 있을 수 없다
             ★ 바닥은 **부스러기를 거른 BC** 로 잡는다 (점점이 튄 조각이 바닥을 끌어내리지 않도록)
  ② 홍채     PUP 을 볼록하게 채우고 반지름 10 % 키워 뺀다  PUP 은 홍채의 80 % 만 덮는다
  ③ 윗속눈썹 EL 중 무게중심이 홍채 세로중심보다 위인 덩어리만 뺀다
  ④ 부스러기 제일 큰 덩어리의 5 % 가 안 되는 조각을 버린다   ★ 09-03 추가

  SCCA = BC ∪ 손질한 공막

왜 이 순서인가 (전부 실측)
  1·2등만 쓰기   모든 덩어리를 합치면 아래 피부를 문다 — 335장 중 167장 회복 (p<1e-5)
  ALT 로 갈아타기 `white` 는 색을 부르는 말이라 붉거나 누런 공막에서 진다 — 0.021 -> 0.836
  ② 홍채 +10 %  홍채몫 0.135 -> 0.034 · 공막 -0.001 (사람 홍채 주석을 상한으로 확인)
  ① 바닥        문제의 11장에서 아래삐짐 0.250 -> 0.146 · 공막 손실 0
  ③ 윗속눈썹    공막 손실 0 · 결막 깎임 0   🔴 통째로 빼면 43장에서 공막 20 % 손실
  ④ 부스러기    전수 공막포함 0.890 그대로 · 공막 0.02 넘게 잃는 장 1

사용법
    conda activate torch271
    python analysis/260903_pipeline_final.py --set internal --limit 6    # 맛보기
    python analysis/260903_pipeline_final.py --set internal             # 1,479장 약 45분
    python analysis/260903_pipeline_final.py --set external             # 202장 약 8분
    python analysis/260903_build_review_v3.py                           # 판정표
결과
    results/260903_final_v3/          내부 : ALL_보통 · ALL_카드 · UW107
    results/260903_final_v3_ext/      외부 : India · Italy
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from render_bc_overlay import polygon_mask, read_raw
from metrics_seg import bottom_line, below_frac

ROOT = HERE.parent
OLD = Path("PATH/TO/PROJECT/2608_eClinicalMedicine_SCCA_imageOnly_filtered_2")
PAIRS_IN = Path("PATH/TO/PROJECT/260910_eClinicalMedicine_filtered_2_autosegmentation/results/extra42_index.json")   # ★ 09-10 : 되살린 42장
PAIRS_EX = OLD / "autosegmentation/external/results/external_index.json"
MODEL_ID = "facebook/sam3"

P_BC  = "lower eyelid inner surface"
P_WC  = "white part of the eye"
P_ALT = "the bright area in the eye"
P_PUP = "pupil"
P_EL  = "eyelashes"

TRIG, GROW, FRAG = 0.75, 0.10, 0.05
CARD_FROM = 884
GT = (60, 230, 90); BCC = (255, 70, 220); SCC = (60, 200, 255)
IRC = (255, 120, 40); ELC = (80, 140, 255); WCC = (170, 170, 170)


def _font(sz):
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/nanum/NanumGothic.otf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def tint(img, m, c, al=0.42):
    o = img.copy()
    if m is None or m.sum() == 0:
        return o
    o[m] = (o[m] * (1 - al) + np.array(c) * al).astype(np.uint8)
    ct, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, ct, -1, c, max(2, img.shape[1] // 400))
    return o


# ── 손질 넷 ───────────────────────────────────────────────────────────────
def iris_of(pu):
    """② PUP -> 홍채. 볼록하게 채우고 반지름 10 % 키운다.  ★ 사람 주석을 쓰지 않는다"""
    if pu.sum() < 20:
        return pu
    u = pu.astype(np.uint8)
    ct, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ct:
        return pu
    o = np.zeros_like(u); cv2.drawContours(o, [cv2.convexHull(np.vstack(ct))], -1, 1, -1)
    r = np.sqrt(pu.sum() / np.pi); k = max(3, int(round(r * GROW)) * 2 + 1)
    return cv2.dilate(o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def upper_only(el, ir, H):
    """③ 무게중심이 홍채 세로중심보다 위인 덩어리만.
    🔴 EL 이 실패하면 '눈 전체 덩어리' 를 내는데, 그 무게중심은 아래라 저절로 빠진다."""
    out = np.zeros_like(el)
    if el.sum() == 0:
        return out
    ymid = float(np.nonzero(ir.any(1))[0].mean()) if ir.sum() > 20 else H / 2.0
    n, lab, st, cen = cv2.connectedComponentsWithStats(el.astype(np.uint8))
    for v in range(1, n):
        if cen[v][1] < ymid:
            out[lab == v] = True
    return out


def drop_small(m, frac=FRAG):
    """④ 제일 큰 덩어리의 frac 배가 안 되는 조각을 버린다.
    공막은 홍채 좌우 두 덩어리다. 속눈썹에 얹힌 점들은 여기서 떨어져 나간다."""
    if m.sum() == 0:
        return m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 2:
        return m
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    for v in range(1, n):
        if st[v, cv2.CC_STAT_AREA] >= frac * big:
            keep |= (lab == v)
    return keep


@torch.no_grad()
def main(a):
    from transformers import Sam3Model, Sam3Processor
    EXT = (a.set == "external")
    pj = json.load(open(PAIRS_EX if EXT else PAIRS_IN))
    items, exif = pj["items"], pj["apply_exif"]
    OUT = ROOT / ("results/260903_final_v3_ext" if EXT else "results/260903_final_v3")

    if EXT:
        keys = sorted(items)
        meta = {k: dict(anon=Path(k).stem, batch=items[k].get("batch", ""),
                        group=items[k]["site"], poly=items[k]["polygons"],
                        raw=items[k]["raw"], uw=False) for k in keys}
        GROUPS = ["India", "Italy"]
    else:
        full = json.load(open(ROOT / "results/polygon_index_full.json"))
        G = pd.read_csv(ROOT / "results/260901_final/장별.csv", encoding="utf-8-sig").set_index("stem")
        keys = [s for s in G.index if (s + ".jpg") in items and s in full]
        meta = {}
        for s in keys:
            case = int(s.split("_")[1])
            meta[s] = dict(anon=full[s].get("anon", ""), batch=full[s].get("batch", ""),
                           group=("ALL_카드" if case >= CARD_FROM else "ALL_보통"),
                           poly=full[s]["polygons"], raw=items[s + ".jpg"]["raw"],
                           uw=bool(G.at[s, "공막실패였음"]), case=case)
        GROUPS = ["ALL_보통", "ALL_카드"]
    if a.limit:
        keys = keys[: a.limit]

    DIRS = {g: OUT / g for g in GROUPS}
    D_UW, MD = OUT / "UW107", OUT / "mask"
    for d in list(DIRS.values()) + [MD] + ([] if EXT else [D_UW]):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[{a.set}] 대상 {len(keys)}장 · SAM {len(keys)*5}회", flush=True)

    dev = f"cuda:{a.device}"
    model = Sam3Model.from_pretrained(MODEL_ID).to(dev).eval()
    proc = Sam3Processor.from_pretrained(MODEL_ID)
    fnt, fnt2 = _font(25), _font(24)

    def seg(pil, text, shape):
        """(1등, 2등까지, 전부, 조각수, 1등점수)"""
        z = np.zeros(shape, bool)
        inp = proc(images=pil, text=text, return_tensors="pt").to(dev)
        R = proc.post_process_instance_segmentation(
            model(**inp), threshold=a.threshold, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())[0]
        ms = R["masks"].cpu().numpy() > 0
        if not len(ms):
            return z, z, z, 0, 0.0
        o = np.argsort(-R["scores"].cpu().numpy())
        sc = R["scores"].cpu().numpy()[o]; ms = ms[o]
        return ms[0], ms[:2].any(axis=0), ms.any(axis=0), len(ms), float(sc[0])

    rows = []
    for i, key in enumerate(keys):
        M = meta[key]; P = M["poly"]
        if not all(k in P and len(P[k]) >= 3 for k in ("BC", "WCBC")):
            continue
        img = read_raw(M["raw"], exif); H, W = img.shape[:2]
        pil = Image.fromarray(img)
        gw = polygon_mask(P["WCBC"], (W, H)); gb = polygon_mask(P["BC"], (W, H))
        gs = gw & ~gb
        gwS, gbS = max(int(gw.sum()), 1), max(int(gb.sum()), 1)

        b1_raw, _, _, nb, sb = seg(pil, P_BC, (H, W))        # 결막 1등
        b1 = drop_small(b1_raw)                              # ④ BC 에도 부스러기 거르기
        _, wb, _, nw, sw = seg(pil, P_WC, (H, W))            # 공막 기준선 2등까지
        _, wa, _, na, sa = seg(pil, P_ALT, (H, W))           # 공막 대체 2등까지
        pu, _, _, npu, sp = seg(pil, P_PUP, (H, W))          # 동공 1등
        _, _, el_all, nel, sel = seg(pil, P_EL, (H, W))      # 속눈썹 전부

        갈아탐 = bool(sw < TRIG)
        w_raw = wa if 갈아탐 else wb
        ir = iris_of(pu)                                     # ②
        el = upper_only(el_all, ir, H)                       # ③
        Lb = bottom_line(b1, W); yy = np.arange(H)[:, None]
        floor = (yy <= Lb[None, :]) if Lb is not None else np.ones((H, W), bool)   # ①
        w_cut = ((w_raw & ~ir) & ~el) & floor
        w_fin = drop_small(w_cut)                            # ④
        scca = b1 | w_fin

        lw = bottom_line(gw, W)
        r = dict(key=key, anon=M["anon"], batch=M["batch"], group=M["group"],
                 갈아탐=갈아탐, 점수_BC=sb, 점수_WC=sw, 점수_대체=sa, 점수_동공=sp, 점수_속눈썹=sel,
                 조각_BC=nb, 조각_WC=nw, 조각_대체=na, 조각_속눈썹=nel,
                 SCCA_덮음=float((scca & gw).sum() / gwS),
                 SCCA_크기비=float(scca.sum() / gwS),
                 SCCA_아래삐짐=below_frac(scca, lw, H),
                 BC_덮음=float((b1 & gb).sum() / gbS),
                 BC_아래삐짐=below_frac(b1, lw, H),
                 공막포함=float((w_fin & gs).sum() / gs.sum()) if gs.sum() >= 100 else np.nan,
                 공막_아래삐짐=below_frac(w_fin, lw, H),
                 뺀_홍채=float((ir & w_raw).sum() / max(w_raw.sum(), 1)),
                 뺀_속눈썹=float((el & w_raw).sum() / max(w_raw.sum(), 1)),
                 뺀_바닥=float(((~floor) & w_raw).sum() / max(w_raw.sum(), 1)),
                 뺀_부스러기=float((w_cut & ~w_fin).sum() / max(w_cut.sum(), 1)),
                 뺀_부스러기_BC=float((b1_raw & ~b1).sum() / max(b1_raw.sum(), 1)),
                 조각_BC원본=int(cv2.connectedComponents(b1_raw.astype(np.uint8))[0]) - 1)
        if not EXT:
            r["사례"] = M["case"]; r["카드"] = (M["group"] == "ALL_카드"); r["공막실패였음"] = M["uw"]
        if len(P.get("CO", [])) >= 3:
            co = polygon_mask(P["CO"], (W, H))
            if co.sum():
                r["홍채오염"] = float((w_fin & co).sum() / max(w_fin.sum(), 1))
        rows.append(r)
        np.savez_compressed(MD / f"{Path(key).stem}.npz", shape=np.array([H, W]),
                            b1=np.packbits(b1), b1_raw=np.packbits(b1_raw),
                            w_base=np.packbits(wb), w_alt=np.packbits(wa),
                            pupil=np.packbits(pu), iris=np.packbits(ir),
                            el_all=np.packbits(el_all), el_up=np.packbits(el),
                            w_fin=np.packbits(w_fin), scca=np.packbits(scca))

        # ── 그림 여섯 칸 ──
        ys, xs = np.nonzero(gw)
        ph, pw = int((ys.max()-ys.min())*1.1), int((xs.max()-xs.min())*.7)
        cy0, cy1 = max(0, ys.min()-ph), min(H, ys.max()+ph)
        cx0, cx1 = max(0, xs.min()-pw), min(W, xs.max()+pw)
        base = img[cy0:cy1, cx0:cx1]
        cm = lambda m: m[cy0:cy1, cx0:cx1]
        말 = "밝은영역" if 갈아탐 else "흰눈동자"
        tiles = [("① 원본", None, None), ("② 최종 BC", b1, BCC), ("③ 최종 SCCA", scca, SCC),
                 (f"④ 쓴 공막 [{말}] · 뺀 홍채 주황 · 뺀 속눈썹 파랑", w_fin, WCC),
                 ("⑤ 사람 SCCA", gw, GT), ("⑥ 사람 BC", gb, GT)]
        imgs = []
        for name, m, col in tiles:
            o = np.ascontiguousarray(base if m is None else tint(base, cm(m), col))
            if name.startswith("④"):
                if ir.sum():
                    o = tint(o, cm(ir), IRC, 0.30)
                if el.sum():
                    o = tint(o, cm(el), ELC, 0.35)
            p = Image.fromarray(o)
            p = p.resize((a.width, int(p.height*a.width/p.width)), Image.LANCZOS)
            d = ImageDraw.Draw(p); d.rectangle([0, 0, a.width, 42], fill=(0, 0, 0))
            d.text((8, 5), name, font=fnt2, fill=(255, 255, 255))
            imgs.append(p)
        tw_, th_ = imgs[0].width, imgs[0].height
        cv = Image.new("RGB", (tw_*3, th_*2+46), (18, 18, 18))
        for k2, p in enumerate(imgs):
            cv.paste(p, ((k2 % 3)*tw_, 46+(k2//3)*th_))
        ImageDraw.Draw(cv).text(
            (8, 6), f"{r['anon']} {M['batch']} [{말}] 점수{sw:.2f}"
                    f" | 덮음 {r['SCCA_덮음']:.2f} 크기비 {r['SCCA_크기비']:.1f}"
                    f" 아래삐짐 {r['SCCA_아래삐짐']:.2f}"
                    f" | 뺀몫 홍채{r['뺀_홍채']*100:.0f} 속눈썹{r['뺀_속눈썹']*100:.0f}"
                    f" 바닥{r['뺀_바닥']*100:.0f} 부스러기{r['뺀_부스러기']*100:.0f}"
                    f" | BC부스러기{r['뺀_부스러기_BC']*100:.0f}"
                    + (f"  [{M['group']}]" if EXT else "")
                    + ("  [08-31공막실패]" if M["uw"] else ""),
            font=fnt, fill=(255, 220, 90))
        nn = int(round((1-r["SCCA_덮음"])*100))
        fn = f"{nn:03d}_{r['anon']}_못덮음{nn:03d}.jpg"
        cv.save(DIRS[M["group"]] / fn, quality=84)
        if M["uw"]:
            cv.save(D_UW / fn, quality=84)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(keys)}] 갈아탐 {sum(x['갈아탐'] for x in rows)}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "장별.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(D)} 장 · 갈아탐 {int(D.갈아탐.sum())} ({D.갈아탐.mean()*100:.1f} %)")
    print(f"  SCCA 덮음 {D.SCCA_덮음.mean():.3f} · 크기비 {D.SCCA_크기비.median():.2f}"
          f" · 아래삐짐 {D.SCCA_아래삐짐.mean():.3f}")
    print(f"  BC   덮음 {D.BC_덮음.mean():.3f} · 아래삐짐 {D.BC_아래삐짐.mean():.3f}")
    print(f"  공막포함 {D.공막포함.mean():.3f} · 공막 홀로 아래삐짐 {D.공막_아래삐짐.mean():.3f}")
    print(f"  뺀 몫  홍채 {D.뺀_홍채.mean()*100:.1f}% · 속눈썹 {D.뺀_속눈썹.mean()*100:.1f}%"
          f" · 바닥 {D.뺀_바닥.mean()*100:.1f}% · 부스러기 {D.뺀_부스러기.mean()*100:.1f}%")
    print(f"  BC 부스러기  버린 몫 {D.뺀_부스러기_BC.mean()*100:.2f}% · 조각 수 중앙 {D.조각_BC원본.median():.0f}"
          f" · 5 % 넘게 버린 장 {int((D.뺀_부스러기_BC>0.05).sum())}")
    for g, s in D.groupby("group"):
        print(f"    [{g}] {len(s):5d}장  덮음 {s.SCCA_덮음.mean():.3f}"
              f" · 아래삐짐 {s.SCCA_아래삐짐.mean():.3f} · 갈아탐 {s.갈아탐.mean()*100:.0f}%")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["internal", "external"], default="internal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=430)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--device", type=int, default=0)
    main(ap.parse_args())
