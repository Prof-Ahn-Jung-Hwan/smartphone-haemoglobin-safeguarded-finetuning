"""
색인에 있는 사진 전부에 SAM 3 를 돌려 자동 ROI 를 만들고, 눈으로 확인할 자료를 남긴다.

만드는 것
    SCCA 입력 = SAM("lower eyelid inner surface")
              ∪ SAM("exposed white sclera surrounding the iris")
              -> 구멍 메우기(morphological closing)
    BC   입력 = SAM("lower eyelid inner surface")

    두 프롬프트를 합치는 이유: 우리 SCCA 입력은 WC 가 아니라 **WCBC**(공막+결막 합친 영역)다.
    한 프롬프트로는 Dice 0.62 인데 합치면 0.74 가 된다 (guide §3-4 발견 2-3).

저장
    autosegmentation/sam3_stage1/results/sam_roi_260820/
    ├── used/      Hb 학습에 쓰이는 사진
    │   ├── overlay/          ★ 눈으로 확인 (원본 | 사람 | SAM | 겹침)
    │   ├── composite_scca/     모델 입력 (SAM WCBC)
    │   ├── composite_bc/       모델 입력 (SAM BC)
    │   └── mask/               .npz
    ├── unused/    그 외 (Hb 엑셀에 없는 사진)
    └── check_used.csv / check_unused.csv / check.xlsx

★ check.xlsx 는 **위험도 순으로 정렬**되어 있다. 위에서부터 보면 나쁜 것이 먼저 나온다.
   '판정' 칸에 OK / NG 를 적어가며 확인하면 된다.

사용법
    conda activate torch271
    python autosegmentation/sam3_stage1/make_sam_roi.py            # 전체 (~2시간)
    python autosegmentation/sam3_stage1/make_sam_roi.py --limit 30 # 맛보기

참조: strategy/260820_autosegmentation_guide1.md §3-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
AUTOSEG = HERE.parent
sys.path.insert(0, str(AUTOSEG / "common"))
from roi_lib import ANNO_XLSX, compose_from_mask, polygon_mask, read_raw  # noqa: E402

INDEX = AUTOSEG / "common" / "results" / "pairs_index.json"
WORK = HERE / "results" / "sam_roi_260820"   # 프로젝트 폴더 안 (시놀로지 동기화됨 — 연구용이라 무방)
MODEL_ID = "facebook/sam3"

P_BC = "lower eyelid inner surface"
# ★ 공막은 프롬프트 하나로 부족하다 (2026-08-20 표본 100장 실측)
#   'exposed white sclera surrounding the iris' : 실패 5% — Dice 는 더 높지만 가끔 못 찾는다
#   'white part of the eye'                     : 실패 0% — 조금 무디지만 항상 찾는다
#   실패한 사진은 **공막이 실제로 작은 사진**이다 (WCBC 대비 0.127 대 0.272, p=0.0004).
#   -> 1순위로 좋은 것을 쓰고, 못 찾으면 2순위로 받는다. 둘 다 실패한 사진은 0장이었다.
P_WC = ["exposed white sclera surrounding the iris", "white part of the eye"]
COL = {"gt": (60, 200, 90), "sam": (255, 90, 90)}


def _font(size=22):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 0.0


def col_stats(img, m):
    if m.sum() < 50:
        return dict(bright=np.nan, logRG=np.nan)
    px = img[m].astype(np.float64)
    R, G = px[:, 0].mean(), px[:, 1].mean()
    return dict(bright=float(px.mean()), logRG=float(np.log((R + 1) / (G + 1))))


def dark_fraction(img, mask, thr=60):
    """마스크 안에서 '어두운 픽셀'(속눈썹·그림자)이 차지하는 비율.

    사용자 관찰(2026-08-20): *속눈썹이 진하면서 공막을 1/3 이상 덮으면 SAM 이 공막을 잘 못 찾는다.*
    이 값이 그 가설을 검증할 수 있게 해 준다. 속눈썹은 매우 어두워서 절대 문턱으로 잡힌다.
    """
    if mask.sum() < 50:
        return np.nan
    v = img[mask].mean(axis=1)
    return float((v < thr).mean())


def remove_dark(img, mask, thr=60, min_keep=0.3):
    """속눈썹·그림자를 마스크에서 뺀다. 합성 이미지에서는 그 자리가 검게 남는다.

    ⚠️ 이건 **ROI 정의를 바꾸는 설계 결정**이다. 몰래 쓰지 말 것.
       지금은 '정리본(clean)' 을 따로 만들어 두기만 하고, 어느 쪽이 Hb 를 잘 맞히는지는
       재학습으로 판정한다 (guide §7 ROI 정의 비교).
       너무 많이 지워지면(남는 비율 < min_keep) 원본 마스크를 그대로 쓴다.
    """
    v = img.mean(axis=2)
    out = mask & (v >= thr)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    if out.sum() < min_keep * mask.sum():
        return mask, False
    return out, True


def close_holes(mask):
    """두 프롬프트 사이의 틈을 메운다. 커널은 ROI 크기에 비례시킨다."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    k = max(21, int(0.05 * (ys.max() - ys.min() + 1)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker) > 0


def overlay(img, gt, sam, title, dst, width=520):
    def tint(base, m, c, a=0.45):
        o = base.copy().astype(np.float32)
        o[m] = o[m] * (1 - a) + np.array(c, np.float32) * a
        return o.astype(np.uint8)
    ys, xs = np.nonzero(gt | sam)
    if len(xs) == 0:
        return
    pad = int(0.3 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 40)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    c = (slice(y0, y1), slice(x0, x1))
    base, g, s = img[c], gt[c], sam[c]
    panels = [base, tint(base, g, COL["gt"]), tint(base, s, COL["sam"]),
              tint(tint(base, g, COL["gt"]), s, COL["sam"], 0.40)]
    labels = ["원본", "사람이 그린 것", "SAM 3", "겹침"]
    h = int(width * base.shape[0] / base.shape[1])
    f, fs = _font(21), _font(19)
    W = Image.new("RGB", (width * 4 + 30, h + 60), (25, 25, 25))
    for i, p in enumerate(panels):
        W.paste(Image.fromarray(p).resize((width, h)), (i * (width + 10), 30))
    d = ImageDraw.Draw(W)
    d.text((6, 4), f"{title}   초록 = 사람   빨강 = SAM 3", fill=(240, 240, 240), font=f)
    for i, lab in enumerate(labels):
        d.text((i * (width + 10) + 6, h + 36), lab, fill=(205, 205, 205), font=fs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    W.save(dst, quality=85)


# ---------------------------------------------------------------------------
def risk_flags(r):
    """왜 의심스러운지 사람이 읽을 수 있는 말로 적는다."""
    f = []
    if r["n_obj_bc"] == 0:
        f.append("결막 못찾음")
    if r["n_obj_wc"] == 0:
        f.append("공막 못찾음")
    elif r.get("공막출처") == "2순위(대체)":
        f.append("공막 대체프롬프트")
    if r["dice_wcbc"] < 0.50:
        f.append(f"WCBC겹침낮음({r['dice_wcbc']:.2f})")
    if r["dice_bc"] < 0.50:
        f.append(f"BC겹침낮음({r['dice_bc']:.2f})")
    if r["area_ratio_wcbc"] > 3.0:
        f.append(f"너무넓음({r['area_ratio_wcbc']:.1f}배)")
    if r["area_ratio_wcbc"] < 0.4:
        f.append(f"너무좁음({r['area_ratio_wcbc']:.1f}배)")
    if abs(r["d_bright_wcbc"]) > 25:
        f.append(f"밝기차큼({r['d_bright_wcbc']:.0f})")
    if r.get("어두운픽셀비율", 0) > 0.20:
        f.append(f"속눈썹많음({r['어두운픽셀비율']:.0%})")
    return " · ".join(f)


def write_xlsx(dfs, dst):
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    with pd.ExcelWriter(dst, engine="openpyxl") as w:
        for name, df in dfs.items():
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = "C2"
            for i, c in enumerate(df.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = \
                    36 if c in ("filename", "왜의심스러운가", "overlay") else 13
            n = len(df) + 1
            for c in ("dice_wcbc", "dice_bc"):
                if c in df.columns:
                    L = get_column_letter(df.columns.get_loc(c) + 1)
                    ws.conditional_formatting.add(
                        f"{L}2:{L}{n}", ColorScaleRule(
                            start_type="num", start_value=0.3, start_color="F8696B",
                            mid_type="num", mid_value=0.6, mid_color="FFEB84",
                            end_type="num", end_value=0.85, end_color="63BE7B"))
            if "판정" in df.columns:
                L = get_column_letter(df.columns.get_loc("판정") + 1)
                dv = DataValidation(type="list", formula1='"OK,NG,애매"', allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{L}2:{L}{n}")


@torch.no_grad()
def main(a):
    from transformers import Sam3Model, Sam3Processor

    idx = json.load(open(INDEX))
    items, ex = idx["items"], idx["apply_exif"]
    used = set(Path(p).name for p in pd.read_excel(ANNO_XLSX)["file_path"])
    keys = sorted(items)
    if a.limit:
        keys = keys[:a.limit]

    device = f"cuda:{a.device}"
    model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    proc = Sam3Processor.from_pretrained(MODEL_ID)

    def seg(pil, text):
        inp = proc(images=pil, text=text, return_tensors="pt").to(device)
        out = model(**inp)
        R = proc.post_process_instance_segmentation(
            out, threshold=a.threshold, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())[0]
        if len(R["masks"]) == 0:
            return None, 0, np.nan
        ms = R["masks"].cpu().numpy() > 0
        sc = R["scores"].cpu().numpy()
        k = int(np.argmax(sc))
        return ms[k], len(ms), float(sc[k])

    rows = []
    for i, key in enumerate(keys):
        it = items[key]
        grp = "used" if key in used else "unused"
        root = WORK / grp
        ov = root / "overlay" / f"{key[:-4]}.jpg"
        if ov.exists() and not a.force:
            continue
        img = read_raw(it["raw"], ex)
        pil = Image.fromarray(img)
        sz = (img.shape[1], img.shape[0])
        gt = {c: polygon_mask(it["polygons"][c], sz) for c in ("WCBC", "BC")}

        m_bc, n_bc, s_bc = seg(pil, P_BC)
        # 공막: 1순위 -> 실패하면 2순위 (어느 것을 썼는지 기록한다)
        wc_src = ""
        m_wc, n_wc, s_wc = None, 0, np.nan
        for rank, pw in enumerate(P_WC):
            m_wc, n_wc, s_wc = seg(pil, pw)
            if m_wc is not None:
                wc_src = "1순위" if rank == 0 else "2순위(대체)"
                break
        if m_bc is None and m_wc is None:
            rows.append(dict(filename=key, group=grp, anon_id=it["anon"],
                             batch=it["batch"], n_obj_bc=0, n_obj_wc=0, 공막출처="둘다실패",
                             dice_wcbc=0.0, dice_bc=0.0, area_ratio_wcbc=np.nan,
                             area_ratio_bc=np.nan, d_bright_wcbc=np.nan,
                             d_logRG_wcbc=np.nan, score_bc=np.nan, score_wc=np.nan))
            continue
        base = m_bc if m_bc is not None else m_wc
        m_scca = close_holes(base | m_wc) if (m_bc is not None and m_wc is not None) \
            else close_holes(base)
        m_bc_out = m_bc if m_bc is not None else m_scca

        # 정리본 = 속눈썹·그림자 제거. 재학습 arm 후보로 미리 만들어 둔다.
        m_scca_c, ok_s = remove_dark(img, m_scca)
        m_bc_c, ok_b = remove_dark(img, m_bc_out)
        for tag, mk in (("scca", m_scca), ("bc", m_bc_out),
                        ("scca_clean", m_scca_c), ("bc_clean", m_bc_c)):
            c = compose_from_mask(img, mk)
            if c is not None:
                d = root / f"composite_{tag}"
                d.mkdir(parents=True, exist_ok=True)
                c.save(d / key, quality=95)
        (root / "mask").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "mask" / f"{key[:-4]}.npz",
                            scca=np.packbits(m_scca), bc=np.packbits(m_bc_out),
                            scca_clean=np.packbits(m_scca_c), bc_clean=np.packbits(m_bc_c),
                            shape=np.array(m_scca.shape))

        cg, cs = col_stats(img, gt["WCBC"]), col_stats(img, m_scca)
        rec = dict(filename=key, group=grp, anon_id=it["anon"], batch=it["batch"],
                   n_obj_bc=n_bc, n_obj_wc=n_wc, 공막출처=wc_src,
                   score_bc=s_bc, score_wc=s_wc,
                   dice_wcbc=dice(m_scca, gt["WCBC"]), dice_bc=dice(m_bc_out, gt["BC"]),
                   area_ratio_wcbc=m_scca.sum() / max(gt["WCBC"].sum(), 1),
                   area_ratio_bc=m_bc_out.sum() / max(gt["BC"].sum(), 1),
                   d_bright_wcbc=cs["bright"] - cg["bright"],
                   d_logRG_wcbc=cs["logRG"] - cg["logRG"],
                   # 속눈썹 가설 검증용
                   어두운픽셀비율=dark_fraction(img, m_scca),
                   사람ROI어두운비율=dark_fraction(img, gt["WCBC"]),
                   공막비중_사람=float(polygon_mask(it["polygons"]["WC"], sz).sum()
                                      / max(gt["WCBC"].sum(), 1)),
                   정리후남은비율=float(m_scca_c.sum() / max(m_scca.sum(), 1)))
        rows.append(rec)
        overlay(img, gt["WCBC"], m_scca,
                f"{key}  SCCA(WCBC)  Dice={rec['dice_wcbc']:.3f}  "
                f"면적비={rec['area_ratio_wcbc']:.2f}", ov)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)

    if not rows:
        print("새로 만든 것이 없다 (--force 로 다시 만들 수 있다)")
        return
    df = pd.DataFrame(rows)
    # ★ 이어하기 버그 방지 (2026-08-20 발견)
    #   이미 만들어진 사진은 위에서 continue 로 건너뛰므로 rows 에 안 들어간다.
    #   그대로 저장하면 **이번에 처리한 것만 남고 이전 행이 사라진다.**
    #   (실제로 12장짜리 맛보기 실행 뒤 본 실행을 하니 표가 1479 대신 1467행이 됐다.)
    #   그래서 기존 표가 있으면 합친다. 같은 사진은 이번 것이 이긴다.
    prev = []
    for grp in ("used", "unused"):
        f = WORK / f"check_{grp}.csv"
        if f.exists():
            old_df = pd.read_csv(f)
            old_df["group"] = grp
            prev.append(old_df)
    if prev:
        keep = pd.concat(prev, ignore_index=True)
        keep = keep[~keep.filename.isin(set(df.filename))]
        n_before = len(df)
        df = pd.concat([df, keep], ignore_index=True)
        print(f"  기존 표 {len(keep)}행과 합침 (이번 실행 {n_before}행)")
    if "왜의심스러운가" in df.columns:          # 합쳐온 행은 이미 값이 있다
        need = df["왜의심스러운가"].isna()
        df.loc[need, "왜의심스러운가"] = df[need].apply(risk_flags, axis=1)
    else:
        df["왜의심스러운가"] = df.apply(risk_flags, axis=1)
    # 위험도: 겹침이 낮고 면적이 이상할수록 위로
    df["위험도"] = ((1 - df.dice_wcbc.fillna(0)) * 2
                    + (df.area_ratio_wcbc.fillna(9) - 1).abs().clip(0, 5)
                    + (df.d_bright_wcbc.abs().fillna(50) / 25)).round(3)
    if "판정" not in df.columns:
        df["판정"] = ""
    df["판정"] = df["판정"].fillna("")
    df["overlay"] = df.filename.str[:-4] + ".jpg"
    cols = ["위험도", "판정", "filename", "왜의심스러운가", "공막출처", "dice_wcbc", "dice_bc",
            "area_ratio_wcbc", "area_ratio_bc", "d_bright_wcbc", "d_logRG_wcbc",
            "어두운픽셀비율", "사람ROI어두운비율", "공막비중_사람", "정리후남은비율",
            "n_obj_bc", "n_obj_wc", "score_bc", "score_wc", "batch", "anon_id", "overlay"]
    df = df[cols + ["group"]].sort_values("위험도", ascending=False)

    WORK.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for grp, g in df.groupby("group"):
        g = g.drop(columns=["group"])
        g.to_csv(WORK / f"check_{grp}.csv", index=False, encoding="utf-8-sig")
        sheets["Hb사용" if grp == "used" else "그외"] = g
        print(f"\n[{grp}] {len(g)}장  Dice(WCBC) 평균 {g.dice_wcbc.mean():.3f} "
              f"중앙 {g.dice_wcbc.median():.3f}  면적비 {g.area_ratio_wcbc.mean():.2f}")
        print(f"   의심 표시된 것 {int((g['왜의심스러운가'] != '').sum())}장 "
              f"({(g['왜의심스러운가'] != '').mean():.1%})")
        vc = g["공막출처"].value_counts()
        print(f"   공막 출처: " + "  ".join(f"{k} {v}장({v/len(g):.1%})" for k, v in vc.items()))
        print(f"   어두운픽셀(속눈썹) 비율  SAM {g.어두운픽셀비율.mean():.3f}  "
              f"사람 {g.사람ROI어두운비율.mean():.3f}   "
              f"20%% 넘는 사진 {int((g.어두운픽셀비율 > 0.2).sum())}장")
        # ★ 사용자 가설: 속눈썹이 많으면 공막 1순위 프롬프트가 실패하는가
        alt = g["공막출처"] == "2순위(대체)"
        if alt.sum() >= 3:
            from scipy.stats import mannwhitneyu
            try:
                _, pv = mannwhitneyu(g[alt].어두운픽셀비율.dropna(),
                                     g[~alt].어두운픽셀비율.dropna())
                print(f"   ★ 속눈썹 가설: 대체 사용 {g[alt].어두운픽셀비율.mean():.3f} 대 "
                      f"1순위 {g[~alt].어두운픽셀비율.mean():.3f}  p={pv:.4f}")
            except ValueError:
                pass
    write_xlsx(sheets, WORK / "check.xlsx")
    print(f"\n저장: {WORK}")
    print(f"  ★ 확인용 표  {WORK/'check.xlsx'}   (위험도 높은 순 정렬, '판정' 칸에 OK/NG)")
    print(f"  ★ 그림      {WORK}/used/overlay/  ,  {WORK}/unused/overlay/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
