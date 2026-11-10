# -*- coding: utf-8 -*-
"""외부 202장의 4단 검토 그림을 만든다 — BC(결막 단독) 또는 SCCA(공막+결막).

왜 필요한가
    BC 와 SCCA(WCBC) 를 각각 눈으로 감사해야 한다.
    BC 쪽에 실패가 몰려 있지만, SCCA 마스크가 멀쩡한지 확인해야
    "BC만 망가졌고 SCCA 입력은 살아 있었다" 가 성립한다.

만드는 것
    원본 | 사람이 그린 마스크 | SAM 마스크 | 겹침
    그리고 **홍채 빛반사 지표**를 새로 잰다 (사용자 관찰: 반짝이는 곳을 공막으로 잡는다).

사용법
    conda activate torch271
    python analysis/render_bc_overlay.py                    # BC (기본)
    python analysis/render_bc_overlay.py --target scca      # SCCA(WCBC)
    python analysis/render_bc_overlay.py --limit 5           # 맛보기
산출물
    BC   : results/mask_review/panel_bc/{key}.jpg
    SCCA : results/mask_review/panel_scca/{key}.jpg
    (공통) results/mask_review/qc_extra.csv     (빛반사 지표 — BC 기준)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OLD = Path("PATH/TO/PROJECT/2608_eClinicalMedicine_SCCA_imageOnly_filtered_2")
EXT = OLD / "autosegmentation" / "external"
WORK = EXT / "results" / "sam_roi_external"
INDEX = EXT / "results" / "external_index.json"
OUT = Path(__file__).resolve().parent.parent / "results" / "mask_review"

COL = {"gt": (60, 230, 90), "sam": (250, 70, 70), "sam2": (70, 150, 255)}
FONTS = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _font(sz):
    for f in FONTS:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def polygon_mask(points, size):
    """폴리곤을 불리언 마스크로. ★ points 는 **점의 목록** 이다 (다각형의 목록이 아니다).
    원본 `autosegmentation/common/roi_lib.py:109` 와 동일하게 맞춘다 —
    다르게 그리면 사람 ROI 가 통째로 비어 조용히 틀린 그림이 나온다 (2026-08-24 에 겪음)."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=255)
    return np.array(m) > 0


def read_raw(path, apply_exif):
    im = Image.open(path)
    if apply_exif:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    return np.array(im.convert("RGB"))


def unpack(z, key, shape):
    return np.unpackbits(z[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def glint_fraction(img, m):
    """★ 홍채 빛반사 지표 (2026-08-24 추가).

    사용자 관찰 : 홍채 위의 **반짝이는 밝은 점**을 공막으로 잡는 실패가 있다.
    어두운 화소 비율로는 이것을 못 잡는다 — 반사는 밝기 때문이다.
    포화에 가까운 밝고 채도 낮은 화소(specular highlight)의 비율을 센다.
    """
    if m.sum() == 0:
        return np.nan
    x = img[m].astype(np.float32)
    v = x.max(1)
    s = np.where(v > 0, (v - x.min(1)) / np.maximum(v, 1), 0)
    return float(((v > 245) & (s < 0.15)).mean())


def panel(img, gt, sam, title, dst, width=560, labels=None, sam2=None):
    """원본 | 사람 | SAM | 겹침 — 기존 overlay 와 같은 형식.
    labels 를 주면 [원본, 사람라벨, SAM라벨, 겹침] 로 바뀐다.

    sam2 를 주면 **5단**이 된다 (2026-08-26, 분할기 A 검토용):
        원본 | 사람 | SAM 1단계 | SAM 2단계 | 겹침(사람+2단계)
    색 : 초록=사람 · 빨강=1단계 · 파랑=2단계.  1단계와 2단계를 같은 색으로 칠하면
         어느 쪽이 좋아졌는지 눈으로 못 가린다."""
    def tint(base, m, c, a=0.45):
        o = base.copy().astype(np.float32)
        o[m] = o[m] * (1 - a) + np.array(c, np.float32) * a
        return o.astype(np.uint8)
    allm = gt | sam if sam2 is None else (gt | sam | sam2)
    ys, xs = np.nonzero(allm)
    if len(xs) == 0:
        return False
    pad = int(0.35 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 40)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    c = (slice(y0, y1), slice(x0, x1))
    base, g, s = img[c], gt[c], sam[c]
    if sam2 is None:
        both = tint(tint(base, g, COL["gt"]), s, COL["sam"], 0.40)
        panels = [base, tint(base, g, COL["gt"]), tint(base, s, COL["sam"]), both]
        if labels is None:
            labels = ["원본", "사람", "SAM", "겹침"]
    else:
        s2 = sam2[c]
        both = tint(tint(base, g, COL["gt"]), s2, COL["sam2"], 0.40)
        panels = [base, tint(base, g, COL["gt"]), tint(base, s, COL["sam"]),
                  tint(base, s2, COL["sam2"]), both]
        if labels is None or len(labels) < 5:
            labels = ["원본", "사람", "SAM 1단계 (현행)", "SAM 2단계 (크롭·확대)",
                      "겹침 — 초록 사람 · 파랑 2단계"]
    n = len(panels)
    h = int(width * base.shape[0] / base.shape[1])
    ims = [Image.fromarray(p).resize((width, h)) for p in panels]
    f, fs = _font(22), _font(20)
    W = Image.new("RGB", (width * n + 10 * (n - 1) + 10, h + 62), (25, 25, 25))
    for i, im in enumerate(ims):
        W.paste(im, (i * (width + 10), 34))
    d = ImageDraw.Draw(W)
    d.text((8, 6), title, font=f, fill=(240, 240, 240))
    for i, lb in enumerate(labels):
        d.text((i * (width + 10) + 8, h + 38), lb, font=fs, fill=(190, 190, 190))
    dst.parent.mkdir(parents=True, exist_ok=True)
    W.save(dst, quality=88)
    return True


def main(a):
    # ★ target 에 따라 마스크·출력 폴더·라벨·QC 저장 여부가 바뀐다.
    #   bc   : SAM BC  vs 사람 BC   → panel_bc   · qc_extra.csv 갱신
    #   scca : SAM SCCA vs 사람 WCBC → panel_scca · qc_extra.csv 갱신 안 함 (BC 기준이므로)
    TARGETS = {
        "bc":   dict(mask_key="bc",   poly_key="BC",   folder="panel_bc",
                     labels=["원본", "사람이 그린 BC",   "SAM BC",   "겹침"],
                     tag="BC",   write_qc=True),
        "scca": dict(mask_key="scca", poly_key="WCBC", folder="panel_scca",
                     labels=["원본", "사람이 그린 SCCA", "SAM SCCA", "겹침"],
                     tag="SCCA", write_qc=False),
    }
    if a.target not in TARGETS:
        raise SystemExit(f"[ERROR] --target 은 {list(TARGETS)} 중 하나")
    cfg = TARGETS[a.target]
    panel_dir = OUT / cfg["folder"]

    if not INDEX.exists():
        raise SystemExit(f"[ERROR] 색인이 없다: {INDEX}")
    idx = json.load(open(INDEX))
    items, master, ex = idx["items"], idx["master"], idx["apply_exif"]
    keys = sorted(items)
    if a.limit:
        keys = keys[: a.limit]

    chk = pd.read_excel(WORK / "check_external.xlsx").set_index("filename")
    rows, made = [], 0
    for n, key in enumerate(keys, 1):
        it = items[key]
        mp = WORK / "mask" / f"{key[:-4]}.npz"
        if not mp.exists():
            continue
        z = np.load(mp)
        shape = tuple(z["shape"])
        m_bc = unpack(z, "bc", shape)
        m_wc = unpack(z, "scca", shape)          # scca = WCBC(공막+결막) 합집합
        img = read_raw(it["raw"], ex)
        sz = (img.shape[1], img.shape[0])

        # ★ target 에 따라 사람 폴리곤·SAM 마스크·라벨 을 고른다.
        gt = polygon_mask(it["polygons"][cfg["poly_key"]], sz)
        m_target = m_wc if a.target == "scca" else m_bc

        r = chk.loc[key] if key in chk.index else None
        if r is not None:
            ttl = (f"{key} [{it['site']}]  {cfg['tag']}  "
                   f"Dice={r['dice_bc']:.3f}  면적비={r['area_ratio_bc']:.2f}  "
                   f"Hb={master[key]['Hb']}")
        else:
            ttl = f"{key} [{it['site']}] {cfg['tag']}"
        dst = panel_dir / f"{key[:-4]}.jpg"
        if (not dst.exists() or a.force) and panel(
                img, gt, m_target, ttl, dst, labels=cfg["labels"]):
            made += 1

        # ★ QC 지표(qc_extra.csv) 는 BC 기준으로만 저장한다.
        #   SCCA 런에서는 마스크를 다시 계산하지 않고 건너뛴다.
        if cfg["write_qc"]:
            m_wconly = m_wc & ~m_bc
            rows.append(dict(filename=key, site=it["site"],
                             공막결막면적비=float(m_wconly.sum() / max(m_bc.sum(), 1)),
                             공막비중=float(m_wconly.sum() / max(m_wc.sum(), 1)),
                             빛반사_WC=glint_fraction(img, m_wc),
                             빛반사_BC=glint_fraction(img, m_bc),
                             BC_아래쪽비중=float(np.nonzero(m_bc)[0].mean() / shape[0])
                             if m_bc.sum() else np.nan,
                             사람BC_아래쪽비중=float(np.nonzero(gt)[0].mean() / shape[0])
                             if gt.sum() else np.nan))
        if n % 25 == 0:
            print(f"  {n}/{len(keys)}")

    OUT.mkdir(parents=True, exist_ok=True)
    if cfg["write_qc"] and rows:
        df = pd.DataFrame(rows)
        df["BC_중심하강"] = df["BC_아래쪽비중"] - df["사람BC_아래쪽비중"]
        df.round(4).to_csv(OUT / "qc_extra.csv", index=False, encoding="utf-8-sig")
        print(f"\n추가 QC → {OUT/'qc_extra.csv'}")
        print(f"  빛반사_WC  중앙 {df.빛반사_WC.median():.4f}  "
              f"상위5% {df.빛반사_WC.quantile(.95):.4f}")
        print(f"  BC_중심하강 중앙 {df.BC_중심하강.median():+.4f}  "
              f"상위5% {df.BC_중심하강.quantile(.95):+.4f}")
        print("    (양수 = SAM 이 사람보다 아래를 잡았다 → 피부를 물었을 가능성)")

    print(f"\n[{cfg['tag']}] 그림 {made}장 생성 → {panel_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="bc", choices=["bc", "scca"],
                    help="bc(결막 단독, 기본) | scca(공막+결막 WCBC)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
