# -*- coding: utf-8 -*-
"""★ NiADA 식 입력 BC_wb — 공막으로 흰색보정한 결막   2026-09-13

무엇을 하나
    사진마다 **공막(자동분할 'sclera' 마스크)의 평균색**을 재서, 원본 사진 전체에
    채널별 이득 g_c = (공막 R·G·B 평균의 평균) / (공막 채널 c 평균) 을 곱한다 (von Kries).
    = 채널마다 **공막 평균색으로 나누는 것**. 그 뒤 결막(BC) 마스크로 잘라 합성한다.
    합성 방식(마스크 밖 검정 · bbox 자르기)은 auto_bc 와 똑같다 — 다른 것은 흰색보정뿐이다.

왜
    Das et al. (NiADA, Cureus 2024) 는 공막을 흰색 기준으로 쓰고 결막을 모델에 넣는다.
    BC vs BC_wb vs SCCA 를 같은 백본·같은 절차로 비교해
    "공막은 기준인가, 추가 정보인가" 를 딥러닝 조건에서 가린다.

결과
    data/internal/auto_bcwb/*.jpg  (1,524)   data/external_{india,italy}/auto_bcwb/*.jpg (202)
    data/_bcwb_이득.csv  — 사진별 공막 화소 수와 채널 이득 (공막이 너무 작으면 보정하지 않고 적는다)

사용
    python segmentation/analysis/260913_build_bcwb.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent; SEG = HERE.parent; ROOT = SEG.parent
sys.path.insert(0, str(ROOT / "code"))
from common.roi_lib import compose_from_mask, read_raw   # noqa: E402

SRC = Path("PATH/TO/PROJECT/260822_eClinicalMedicine_filtered_2_mobilenet_sccaVSbc")
OLD = Path("PATH/TO/PROJECT/2608_eClinicalMedicine_SCCA_imageOnly_filtered_2")
MASKS = [SRC / "results/260909_final3/mask", SEG / "results/260909_final3/mask",
         SEG / "cand_260911/results/260909_final3/mask"]
D = ROOT / "data"
MIN_SCLERA = 2000          # 공막 화소가 이보다 적으면 평균색이 불안정하다 → 보정하지 않고 기록


def mask_of(stem):
    return next((d / f"{stem}.npz" for d in MASKS if (d / f"{stem}.npz").exists()), None)


def index_internal():
    raws, exif = {}, None
    for f in [OLD / "autosegmentation/common/results/pairs_index.json",
              SEG / "results/extra42_index.json", SEG / "cand_260911/results/cand_index.json"]:
        j = json.load(open(f)); exif = j.get("apply_exif", exif)
        for k, v in j["items"].items():
            raws[Path(k).stem] = v["raw"]
    return raws, bool(exif)


def one(stem, raw, exif, out: Path):
    f = mask_of(stem)
    if f is None:
        return dict(stem=stem, 상태="마스크없음")
    z = np.load(f); H, W = z["shape"]
    get = lambda k: np.unpackbits(z[k])[:H * W].reshape(H, W).astype(bool)
    bc, sc = get("bc"), get("sclera")
    img = read_raw(raw, exif).astype(np.float32)
    n = int(sc.sum())
    if n >= MIN_SCLERA:
        mu = img[sc].mean(axis=0)                       # 공막 R,G,B 평균
        g = mu.mean() / np.maximum(mu, 1.0)             # 채널별 이득 = 회색 / 채널 → 공막이 무채색이 된다
        img = np.clip(img * g[None, None, :], 0, 255)
        상태 = "보정"
    else:
        g = np.ones(3, np.float32); 상태 = "공막부족_보정안함"
    c = compose_from_mask(img.astype(np.uint8), bc)
    if c is None:
        return dict(stem=stem, 상태="BC없음", 공막화소=n)
    c.save(out / f"{stem}.jpg", quality=95)
    return dict(stem=stem, 상태=상태, 공막화소=n, gR=float(g[0]), gG=float(g[1]), gB=float(g[2]))


def main():
    rows = []
    # 안쪽
    raws, exif = index_internal()
    M = pd.read_csv(D / "internal/명부.csv"); M.columns = [c.lstrip("﻿") for c in M.columns]
    out = D / "internal/auto_bcwb"; out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(M.stem.astype(str)):
        r = one(s, raws[s], exif, out) if s in raws else dict(stem=s, 상태="원본없음")
        r["무리"] = "internal"; rows.append(r)
        if (i + 1) % 200 == 0: print(f"  안쪽 [{i+1}/{len(M)}]", flush=True)
    M["auto_bcwb"] = [str(out / f"{s}.jpg") if (out / f"{s}.jpg").exists() else "" for s in M.stem.astype(str)]
    M.to_csv(D / "internal/명부.csv", index=False, encoding="utf-8-sig")
    # 바깥쪽
    pj = json.load(open(OLD / "autosegmentation/external/results/external_index.json"))
    xraws = {Path(k).stem: v["raw"] for k, v in pj["items"].items()}; xexif = bool(pj["apply_exif"])
    for 폴더 in ["external_india", "external_italy"]:
        X = pd.read_csv(D / 폴더 / "명부.csv"); X.columns = [c.lstrip("﻿") for c in X.columns]
        out = D / 폴더 / "auto_bcwb"; out.mkdir(parents=True, exist_ok=True)
        for k in X.key.astype(str):
            s = k.replace(".jpg", "")
            r = one(s, xraws[s], xexif, out) if s in xraws else dict(stem=s, 상태="원본없음")
            if (out / f"{s}.jpg").exists() and s + ".jpg" != k:
                (out / f"{s}.jpg").rename(out / k)
            r["무리"] = 폴더; rows.append(r)
        X["auto_bcwb"] = [str(out / k) if (out / k).exists() else "" for k in X.key.astype(str)]
        X.to_csv(D / 폴더 / "명부.csv", index=False, encoding="utf-8-sig")
        print(f"  {폴더} 끝", flush=True)
    T = pd.DataFrame(rows); T.to_csv(D / "_bcwb_이득.csv", index=False, encoding="utf-8-sig")
    print("\n★ 상태별:", T.groupby(["무리", "상태"]).size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
