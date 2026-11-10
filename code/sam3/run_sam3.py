"""
SAM 3 로 공막(WC)·결막(BC) 을 찾아보고, 사람이 그린 것과 비교한다.

★ 무엇을 SAM 에 넣는가 — **원본 사진이다. 합성 이미지가 아니다.**
    합성 이미지(검게 칠한 것)는 SAM 의 **출력**이지 입력이 아니다.
    이미 오려진 것에서 오려낼 곳을 찾는 건 답을 보고 문제를 푸는 셈이고,
    배포 상황(폰으로 찍은 사진 한 장)과도 맞지 않는다.

프롬프트 4가지 — 상한선과 배포를 반드시 나눠서 잰다

    [상한선] 정답을 프롬프트로 준다. 실제 배포에서는 쓸 수 없다.
      point  정답 폴리곤의 무게중심 점 1개        Sam3TrackerModel  (PVS)
      box    정답 폴리곤의 바운딩 박스            Sam3TrackerModel  (PVS)

    [배포]   정답이 필요 없다. 이게 진짜 성능이다.
      text   "inner lower eyelid" 같은 문구       Sam3Model         (PCS)
      auto   프롬프트 없이 전부 뽑고 규칙으로 고름  pipeline("mask-generation")

    ⚠️ 상한선만 보고하면 과대평가다. 논문에는 둘 다 싣는다.

평가 3층 — ★ Dice 만 보면 틀린 결론이 나온다
    1층 Dice / IoU          사람 주석과 얼마나 겹치나
    2층 ROI 안 색 통계 차이  ★ 모양이 달라도 색이 같으면 Hb 예측에는 문제없다
    3층 Hb 예측 차이         별도 스크립트 (end-to-end)

    정답 폴리곤은 꼭짓점 중앙값 11개로 매우 거칠다. SAM 이 사람보다 정확해도
    Dice 는 낮게 나온다. **우리 모델은 색을 보지 모양을 보지 않는다.**

사용법
    conda activate torch271
    python autosegmentation/sam3_stage1/run_sam3.py --mode point --n 40
    python autosegmentation/sam3_stage1/run_sam3.py --mode text --n 40 \
           --text-wc "white of the eye" --text-bc "inner lower eyelid"

산출물 (모드별)
    results/{mode}/overlay/*.png   ★ 눈으로 확인하는 그림 (원본 | 정답 | SAM | 겹침)
    results/{mode}/metrics.csv     사진별 Dice/IoU/색차이
    results/{mode}/summary.txt
    results/{mode}/masks/*.npz     SAM 마스크 (end-to-end 에서 재사용)

참조: strategy/260819_autosegmentation_guide1.md §4
      strategy/260820_autosegmentation_guide1.md §3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
AUTOSEG = HERE.parent
sys.path.insert(0, str(AUTOSEG / "common"))
from roi_lib import polygon_mask, read_raw  # noqa: E402

INDEX = AUTOSEG / "common" / "results" / "pairs_index.json"
SUBSET = HERE / "results" / "subset.csv"
MODEL_ID = "facebook/sam3"

# 눈으로 확인할 때 쓰는 색
COL = {"gt": (60, 200, 90), "sam": (255, 90, 90)}

# 한글 라벨이 네모로 깨지지 않게 폰트를 직접 지정한다 (PIL 기본 폰트는 한글이 없다)
def _font(size=22):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf",
              "/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def dice_iou(a: np.ndarray, b: np.ndarray):
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = sa + sb - inter
    return (2 * inter / (sa + sb) if sa + sb else np.nan,
            inter / union if union else np.nan)


def roi_color(img: np.ndarray, mask: np.ndarray):
    """★ 2층 지표. 모양이 아니라 **모델이 실제로 보는 것**을 잰다."""
    if mask.sum() < 50:
        return dict(n_px=int(mask.sum()))
    px = img[mask].astype(np.float64)
    R, G, B = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    return dict(n_px=int(mask.sum()), R=R, G=G, B=B,
                logRG=float(np.log((R + 1) / (G + 1))),
                bright=float(px.mean()))


# ---------------------------------------------------------------------------
# 눈으로 확인할 그림
# ---------------------------------------------------------------------------
def overlay(img: np.ndarray, gt: np.ndarray, sam: np.ndarray, title: str, dst: Path,
            width=560):
    """원본 | 정답 | SAM | 겹침  네 칸을 가로로 붙인다."""
    def tint(base, m, color, alpha=0.45):
        o = base.copy().astype(np.float32)
        o[m] = o[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
        return o.astype(np.uint8)

    # 관심 영역만 잘라서 크게 본다 (전체 사진은 눈이 작게 나온다)
    ys, xs = np.nonzero(gt | sam)
    if len(xs) == 0:
        return
    pad = int(0.35 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 40)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    c = (slice(y0, y1), slice(x0, x1))
    base, g, s = img[c], gt[c], sam[c]

    both = tint(tint(base, g, COL["gt"]), s, COL["sam"], 0.40)
    panels = [base, tint(base, g, COL["gt"]), tint(base, s, COL["sam"]), both]
    labels = ["원본", "정답(사람)", "SAM 3", "겹침"]

    h = int(width * base.shape[0] / base.shape[1])
    ims = [Image.fromarray(p).resize((width, h)) for p in panels]
    f, fs = _font(22), _font(20)
    W = Image.new("RGB", (width * 4 + 30, h + 62), (25, 25, 25))
    for i, im in enumerate(ims):
        W.paste(im, (i * (width + 10), 32))
    d = ImageDraw.Draw(W)
    d.text((6, 5), f"{title}    초록 = 사람이 그린 것    빨강 = SAM 3", fill=(240, 240, 240), font=f)
    for i, lab in enumerate(labels):
        d.text((i * (width + 10) + 6, h + 38), lab, fill=(205, 205, 205), font=fs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    W.save(dst, quality=88)


# ---------------------------------------------------------------------------
def load_models(mode, device):
    if mode in ("point", "box"):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        m = Sam3TrackerModel.from_pretrained(MODEL_ID).to(device).eval()
        return m, Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    if mode == "text":
        from transformers import Sam3Model, Sam3Processor
        m = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
        return m, Sam3Processor.from_pretrained(MODEL_ID)
    raise ValueError(mode)


@torch.no_grad()
def predict(mode, model, proc, img_pil, poly, device, text=None, thr=0.4):
    """SAM 마스크 후보들을 (masks, scores) 로 돌려준다."""
    if mode in ("point", "box"):
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        if mode == "point":
            kw = dict(input_points=[[[[float(np.mean(xs)), float(np.mean(ys))]]]],
                      input_labels=[[[1]]])
        else:
            kw = dict(input_boxes=[[[min(xs), min(ys), max(xs), max(ys)]]])
        inp = proc(images=img_pil, return_tensors="pt", **kw).to(device)
        out = model(**inp, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        masks = masks.squeeze(0).numpy() > 0
        return masks, out.iou_scores.cpu().numpy().flatten()

    inp = proc(images=img_pil, text=text, return_tensors="pt").to(device)
    out = model(**inp)
    res = proc.post_process_instance_segmentation(
        out, threshold=thr, mask_threshold=0.5,
        target_sizes=inp.get("original_sizes").tolist())[0]
    if len(res["masks"]) == 0:
        return np.zeros((0, img_pil.height, img_pil.width), bool), np.array([])
    return res["masks"].cpu().numpy() > 0, res["scores"].cpu().numpy()


def main(a):
    device = f"cuda:{a.device}"
    out_root = HERE / "results" / a.mode
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    idx = json.load(open(INDEX))
    items, apply_exif = idx["items"], idx["apply_exif"]
    sub = pd.read_csv(SUBSET)
    if a.n:
        sub = sub.groupby("number").head(2).head(a.n)

    model, proc = load_models(a.mode, device)
    TEXT = {"WC": a.text_wc, "BC": a.text_bc}
    rows = []
    for i, r in sub.reset_index(drop=True).iterrows():
        it = items[r.key]
        img = read_raw(it["raw"], apply_exif)
        img_pil = Image.fromarray(img)
        for cls in a.classes:
            poly = it["polygons"][cls]
            gt = polygon_mask(poly, (img.shape[1], img.shape[0]))
            masks, scores = predict(a.mode, model, proc, img_pil, poly, device,
                                    TEXT.get(cls), a.threshold)
            if len(masks) == 0:
                rows.append(dict(key=r.key, cls=cls, Hb=r.Hb, n_cand=0, found=False))
                continue
            ds = [dice_iou(m, gt)[0] for m in masks]
            top = int(np.argmax(scores)) if len(scores) else 0   # 모델이 고른 것
            best = int(np.nanargmax(ds))                          # 정답 대비 최선
            sel = masks[top]
            d_top, i_top = dice_iou(sel, gt)
            cg, cs = roi_color(img, gt), roi_color(img, sel)
            rows.append(dict(
                key=r.key, cls=cls, Hb=r.Hb, n_cand=len(masks), found=True,
                score=float(scores[top]) if len(scores) else np.nan,
                dice=d_top, iou=i_top, dice_oracle=float(ds[best]),
                area_ratio=sel.sum() / max(gt.sum(), 1),
                **{f"gt_{k}": v for k, v in cg.items()},
                **{f"sam_{k}": v for k, v in cs.items()}))
            np.savez_compressed(out_root / "masks" / f"{r.key[:-4]}_{cls}.npz",
                                mask=np.packbits(sel), shape=sel.shape)
            if i < a.n_overlay:
                overlay(img, gt, sel, f"{r.key}  {cls}  Hb={r.Hb}  Dice={d_top:.3f}",
                        out_root / "overlay" / f"{r.key[:-4]}_{cls}.jpg")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)

    lines = []
    def say(*x):
        s = " ".join(str(i) for i in x); print(s); lines.append(s)
    say("=" * 74)
    say(f"SAM 3 · 모드 {a.mode}" + (f" · WC='{a.text_wc}' BC='{a.text_bc}'"
                                    if a.mode == "text" else ""))
    say(f"사진 {sub.key.nunique()}장 · 원본 사진 입력 · EXIF 적용={apply_exif}")
    say("=" * 74)
    for cls, g in df.groupby("cls"):
        f = g[g.found == True]
        say(f"\n[{cls}]  검출 {len(f)}/{len(g)}")
        if not len(f):
            say("  ❌ 하나도 못 찾았다"); continue
        say(f"  1층 Dice {f.dice.mean():.3f} (중앙 {f.dice.median():.3f})   "
            f"IoU {f.iou.mean():.3f}   상한(정답대비 최선) {f.dice_oracle.mean():.3f}")
        say(f"      면적비 SAM/정답 {f.area_ratio.mean():.2f}  후보수 {f.n_cand.mean():.1f}")
        if "sam_bright" in f and f.sam_bright.notna().any():
            v = f.dropna(subset=["sam_bright", "gt_bright"])
            say(f"  2층 ★ 색 차이 (모델이 실제로 보는 것)")
            say(f"      밝기  정답 {v.gt_bright.mean():6.1f}  SAM {v.sam_bright.mean():6.1f}  "
                f"차이 {abs(v.sam_bright-v.gt_bright).mean():5.2f}")
            say(f"      log(R/G) 정답 {v.gt_logRG.mean():.4f}  SAM {v.sam_logRG.mean():.4f}  "
                f"차이 {abs(v.sam_logRG-v.gt_logRG).mean():.4f}")
    say(f"\n저장: {out_root}")
    say(f"  ★ 눈으로 확인 → {out_root/'overlay'} (원본 | 정답 | SAM | 겹침)")
    (out_root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["point", "box", "text"])
    ap.add_argument("--classes", nargs="+", default=["WC", "BC"])
    ap.add_argument("--n", type=int, default=40, help="사진 수 (0=전체)")
    ap.add_argument("--n-overlay", type=int, default=20, help="그림으로 저장할 장수")
    ap.add_argument("--text-wc", default="white of the eye")
    ap.add_argument("--text-bc", default="inner lower eyelid")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--device", default="0")
    main(ap.parse_args())
