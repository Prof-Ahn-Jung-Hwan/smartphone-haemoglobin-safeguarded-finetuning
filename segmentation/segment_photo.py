# -*- coding: utf-8 -*-
"""Automatic SCCA segmentation from one smartphone eye photograph — the study pipeline in one script.

No human input. SAM 3 (`facebook/sam3`, Hugging Face `transformers`) is prompted with everyday words only.

  stage                SAM prompts (calls per photograph)
  1  first pass        "lower eyelid inner surface" · "white part of the eye" ·
                       "the bright area in the eye" · "pupil" · "eyelashes"                         (5)
  2  conjunctiva       "the reddish area inside the lower eyelid" ·
                       "the pink area inside the lower eyelid"  -> BC = union                        (2)
  3  second pass       "the eye surface next to the iris" · "skin"                                    (2)
                       region, floor/ceiling, ellipse trimming
  4  final             first or second pass chosen without human review:
                       second pass if |first-pass sclera − BC| / |BC| < 0·30;
                       iris (×1.1) and upper eyelashes removed at the end
  5  ROI               outside the mask set to black, cropped to the mask's box
                       (code/common/roi_lib.compose_from_mask) — the model input

Details that are easy to miss (all reproduced here as in the study)
  · "lower eyelid inner surface" (stage 1) is not the final conjunctiva; its mask only sets the floor below
    which first-pass sclera is discarded. The final conjunctiva is the stage-2 union.
  · Stage 3 has its own pass choice (first-pass sclera / BC < 0·90) that only decides which region the
    trimming ellipse is fitted to; the pass actually used is decided in stage 4 with the 0·30 rule.
  · A brightness cut is applied only above the iris's vertical centre (eyelashes); the conjunctiva, below
    the iris, is never cut by brightness.

Usage
  python segmentation/segment_photo.py --images <folder or files> --out <folder> [--device 0]
Outputs (in --out)
  roi_scca/<name>.jpg    sclera + conjunctiva ROI (the model input, JPEG quality 95)
  roi_bc/<name>.jpg      conjunctiva-only ROI
  masks/<name>.npz       bc · sclera · scca · iris (np.packbits, with `shape`)
  segmentation_log.csv   per photograph: pass used (1 or 2), sclera share, fallbacks, pixel counts
Check against reference masks:  --compare <folder of .npz masks>
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from scipy import ndimage as ndi
from PIL import Image, ImageOps

MODEL_ID = "facebook/sam3"
THRESHOLD = 0.4                 # SAM instance threshold used in every stage of the study (mask threshold 0.5)

# ── stage 1 constants ──────────────────────────────────────────────────────────────
P_BC, P_WC, P_ALT, P_PUP, P_EL = ("lower eyelid inner surface", "white part of the eye",
                                  "the bright area in the eye", "pupil", "eyelashes")
TRIG, GROW_IRIS, FRAG = 0.75, 0.10, 0.05
# ── stage 2 (conjunctiva: union of the two prompts) ─────────────────────────────────────
P_BC_RED, P_BC_PINK = "the reddish area inside the lower eyelid", "the pink area inside the lower eyelid"
# ── stage 3 ────────────────────────────────────────────────────────────────────────
P_REG, P_SKIN = "the eye surface next to the iris", "skin"
TAU, GATE8, GROW_ELL = 0.05, 0.90, 1.06
# ── stage 4 ────────────────────────────────────────────────────────────────────────
GROW1, GROW2, GATE = 1.1, 1.1, 0.30
ALPHA, TOPCUT = 0.35, 0.15
MAIN_OPTS = ("v1", "cut", "dark")


# ═════════════ helpers ═════════════
def bottom_line(patch, W):
    ys, xs = np.nonzero(patch)
    if len(ys) < 5:
        return None
    b = np.full(W, -1.0)
    np.maximum.at(b, xs, ys.astype(float))
    ok = b >= 0
    if ok.sum() < 5:
        return None
    return np.interp(np.arange(W), np.nonzero(ok)[0], b[ok])


def read_raw(path, apply_exif=True):
    im = Image.open(path)
    if apply_exif:
        im = ImageOps.exif_transpose(im)
    return np.array(im.convert("RGB"))


def compose_from_mask(img, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = img[y0:y1 + 1, x0:x1 + 1] * mask[y0:y1 + 1, x0:x1 + 1, None]
    return Image.fromarray(out.astype(np.uint8))


# ═════════════ stage 1 functions ═════════════
def iris_of(pu):
    if pu.sum() < 20:
        return pu
    u = pu.astype(np.uint8)
    ct, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ct:
        return pu
    o = np.zeros_like(u); cv2.drawContours(o, [cv2.convexHull(np.vstack(ct))], -1, 1, -1)
    r = np.sqrt(pu.sum() / np.pi); k = max(3, int(round(r * GROW_IRIS)) * 2 + 1)
    return cv2.dilate(o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def upper_only(el, ir, H):
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


# ═════════════ stage 3 functions ═════════════
def fit_ellipse_all(m, H, W, grow=1.06):
    if m.sum() < 200:
        return np.ones((H, W), bool)
    ct, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ct:
        return np.ones((H, W), bool)
    pts = np.vstack(ct)
    if len(pts) < 5:
        return np.ones((H, W), bool)
    (cx, cy), (aw, ah), ang = cv2.fitEllipse(pts)
    o = np.zeros((H, W), np.uint8)
    cv2.ellipse(o, ((cx, cy), (aw*grow, ah*grow), ang), 1, -1)
    return o.astype(bool)


def fill_gap(m, pu, eyeW, extra=None, frac=0.02):
    if m.sum() == 0:
        return m, np.zeros_like(m)
    if extra is None:
        extra = np.zeros_like(m)
    k = max(3, int(round(eyeW * frac)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cl = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    hole = ((cl | ndi.binary_fill_holes(cl)) | extra) & ~m
    if not hole.any():
        return m, hole
    n, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8))
    keep = np.zeros_like(m)
    for v in range(1, n):
        c = (lab == v)
        if pu.sum() > 20 and (c & pu).any():
            continue
        keep |= c
    return m | keep, keep


def bridge_columns(U, R, H, W, maxfrac=0.30):
    band = np.zeros((H, W), bool)
    if U.sum() < 50 or R.sum() < 50:
        return band
    ys_u, xs_u = np.nonzero(U); ys_r, xs_r = np.nonzero(R)
    tu = np.full(W, 1e9); np.minimum.at(tu, xs_u, ys_u.astype(float))
    br = np.full(W, -1.0);  np.maximum.at(br, xs_r, ys_r.astype(float))
    ys_a = np.nonzero((U | R).any(1))[0]
    eyeH = max(20, ys_a.max() - ys_a.min() + 1)
    ok = (tu < 1e9) & (br >= 0) & (tu > br + 1) & ((tu - br) <= maxfrac * eyeH)
    if not ok.any():
        return band
    yy = np.arange(H)[:, None]
    band = (yy > br[None, :]) & (yy < tu[None, :]) & ok[None, :]
    return band


def keep_touching(m, seed, W):
    if m.sum() == 0 or seed.sum() == 0:
        return m
    k = max(3, int(round(W * 0.006)) * 2 + 1)
    sd = cv2.dilate(seed.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = np.zeros_like(m)
    for v in range(1, n):
        c = (lab == v)
        if (c & sd).any():
            out |= c
    return out


def build_one(U0, w1, ir, el, pu, m2r, mas, H, W):
    """Stage 3 for one photograph (arrays in, arrays out)."""
    yy = np.arange(H)[:, None]
    skipB = False
    if U0.sum() > 100 and ir.sum() > 100:
        skipB = (U0 & (yy < np.nonzero(ir.any(1))[0].min())).sum() > 0.5 * U0.sum()
    Lb = None if (U0.sum() <= 100 or skipB) else bottom_line(U0, W)
    fl = (yy <= Lb[None, :]) if Lb is not None else np.ones((H, W), bool)
    ty = None if ir.sum() < 20 else float(np.nonzero(ir.any(1))[0].min())
    ce = np.ones((H, W), bool) if ty is None else (yy >= ty)
    skin_overlap = float((mas & U0).sum() / max(U0.sum(), 1))
    skin_used = skin_overlap <= TAU

    RAW = ((((m2r & ~ir) & ~el) & ~pu) & fl) & ce
    core = (RAW & ~mas) if skin_used else RAW
    reg2 = drop_small(core)
    reg2 = keep_touching(reg2, ir | U0, W)
    firedA = reg2.sum() < 0.05 * max(RAW.sum(), 1)
    if firedA:
        reg2 = drop_small(RAW)

    reg1 = keep_touching(drop_small((((w1 & ~pu) & fl) & ce)), ir | U0, W)
    if reg1.sum() < 0.05 * max(w1.sum(), 1):
        reg1 = drop_small(w1 & fl)

    sclera_share = float(reg1.sum() / max(U0.sum(), 1))
    pass_used = 2 if sclera_share < GATE8 else 1
    reg = reg2 if pass_used == 2 else reg1

    E = fit_ellipse_all(drop_small(U0 | reg), H, W, GROW_ELL)
    ellipse_used = (U0 & E).sum() >= 0.8 * max(U0.sum(), 1)
    if not ellipse_used:
        E = np.ones((H, W), bool)
    U, R = U0 & E, reg & E
    return dict(BC=U, REG1=reg1, REG2=reg2, IRIS=ir, skin_used=bool(skin_used), safety_A=bool(firedA),
                safety_B=bool(skipB), ellipse_used=bool(ellipse_used))


# ═════════════ stage 4 functions ═════════════
def iris_at(pu, ir0, g):
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
    if m.sum() < 100:
        return m
    q = m & ~ir
    if q.sum() < 0.05 * m.sum():
        return m
    q = drop_small(q)
    out = keep_touching(q, ir | seed, W)
    return out if out.sum() >= 0.05 * m.sum() else q


def build_final(U0, r1, r2, ir0, pu, el, H, W, g1=GROW1, g2=GROW2, gate=GATE, gray=None,
                opts=MAIN_OPTS, noiris=False):
    """Stage 4: final masks — U0 · r1 · r2 · ir0 from stage 3, pu · el (= el_all) from stage 1."""
    U0_full, r1_full, r2_full, ir0_full, pu_full, el_full = U0, r1, r2, ir0, pu, el
    any_ = U0 | r1 | r2 | ir0 | pu
    if not any_.any():
        z = np.zeros((H, W), bool)
        return dict(BC=z, SCCA=z, SCLERA=z, band=z, IRIS=z, pass_no=1, sclera_share=0.0,
                    slices=None, iris_dropped=False, band_share=0.0)
    ys, xs = np.nonzero(any_)
    m_ = int(0.12 * max(ys.max()-ys.min(), xs.max()-xs.min())) + 40
    y0, y1 = max(0, ys.min()-m_), min(H, ys.max()+m_+1)
    x0, x1 = max(0, xs.min()-m_), min(W, xs.max()+m_+1)
    S = (slice(y0, y1), slice(x0, x1))
    h, w = y1-y0, x1-x0
    U0raw = int(U0.sum())
    U0, r1, r2, ir0, pu, el = U0[S], r1[S], r2[S], ir0[S], pu[S], el[S]
    gy = gray[S] if gray is not None else None

    sclera_share = float((r1 & ~U0).sum() / max(U0.sum(), 1))
    pass_no = 2 if sclera_share < gate else 1
    if noiris:
        ir0 = np.zeros_like(ir0); pu = np.zeros_like(pu)
    i1, i2 = iris_at(pu, ir0, g1), iris_at(pu, ir0, g2)
    reg, ir = ((cut_iris(r2, i2, U0, w), i2) if pass_no == 2
               else (cut_iris(r1, i1, U0, w), i1))

    E = fit_ellipse_all(drop_small(U0 | reg), h, w, GROW_ELL)
    if (U0 & E).sum() < 0.8 * max(U0.sum(), 1):
        E = np.ones((h, w), bool)
    U, R = U0 & E, reg & E
    s0 = drop_small(U | R)
    ys_, xs_ = np.nonzero(s0) if s0.any() else (np.array([0]), np.array([0]))
    br = bridge_columns(U, drop_small(R & ~U), h, w) & ~ir
    scca, band = fill_gap(s0, pu, max(20, xs_.max()-xs_.min()+1), extra=br)
    scca = scca & ~ir; U = U & ~ir; band = band & ~ir

    T = np.zeros((h, w), bool)
    yy = np.arange(h)[:, None]
    if ir0.sum() > 20:
        yi = np.nonzero(ir0.any(1))[0]; itop, imid = yi.min(), yi.mean(); ih = max(1, yi.max()-itop)
    else:
        itop, imid, ih = 0, h/2.0, h
    if "v1" in opts and el.sum() > 20:
        T |= el & (yy < imid)
    if "cut" in opts:
        T |= (yy < itop + TOPCUT * ih)
    if "dark" in opts and gy is not None and scca.any():
        body = scca & ~ir
        if pu.sum() > 50 and body.sum() > 50:
            b_pu = float(np.median(gy[pu])); b_sc = float(np.median(gy[body]))
            T |= (gy < b_pu + ALPHA * (b_sc - b_pu)) & (yy < imid)
    if T.any():
        scca = scca & ~T; U = U & ~T; band = band & ~T

    if not noiris and scca.sum() < 0.05 * max(U0raw, 1):         # safety net C (pupil run-away)
        return build_final(U0_full, r1_full, r2_full, ir0_full, pu_full, el_full, H, W,
                           g1, g2, gate, gray, opts, noiris=True)

    return dict(BC=U, SCCA=scca, SCLERA=drop_small(scca & ~U), band=band, IRIS=ir,
                pass_no=pass_no, sclera_share=sclera_share, slices=S, iris_dropped=bool(noiris),
                band_share=float(band.sum()/max(scca.sum(), 1)))


def paste(m, S, H, W):
    o = np.zeros((H, W), bool)
    if S is not None: o[S] = m
    return o


# ═════════════ SAM 3 ═════════════
class SAM:
    def __init__(self, device, threshold=THRESHOLD):
        from transformers import Sam3Model, Sam3Processor
        self.dev = device
        self.model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
        self.proc = Sam3Processor.from_pretrained(MODEL_ID)
        self.th = threshold

    @torch.no_grad()
    def masks(self, pil, text):
        """Instance masks sorted by score (highest first) and their scores."""
        inp = self.proc(images=pil, text=text, return_tensors="pt").to(self.dev)
        R = self.proc.post_process_instance_segmentation(
            self.model(**inp), threshold=self.th, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())[0]
        ms = R["masks"].cpu().numpy() > 0
        if not len(ms):
            return ms, np.zeros(0)
        sc = R["scores"].cpu().numpy(); o = np.argsort(-sc)
        return ms[o], sc[o]


def segment(img, sam):
    """One photograph (RGB array, EXIF already applied) -> final masks and a log row."""
    H, W = img.shape[:2]; pil = Image.fromarray(img); z = np.zeros((H, W), bool)
    top1 = lambda ms: ms[0] if len(ms) else z
    top2 = lambda ms: ms[:2].any(axis=0) if len(ms) else z
    alls = lambda ms: ms.any(axis=0) if len(ms) else z

    # stage 1: first pass
    ms, _ = sam.masks(pil, P_BC); b1 = drop_small(top1(ms))
    ms, sc = sam.masks(pil, P_WC); wb = top2(ms); sw = float(sc[0]) if len(sc) else 0.0
    ms, _ = sam.masks(pil, P_ALT); wa = top2(ms)
    ms, _ = sam.masks(pil, P_PUP); pu = top1(ms)
    ms, _ = sam.masks(pil, P_EL); el_all = alls(ms)
    white_fallback = bool(sw < TRIG)
    w_raw = wa if white_fallback else wb
    ir = iris_of(pu)
    el_up = upper_only(el_all, ir, H)
    Lb = bottom_line(b1, W); yy = np.arange(H)[:, None]
    floor = (yy <= Lb[None, :]) if Lb is not None else np.ones((H, W), bool)
    w_fin = drop_small(((w_raw & ~ir) & ~el_up) & floor)

    # stage 2: conjunctiva = reddish ∪ pink
    ms, _ = sam.masks(pil, P_BC_RED); bc_1 = drop_small(top1(ms))
    ms, _ = sam.masks(pil, P_BC_PINK); bc_2 = drop_small(top1(ms))
    U0 = bc_1 | bc_2

    # stage 3: second-pass region and skin
    ms, _ = sam.masks(pil, P_REG); m2r = top2(ms)
    ms, _ = sam.masks(pil, P_SKIN); mas = alls(ms)
    M3 = build_one(U0, w_fin, ir, el_up, pu, m2r, mas, H, W)

    # stage 4: final masks
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    M = build_final(M3["BC"], M3["REG1"], M3["REG2"], M3["IRIS"], pu, el_all, H, W, gray=gray)
    S = M["slices"]
    out = {k: paste(M[k2], S, H, W) for k, k2 in (("bc", "BC"), ("sclera", "SCLERA"), ("scca", "SCCA"), ("iris", "IRIS"))}
    log = dict(pass_used=M["pass_no"], sclera_share=M["sclera_share"], white_fallback=white_fallback, iris_dropped=M["iris_dropped"],
               skin_used=M3["skin_used"], safety_A=M3["safety_A"], safety_B=M3["safety_B"], ellipse_used=M3["ellipse_used"],
               scca_px=int(out["scca"].sum()), bc_px=int(out["bc"].sum()), sclera_px=int(out["sclera"].sum()))
    return out, log


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="image files or folders (.jpg .jpeg .png)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0", help="CUDA device index (or 'cpu' — slow)")
    ap.add_argument("--no-exif", action="store_true", help="do not apply the EXIF orientation (the study applied it)")
    ap.add_argument("--compare", default="", help="folder of reference masks <name>.npz to compare with (Dice)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = []
    for x in a.images:
        p = Path(x)
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}) if p.is_dir() else [p]
    if a.limit: files = files[:a.limit]
    out = Path(a.out)
    for d in ("roi_scca", "roi_bc", "masks"): (out / d).mkdir(parents=True, exist_ok=True)
    dev = "cpu" if a.device == "cpu" else f"cuda:{a.device}"
    sam = SAM(dev)
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        img = read_raw(f, not a.no_exif); H, W = img.shape[:2]
        M, log = segment(img, sam)
        np.savez_compressed(out / "masks" / f"{f.stem}.npz", shape=np.array([H, W]),
                            **{k: np.packbits(v) for k, v in M.items()})
        for key, d in (("scca", "roi_scca"), ("bc", "roi_bc")):
            c = compose_from_mask(img, M[key])
            if c is not None: c.save(out / d / f"{f.stem}.jpg", quality=95)
        r = dict(name=f.name, **log)
        if a.compare:
            ref = Path(a.compare) / f"{f.stem}.npz"
            if ref.exists():
                z = np.load(ref); h, w = z["shape"]
                for k in ("scca", "bc", "sclera"):
                    m = np.unpackbits(z[k])[:h*w].reshape(h, w).astype(bool)
                    r[f"dice_{k}"] = dice(M[k], m) if m.shape == M[k].shape else np.nan
                    r[f"identical_{k}"] = bool(m.shape == M[k].shape and (m == M[k]).all())
        rows.append(r)
        print(f"  [{i+1}/{len(files)}] {f.name} · pass {log['pass_used']} · {time.time()-t0:.0f}s"
              + (f" · Dice SCCA {r['dice_scca']:.4f}" if "dice_scca" in r else ""), flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "segmentation_log.csv", index=False)
    print(f"done: {len(D)} photographs · second pass {int((D.pass_used == 2).sum())} -> {out}")
    if "dice_scca" in D:
        print(f"compared with reference: identical SCCA masks {int(D.identical_scca.sum())}/{len(D)} · "
              f"Dice SCCA median {D.dice_scca.median():.4f} · min {D.dice_scca.min():.4f}")


if __name__ == "__main__":
    main()
