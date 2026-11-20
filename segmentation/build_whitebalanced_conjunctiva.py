# -*- coding: utf-8 -*-
"""White-balanced conjunctiva input (the comparison input "conjunctiva white-balanced by mean scleral colour").

Each photograph is multiplied channel-wise by g_c = mean(sclera R, G, B means) / sclera channel-c mean
(von Kries correction, using the automatically segmented sclera), and the conjunctiva is then cut out exactly
as for the conjunctiva-only input (outside the mask black, cropped to the mask's box). Photographs with fewer
than MIN_SCLERA sclera pixels are left uncorrected and flagged in the log.

Input   the photographs and the masks written by segmentation/segment_photo.py (masks/<name>.npz: bc · sclera · shape)
Usage   python segmentation/build_whitebalanced_conjunctiva.py --images <photos or folder> --masks seg_out/masks --out wb_out
Output  wb_out/<name>.jpg (JPEG quality 95) · wb_out/whitebalance_log.csv (sclera pixels and channel gains)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from common.roi_lib import compose_from_mask, read_raw   # noqa: E402

MIN_SCLERA = 2000          # with fewer sclera pixels the mean colour is unstable: no correction


def whitebalanced_conjunctiva(raw, mask_npz, apply_exif=True):
    """Photograph + segment_photo mask -> (white-balanced conjunctiva image or None, log row)."""
    z = np.load(mask_npz); H, W = z["shape"]
    get = lambda k: np.unpackbits(z[k])[:H * W].reshape(H, W).astype(bool)
    bc, sc = get("bc"), get("sclera")
    img = read_raw(raw, apply_exif).astype(np.float32)
    n = int(sc.sum())
    if n >= MIN_SCLERA:
        mu = img[sc].mean(axis=0)                       # sclera R, G, B means
        g = mu.mean() / np.maximum(mu, 1.0)             # channel gain = grey / channel -> the sclera becomes neutral
        img = np.clip(img * g[None, None, :], 0, 255)
        status = "corrected"
    else:
        g = np.ones(3, np.float32); status = "too_little_sclera_not_corrected"
    c = compose_from_mask(img.astype(np.uint8), bc)
    if c is None:
        status = "no_conjunctiva"
    return c, dict(status=status, sclera_pixels=n, gain_R=float(g[0]), gain_G=float(g[1]), gain_B=float(g[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="photographs or folders (.jpg .jpeg .png)")
    ap.add_argument("--masks", required=True, help="masks folder written by segment_photo.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-exif", action="store_true", help="do not apply the EXIF orientation (the study applied it)")
    a = ap.parse_args()
    files = []
    for p in map(Path, a.images):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")) if p.is_dir() else [p]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for f in files:
        m = Path(a.masks) / f"{f.stem}.npz"
        if not m.exists():
            rows.append(dict(name=f.stem, status="no_mask")); continue
        img, row = whitebalanced_conjunctiva(f, m, not a.no_exif)
        if img is not None:
            img.save(out / f"{f.stem}.jpg", quality=95)
        rows.append(dict(name=f.stem, **row))
    T = pd.DataFrame(rows); T.to_csv(out / "whitebalance_log.csv", index=False)
    print(T.status.value_counts().to_string())


if __name__ == "__main__":
    main()
