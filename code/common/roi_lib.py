"""Region of interest (ROI) composition: photograph + region mask -> model input image.

The model input is the photograph with every pixel outside the region set to black, cropped to the bounding box
of the region (both ends included). The same rule is used for automatically segmented regions
(segmentation/segment_photo.py) and for manually delineated regions (polygons drawn on SuperAnnotate).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def polygon_mask(points, size) -> np.ndarray:
    """Rasterise a polygon [(x, y), …] into a boolean mask of size (width, height)."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=255)
    return np.array(m) > 0


def compose_from_mask(img: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    """Set pixels outside the mask to black and crop to the mask's bounding box (as for the training data)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = img[y0:y1 + 1, x0:x1 + 1] * mask[y0:y1 + 1, x0:x1 + 1, None]
    return Image.fromarray(out.astype(np.uint8))


def read_raw(path, apply_exif: bool) -> np.ndarray:
    """Read a photograph as an RGB array, optionally applying the EXIF orientation."""
    im = Image.open(path)
    if apply_exif:
        im = ImageOps.exif_transpose(im)
    return np.array(im.convert("RGB"))


def build_composite(raw_path, polygon, apply_exif: bool) -> Image.Image | None:
    """Photograph + one manually drawn polygon -> model input image (manual SCCA or manual conjunctiva)."""
    img = read_raw(raw_path, apply_exif)
    return compose_from_mask(img, polygon_mask(polygon, (img.shape[1], img.shape[0])))
