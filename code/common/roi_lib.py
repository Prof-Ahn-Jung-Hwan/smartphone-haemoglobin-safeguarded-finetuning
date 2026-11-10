"""
원본 사진 + 폴리곤 주석 -> 모델 입력 합성 이미지, 공용 모듈.

★ 2026-08-20 에 알아낸 합성 규칙 (실측 검증됨)

    합성 이미지 = 원본 사진 x 폴리곤 마스크  ->  폴리곤 bbox 로 잘라내기(양끝 포함)
        SCCA 입력 (segmentation_wc_bc_new) = WCBC 클래스 폴리곤
        BC   입력 (segmentation_bc_new)    = BC   클래스 폴리곤

    검증: <stem> 재생성 결과 기존 파일과
          비검은 영역 평균 차이 0.6/255 (JPEG 재압축 수준), 검은픽셀 일치율 0.988

이 모듈이 있으면 폴리곤만 갈아끼워 두 가지를 할 수 있다.
    1) ROI 강건성  — 폴리곤을 일부러 침식/팽창/이동시킨다
    2) 자동 분할   — 폴리곤을 SAM 예측 마스크로 바꾼다

위치: autosegmentation/common/roi_lib.py  (autosegmentation/ 하위 실험이 모두 공유)

참조: strategy/260819_autosegmentation_guide1.md (자료 함정 6가지)
      strategy/260820_autosegmentation_guide1.md (실험 순서)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("SCCA_DATA_ROOT", "."))   # GitHub 판 : 원래는 연구 서버의 절대경로였다 (환경변수로 지정)
RAW_ROOT = DATA_ROOT / "raw_data" / "internal_data_raw"
DATASET_ROOT = DATA_ROOT / "2026"                    # 학습 config 의 dataset_root
REF_DIR = {"scca": DATASET_ROOT / "segmentation_wc_bc_new",
           "bc": DATASET_ROOT / "segmentation_bc_new"}
ANNO_XLSX = DATASET_ROOT / "data_used_raw_251218_github_1_filtered_2.xlsx"

# 입력 종류 -> 어느 클래스 폴리곤을 쓰는가
INPUT_CLASS = {"scca": "WCBC", "bc": "BC"}

# 파일명 규칙 두 가지가 섞여 있다 (§3-7)
#   초기 배치 : {날짜}_{번호}_{separator}_{촬영인덱스}{ID}_{이름}.jpg   촬영인덱스 1자리가 앞에 붙음
#   후기 배치 : {날짜}_{번호}_{촬영인덱스}_{ID}_{이름}.jpg              그냥 붙어 있음
# ★ 함정 3-8: 2025년 배치 273장은 확장자가 .png 다. 합성 이미지는 전부 .jpg.
# ★ 밑줄이 두 개 겹친 파일이 1장 있다 (<stem>).
_FN = re.compile(r"^(\d{8})_(\d+)_(\d+)_+(\d+)_(.*)\.(jpg|jpeg|png)$", re.IGNORECASE)


def composite_keys(raw_name: str) -> set[str]:
    """원본 파일명에서 나올 수 있는 합성 이미지 파일명 후보를 만든다.

    합성 이미지는 확장자가 항상 .jpg 이므로 원본이 .png 여도 .jpg 로 만든다.
    """
    m = _FN.match(raw_name)
    if not m:
        return set()
    d, n, f3, f4, _, _ext = m.groups()
    keys = {f"{d}_{n}_{f3}_{f4}.jpg"}
    if len(f4) == 8:                       # 촬영인덱스가 ID 앞에 붙은 초기 형식
        keys.add(f"{d}_{n}_{f4[0]}_{f4[1:]}.jpg")
    return keys


# ---------------------------------------------------------------------------
# 주석 읽기
# ---------------------------------------------------------------------------
def load_batch(batch_dir: Path) -> tuple[dict, dict]:
    """한 배치의 (주석, id->클래스이름) 을 돌려준다.

    ★ 함정 3-1: 배치 893-916 만 classId 가 다르다 (3=BC 가 아니라 6=BC).
      그래서 배치마다 classes.json 을 읽어 **이름으로** 매칭해야 한다.
      annotations.json 의 className 은 null 이라 쓸 수 없다.
    """
    cls = {c["id"]: c["name"] for c in json.load(open(batch_dir / "classes.json"))}
    ann = json.load(open(batch_dir / "annotations.json"))
    return ann, cls


def extract_polygons(record: dict, cls: dict, log: list | None = None) -> dict:
    """한 사진의 주석에서 {클래스이름: [(x,y), ...]} 를 뽑는다.

    ★ 함정 3-2: classId == -1 (7개) 과 꼭짓점 3개 미만 (7개) 이 섞여 있다.
      조용히 건너뛰되 무엇을 왜 뺐는지 기록한다.
    """
    out = {}
    for inst in record.get("instances", []):
        cid = inst.get("classId")
        name = cls.get(cid)
        if name is None:
            if log is not None:
                log.append(("unknown_classId", cid))
            continue
        pts = inst.get("points") or []
        if len(pts) < 6:                    # 꼭짓점 3개 = 좌표 6개
            if log is not None:
                log.append(("degenerate_polygon", name))
            continue
        out[name] = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
    return out


# ---------------------------------------------------------------------------
# 합성 이미지 만들기
# ---------------------------------------------------------------------------
def polygon_mask(points, size) -> np.ndarray:
    """폴리곤을 불리언 마스크로 래스터화한다."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=255)
    return np.array(m) > 0


def compose_from_mask(img: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    """마스크 밖을 검게 만들고 마스크의 bbox 로 잘라낸다. 학습 데이터와 같은 방식."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = img[y0:y1 + 1, x0:x1 + 1] * mask[y0:y1 + 1, x0:x1 + 1, None]
    return Image.fromarray(out.astype(np.uint8))


def read_raw(path, apply_exif: bool) -> np.ndarray:
    """원본 사진을 읽는다.

    ★ 함정 3-4: EXIF Orientation 이 1 이 아닌 사진이 126장 있다 (3=8장, 6=118장).
      폴리곤 좌표가 회전 전 기준인지 후 기준인지는 **기존 합성 이미지와 대조해서**
      실증적으로 정한다 (build_pairs.py 의 --resolve-exif). 눈대중으로 정하지 않는다.
    """
    im = Image.open(path)
    if apply_exif:
        im = ImageOps.exif_transpose(im)
    return np.array(im.convert("RGB"))


def build_composite(raw_path, polygons, kind: str, apply_exif: bool,
                    perturb=None) -> Image.Image | None:
    """원본 + 폴리곤 -> 모델 입력 이미지 한 장.

    perturb: mask(bool 배열) 를 받아 바꾼 mask 를 돌려주는 함수. None 이면 원본 ROI.
    """
    cname = INPUT_CLASS[kind]
    if cname not in polygons:
        return None
    img = read_raw(raw_path, apply_exif)
    mask = polygon_mask(polygons[cname], (img.shape[1], img.shape[0]))
    if perturb is not None:
        mask = perturb(mask)
    return compose_from_mask(img, mask)


# ---------------------------------------------------------------------------
# ROI 교란 — ROI 강건성 실험용
# ---------------------------------------------------------------------------
def _kernel(px: int):
    import cv2
    k = max(1, int(px) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def make_perturb(kind: str, frac: float, rng=None):
    """ROI 를 흔드는 함수를 만든다.

    kind  : 'erode' | 'dilate' | 'shift' | 'none'
    frac  : ROI 높이 대비 비율 (0.01 = 1%). 사진마다 ROI 크기가 다르므로
            절대 픽셀이 아니라 **상대 비율**로 흔든다. 그래야 사진 간 비교가 된다.
    """
    import cv2

    if kind == "none" or frac == 0:
        return lambda m: m

    def f(mask):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return mask
        px = max(1, int(round((ys.max() - ys.min() + 1) * frac)))
        u8 = mask.astype(np.uint8)
        if kind == "erode":
            return cv2.erode(u8, _kernel(px)) > 0
        if kind == "dilate":
            return cv2.dilate(u8, _kernel(px)) > 0
        if kind == "shift":
            ang = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
            dx, dy = int(round(px * np.cos(ang))), int(round(px * np.sin(ang)))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(u8, M, (mask.shape[1], mask.shape[0])) > 0
        raise ValueError(kind)

    return f
