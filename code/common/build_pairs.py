"""
원본 사진 <-> 폴리곤 주석 <-> 학습용 합성 이미지 를 하나로 잇는 색인을 만든다.

왜 필요한가
    지금 파이프라인은 '이미 만들어진 합성 이미지'에서 시작한다. 그래서
    ROI 를 바꿔보는 실험도, 자동 분할로 갈아끼우는 것도 불가능했다.
    이 색인이 있으면 **폴리곤만 바꿔 끼워** 두 가지를 다 할 수 있다.

무엇을 하는가
    1) 35개 배치를 돌며 배치별 classes.json 으로 id->이름 표를 만든다  (함정 3-1)
    2) 원본 파일명을 합성 이미지 파일명으로 대응시킨다                  (함정 3-7, 두 가지 규칙)
    3) EXIF 회전 규약을 **기존 합성 이미지와 대조해** 실증적으로 정한다  (함정 3-4)
    4) 잘못된 폴리곤을 빼고 무엇을 왜 뺐는지 기록한다                    (함정 3-2)
    5) 재생성 검증 — 기존 합성 이미지를 얼마나 정확히 재현하는지 잰다
    6) 익명 ID 를 붙이고 실명 매핑표는 따로 저장한다                     (함정 3-5)

사용법
    conda activate torch271
    python autosegmentation/common/build_pairs.py --validate 60

산출물
    autosegmentation/common/results/pairs_index.json  색인 (실명 없음)
    autosegmentation/common/results/name_map.csv      ⚠️ 익명ID <-> 실명. 절대 공유 금지
    autosegmentation/common/results/build_pairs.txt   제외 사유와 검증 결과

참조: strategy/260819_autosegmentation_guide1.md §3 (자료 함정)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_lib import (ANNO_XLSX, INPUT_CLASS, RAW_ROOT, REF_DIR, _FN, build_composite,
                     composite_keys, extract_polygons, load_batch)

OUT = Path(__file__).resolve().parent / "results"
NEEDED = ("WC", "BC", "WCBC")          # 이 셋이 다 있어야 SCCA·BC 둘 다 만들 수 있다


def anon_id(raw_name: str) -> str:
    return "P" + hashlib.sha1(raw_name.encode()).hexdigest()[:10]


def build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "build_pairs.txt", "w", encoding="utf-8")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    used = set(Path(p).name for p in pd.read_excel(ANNO_XLSX)["file_path"])
    ref_scca = set(p.name for p in REF_DIR["scca"].glob("*.jpg"))
    say(f"Hb 엑셀이 쓰는 합성 이미지 {len(used)}장 / 합성 폴더에 있는 것 {len(ref_scca)}장")

    batches = sorted(p for p in RAW_ROOT.iterdir() if (p / "annotations.json").exists())
    say(f"배치 {len(batches)}개\n")

    index, drop, polylog = {}, Counter(), []
    dup = defaultdict(list)

    for b in batches:
        ann, cls = load_batch(b)
        for raw_name, rec in ann.items():
            if not isinstance(rec, dict) or not _FN.match(raw_name):
                if raw_name != "___sa_version___":
                    drop["파일명_파싱실패"] += 1
                continue
            keys = composite_keys(raw_name) & ref_scca
            if not keys:
                drop["합성이미지_없음"] += 1
                continue
            if len(keys) > 1:
                drop["합성이름_모호"] += 1
                continue
            key = keys.pop()
            raw_path = b / "images" / raw_name
            if not raw_path.exists():
                drop["원본파일_없음"] += 1
                continue
            polys = extract_polygons(rec, cls, polylog)
            missing = [c for c in NEEDED if c not in polys]
            if missing:
                drop[f"클래스누락_{'+'.join(missing)}"] += 1
                continue
            dup[key].append(raw_name)
            index[key] = dict(raw=str(raw_path), batch=b.name, anon=anon_id(raw_name),
                              polygons={c: polys[c] for c in NEEDED},
                              used_for_hb=key in used)

    for k, v in dup.items():
        if len(v) > 1:
            drop["동일합성_원본중복"] += 1

    say("== 제외 사유 ==")
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        say(f"  {k:28s} {v}")
    say("\n== 폴리곤 수준 제외 ==")
    for k, v in Counter(x[0] for x in polylog).items():
        say(f"  {k:28s} {v}")

    n_hb = sum(1 for v in index.values() if v["used_for_hb"])
    say(f"\n색인 {len(index)}장 (그중 Hb 학습에 쓰인 것 {n_hb} / {len(used)})")
    miss = used - set(index)
    if miss:
        say(f"⚠️ Hb 에는 쓰이는데 색인에 못 넣은 것 {len(miss)}장: {sorted(miss)[:5]}")

    # ---- EXIF 회전 규약을 실증적으로 정한다 -------------------------------
    say("\n== EXIF 회전 규약 판정 (기존 합성 이미지와 대조) ==")
    orient = {}
    for key, v in index.items():
        try:
            o = Image.open(v["raw"]).getexif().get(274, 1)
        except Exception:
            o = 1
        orient[key] = o or 1
    rot = [k for k, o in orient.items() if o != 1]
    say(f"  Orientation != 1 인 사진 {len(rot)}장")

    def content_mad(key, apply_exif):
        """★ 반드시 '보이는 내용'으로 판정해야 한다.

        마스크는 폴리곤에서 나오므로 회전을 적용하든 안 하든 **검은픽셀 패턴은 똑같다.**
        (2026-08-20 에 이걸로 한 번 틀렸다.) 회전이 어긋나면 잘라낸 위치가 달라져
        내용만 달라지므로, 내용 평균차로 봐야 구별된다.
        """
        v = index[key]
        c = build_composite(v["raw"], v["polygons"], "scca", apply_exif)
        ref = REF_DIR["scca"] / key
        if c is None or not ref.exists():
            return None
        r = Image.open(ref).convert("RGB")
        if c.size != r.size:
            return 255.0
        a, b_ = np.array(c).astype(int), np.array(r).astype(int)
        blk = b_.sum(2) == 0
        if blk.all():
            return None
        return float(np.abs(a - b_)[~blk].mean())

    def summarise(sample):
        on = [content_mad(k, True) for k in sample]
        off = [content_mad(k, False) for k in sample]
        on = [x for x in on if x is not None]
        off = [x for x in off if x is not None]
        return on, off

    norm = [k for k, o in orient.items() if o == 1][:15]
    on_r, off_r = summarise(rot[:15])
    on_n, off_n = summarise(norm)
    if on_r:
        say(f"  회전 있음 n={len(on_r)}  exif 적용 평균차 {np.mean(on_r):6.2f}  "
            f"미적용 {np.mean(off_r):6.2f}")
    if on_n:
        say(f"  회전 없음 n={len(on_n)}  exif 적용 평균차 {np.mean(on_n):6.2f}  "
            f"미적용 {np.mean(off_n):6.2f}")

    if on_r:
        apply_exif = bool(np.mean(on_r) < np.mean(off_r))
        say(f"  -> 판정: 폴리곤 좌표는 회전 {'적용 후' if apply_exif else '적용 전'} 기준. "
            f"apply_exif = {apply_exif}")
        say(f"     회전 사진 {len(rot)}장이 여기 걸린다. 틀리면 그 8%가 통째로 쓰레기가 된다.")
    else:
        apply_exif = True
        say("  -> 회전 표본이 없어 판정 못 함. apply_exif=True 로 둔다")

    # ---- 재생성 검증 ------------------------------------------------------
    say(f"\n== 재생성 검증 (무작위 {args.validate}장) ==")
    rng = np.random.default_rng(0)
    keys = sorted(k for k in index if index[k]["used_for_hb"])
    pick = rng.choice(keys, size=min(args.validate, len(keys)), replace=False)
    rows = []
    for k in pick:
        v = index[k]
        for kind in ("scca", "bc"):
            c = build_composite(v["raw"], v["polygons"], kind, apply_exif)
            ref = REF_DIR[kind] / k
            if c is None or not ref.exists():
                rows.append(dict(key=k, kind=kind, ok=False)); continue
            r = Image.open(ref).convert("RGB")
            if c.size != r.size:
                rows.append(dict(key=k, kind=kind, ok=False, note=f"{c.size}!={r.size}"))
                continue
            a, b_ = np.array(c).astype(int), np.array(r).astype(int)
            blk = b_.sum(2) == 0
            rows.append(dict(key=k, kind=kind, ok=True,
                             black_agree=((a.sum(2) == 0) == blk).mean(),
                             mad_visible=np.abs(a - b_)[~blk].mean()))
    df = pd.DataFrame(rows)
    for kind, g in df.groupby("kind"):
        good = g[g.ok == True]
        say(f"  {kind:5s} 성공 {len(good)}/{len(g)}  "
            f"검은픽셀 일치 {good.black_agree.mean():.4f}  "
            f"보이는 영역 평균차 {good.mad_visible.mean():.2f}/255")
    say("\n  해석: 평균차 3 이하면 JPEG 재압축 수준이고 합성 규칙이 맞다는 뜻이다.")
    df.to_csv(OUT / "regen_check.csv", index=False)

    # ---- 저장 -------------------------------------------------------------
    with open(OUT / "name_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["anon_id", "raw_filename"])
        for v in index.values():
            w.writerow([v["anon"], Path(v["raw"]).name])
    (OUT / "name_map.csv").chmod(0o600)

    slim = {k: dict(raw=v["raw"], batch=v["batch"], anon=v["anon"],
                    polygons=v["polygons"], used_for_hb=v["used_for_hb"])
            for k, v in index.items()}
    json.dump(dict(apply_exif=bool(apply_exif), n=len(slim), items=slim),
              open(OUT / "pairs_index.json", "w"))
    say(f"\n저장: {OUT/'pairs_index.json'}  ({len(slim)}장)")
    say(f"⚠️ {OUT/'name_map.csv'} 에 실명이 있다. 권한 600. 절대 공유 금지.")
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=40)
    build(ap.parse_args())
