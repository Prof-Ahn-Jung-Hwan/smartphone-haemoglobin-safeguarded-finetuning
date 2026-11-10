# -*- coding: utf-8 -*-
"""자동분할 채점 지표 — 사람 주석이 **조각**이어도 성립하는 것만.  2026-08-31

왜 새로 만드나 (08-31 확인)
    · 사람 주석의 성격이 서로 다르다 (크롭 대비 넓이)
        WCBC 6 % · BC 4 % · WC 1 %   -> **조각**. 특징적인 곳만 오려냈다
        CO 6 % · SKIN 41 %           -> **전체**. 홍채와 주변 피부를 다 그렸다
    · 그런데 `피부오염 = |SAM ∩ SKIN| / |SAM|` 은 **거의 0 으로만 나왔다** (360장 중 6장만 >10 %).
      SAM 이 삐져나가는 곳은 주석된 SKIN 이 아니라 그 **위쪽의 눈꺼풀 가장자리·속눈썹 띠** 이고,
      거기는 아무도 주석하지 않았다. 그래서 목표를 못 잰다.

    -> **사람 조각의 아래끝을 기준선으로 삼는다.** 조각이어도 그 아래끝은
       결막 위(눈 안쪽)에 있으므로, 그보다 **한참 아래**로 나간 것은 눈이 아니다.

지표
    아래삐짐    SAM 마스크 중 '사람 조각 아래끝' 보다 아래인 비율        ★ 1번 목표
    삐진깊이    (SAM 아래끝 − 사람 조각 아래끝) / 눈 높이               ★ 얼마나 멀리 나갔나
    피부오염    |SAM ∩ 사람SKIN| / |SAM|                              (SKIN 은 전체 주석)
    홍채오염    |SAM ∩ 사람CO| / |SAM|                                (CO 는 전체 주석)
    패치포함    |SAM ∩ 사람조각| / |사람조각|                          1.000 이 목표
    크기비      |SAM| / |사람조각|                                     조각이라 1보다 크다
    (dice)      방향 확인용 보조. 순위 기준으로 쓰지 않는다
"""
from __future__ import annotations
import numpy as np


def bottom_line(patch, W):
    """사람 조각의 **열별 아래끝**. 조각이 없는 열은 좌우를 이어 메운다."""
    ys, xs = np.nonzero(patch)
    if len(ys) < 5:
        return None
    b = np.full(W, -1.0)
    np.maximum.at(b, xs, ys.astype(float))      # 🔴 NaN 으로 초기화하면 전부 NaN 이 된다
    ok = b >= 0
    if ok.sum() < 5:
        return None
    return np.interp(np.arange(W), np.nonzero(ok)[0], b[ok])


def below_frac(mask, line, H):
    """마스크 중 line 보다 아래인 비율."""
    if line is None or mask.sum() == 0:
        return np.nan
    yy = np.arange(H)[:, None]
    return float((mask & (yy > line[None, :])).sum() / mask.sum())


def overhang(mask, line, eye_h):
    """마스크 아래끝이 line 보다 얼마나 아래인가 (눈 높이 대비). 음수면 안 넘었다."""
    if line is None or mask.sum() == 0 or not eye_h:
        return np.nan
    ys, xs = np.nonzero(mask)
    return float(np.percentile(ys - line[xs], 95) / eye_h)


def score_mask(ms, mb, G, eye_h):
    """G = {"WCBC","BC","CO","SKIN"} 사람 마스크 (없으면 None).  ms=SCCA · mb=BC 예측"""
    H, W = ms.shape
    gw, gb = G["WCBC"], G["BC"]
    co = G.get("CO"); sk = G.get("SKIN")
    lw = bottom_line(gw, W); lb = bottom_line(gb, W)
    A = max(1, ms.sum()); Ab = max(1, mb.sum())
    return {
        "아래삐짐": below_frac(ms, lw, H),
        "삐진깊이": overhang(ms, lw, eye_h),
        "피부오염": float((ms & sk).sum() / A) if sk is not None else np.nan,
        "홍채오염": float((ms & co).sum() / A) if co is not None else np.nan,
        "패치포함": float((ms & gw).sum() / max(1, gw.sum())),
        "크기비": float(ms.sum() / max(1, gw.sum())),
        "BC아래삐짐": below_frac(mb, lb, H),
        "BC삐진깊이": overhang(mb, lb, eye_h),
        "BC패치포함": float((mb & gb).sum() / max(1, gb.sum())),
    }


COLS = ["아래삐짐", "삐진깊이", "피부오염", "홍채오염", "패치포함", "크기비",
        "BC아래삐짐", "BC삐진깊이", "BC패치포함"]
