# -*- coding: utf-8 -*-
"""**10-20 — 부록 표 번호를 한 칸 밀었다** (Supplementary Table 1 = 선행 연구 요약이 새로 들어감). 이 설명 안의 표 번호 · 파일 이름도 새 번호로 바꿨다 — 옛 판 스크립트는 `_이전판/` 에 옛 번호 그대로 있다.

**Table ST2. 모바일 백본 10종 — 학습률과 내부 검증 성적 (환자 단위)**  2026-10-05 · 10-19 표기

10-19  검증 MAE 에 환자 bootstrap 100,000회 95 % CI · 차이 | 95 % CI | p 따로 · 실험에 쓴 백본은 "O" · 가운데 점 (교수님 10-19)
       MAE 의 정의는 그대로 (검증 칸 6개 = 3겹 × 입력 2 의 환자 단위 MAE 의 평균). CI 도 같은 정의 — 참여자를 재표집해
       칸마다 MAE 를 다시 구하고 6칸을 평균한다. 이전 판은 previous_strategy_notNow/_표형식_전_백업/

왜  안전장치 실험에 쓴 다섯 백본이 어디서 왔는지 보인다. 10종 모두 가장 단순한 머리(회귀만 · 안전장치 없음)로
    학습률 격자를 돌렸고, 그 학습률에서의 **환자 단위 검증 MAE**(두 눈 예측의 평균)와 1위 대비 차이 · p 를 싣는다.

1차 자료  main/results/grid/val_pred/*.csv  (`paperwork/results_script/analysis/261005_lr_환자단위검증.py` 가 체크포인트 순전파로 만든다)
          main/results/grid/lr_선택.csv · lr_선택_환자단위.csv
          학습률은 실험에 실제로 쓴 값(`lr_선택.csv` 경로 A)이다. 환자 단위로 다시 골라도 9종은 같고,
          MobileViT-XXS 만 1e-3 이 되는데 그 차이는 0.001 g/dL 이다 (각주)
내는 것   results_table_figure/supplementary/TableST2_backbone_screen.xlsx
쓰는 법   python supp_tableST2_backbone_screen.py
"""
import glob
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

G = 뿌리 / "main/results/grid"
이름10 = {"mnl": ("MobileNetV3-Large", "CNN"), "mns": ("MobileNetV3-Small", "CNN"), "mnv2": ("MobileNetV2", "CNN"),
         "eb0": ("EfficientNet-B0", "CNN"), "shf": ("ShuffleNetV2", "CNN"), "mvxx": ("MobileViT-XXS", "CNN–transformer hybrid"),
         "mvx": ("MobileViT-XS", "CNN–transformer hybrid"), "ef2s0": ("EfficientFormerV2-S0", "CNN–transformer hybrid"),
         "enxs": ("EdgeNeXt-XS", "CNN–transformer hybrid"), "fvt8": ("FastViT-T8", "CNN–transformer hybrid")}
# ★ 10-27b (사용자) : 이 표의 10종은 **모두 안전장치 없이** 비교했다 — 열 이름이 "안전장치를 써서 좋았다" 로 읽혀
#   "Carried forward to the safeguard experiments" 로 바꾸고 값은 "Carried forward" / "Not carried forward" 만 (실험 내용은 각주)
실험 = {"mnl", "eb0", "enxs", "mnv2", "mvxx"}
명 = pd.read_csv(자료 / "internal/명부.csv", usecols=["stem", "number"]); 번호 = dict(zip(명.stem.astype(str), 명.number))

시작("Table ST2 — 백본 10종")
옛 = pd.read_csv(G / "lr_선택.csv"); 옛.columns = [c.lstrip("﻿") for c in 옛.columns]; 옛 = 옛[옛["경로"] == "A"].set_index("백본")
새 = pd.read_csv(G / "lr_선택_환자단위.csv").set_index("백본")


def 환자오차(bb, lr):
    s = f"{lr:.0e}".replace("e-0", "e-")
    fs = sorted(glob.glob(str(G / "val_pred" / f"g2_{bb}_A_lr{s}_auto_*-fold*.csv")))
    if len(fs) != 6: sys.exit(f"🔴 {bb} lr {s} : 검증 예측 {len(fs)}칸 (6칸이어야 한다)")
    급 = []
    for f in fs:
        p = pd.read_csv(f); p["n"] = p.key.astype(str).map(번호)
        g = p.groupby("n").agg(pred=("pred", "mean"), y=("y", "first")); 급.append((g.pred - g.y).abs())
    W = pd.concat(급, axis=1)                         # 참여자 × 칸 (그 칸에 없으면 빈칸)
    return W, float(np.mean([e.mean() for e in 급]))


def 칸평균CI(W, B=100000):
    """참여자 재표집 → 칸마다 MAE → 6칸 평균 · 2·5 · 97·5 percentile."""
    A = W.to_numpy(float); n = len(A); rng = np.random.RandomState(0); m = np.empty(B)
    for b in range(B): m[b] = np.nanmean(A[rng.randint(0, n, n)], axis=0).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


E = {bb: 환자오차(bb, float(옛.loc[bb, "lr"])) for bb in 이름10}
순 = sorted(E, key=lambda b: E[b][1]); 일 = 순[0]
행 = []
for i, bb in enumerate(순, 1):
    e, mae = E[bb]
    if bb == 일: pp = "··"
    else:
        pp = p표(대응(E[일][0].mean(axis=1), e.mean(axis=1))[3])
    lo, hi = 칸평균CI(e); m = mae
    행.append({"Rank": i, "Backbone": 이름10[bb][0] + (" (reference)" if bb == 일 else ""), "Family": 이름10[bb][1],
              "Learning rate": f"{float(옛.loc[bb, 'lr']):.0e}".replace("-0", "−").replace("-", "−"),
              "Validation MAE, g/dL (95% CI)": 값CI(m, lo, hi),
              "p value vs reference": pp})   # ★ 10-27c (사용자) : 선택 열은 빼고 각주에 다섯 이름만
T = pd.DataFrame(행); print(T.to_string(index=False))
mv = 새.loc["mvxx"]
엑셀("TableST2_backbone_screen", 부록, {"ST2": T},
    제목="Table ST2. Ten mobile backbones trained without safeguards: learning rate and internal validation error",
    주석=[
      "Each backbone was trained with a regression head and no safeguards. The learning rate was chosen from a grid",
      "  (1e-5 to 3e-3, extended when the best value lay at the edge) by the lowest validation error in the development",
      "  cohort, averaged over three participant-grouped folds and two input regions (sclera and conjunctiva; conjunctiva).",
      "Validation MAE is at the participant level: the two eye predictions of each participant are averaged before the",
      "  absolute error is taken; the MAE is the mean over the six validation cells (three folds x two input regions).",
      "  95% CIs from 100,000 bootstrap resamples of participants (the six-cell mean is recomputed in each resample);",
      "p value vs reference: each backbone compared with the lowest-error backbone (rank 1) in the same participants (each",
      "  participant's error averaged over their cells), two-sided bootstrap.",
      "CNN, convolutional neural network; CNN–transformer hybrid, convolutional stages combined with self-attention (transformer)",
      "  blocks.",
      "All ten backbones in this table were trained without any safeguard.",
      f"Selecting the learning rate by participant-level error gave the same value for nine backbones; for MobileViT-XXS",
      f"  it gave {mv['고른lr_환자단위']:.0e}, whose error differed from that of the rate used by {점(abs(mv['환자단위MAE'] - E['mvxx'][1]))} g/dL.",
      "Five backbones were selected for the safeguard experiments (table 3 and the appendix): EdgeNeXt-XS, MobileNetV2,",
      "  EfficientNet-B0, MobileViT-XXS, and MobileNetV3-Large — spanning both families and including one without batch",
      "  normalisation (EdgeNeXt-XS); MobileNetV3-Large was included as the mobile architecture family most used for anaemia",
      "  detection.",
    ])
