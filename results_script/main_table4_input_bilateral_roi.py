# -*- coding: utf-8 -*-
"""**Table 4 (11-09 번호 바꿈 · 옛 Table 3). 입력 영역 · 양안 평균 · ROI 출처** — MobileNetV3-L · S1 + S2 (11-04 : 1 seed 10겹 앙상블 · 중간값 (범위) · 옛 판 _old_261030/)

칸   열 = 입력 다섯 · 행 = 수치. 개발 (교차검증) · 기관마다 미세조정 전 · SS 미세조정 후.
     비열등 판정은 개발과 미세조정 후에만 (§14). 개발의 seed 앙상블 줄은 뺐다 (개발은 앙상블로 잴 수 없다 · 지침 맨 위 칸).
1차 자료  main/results/최소구성/train · 최소구성/external/{zero,few_bn} (E04 · E07 · E08 · E14)
내는 것   results_table_figure/main/Table4_input_bilateral_roi.xlsx
쓰는 법   python main_table4_input_bilateral_roi.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

입력들 = list(입력열)
시작("Table 4 — 입력 · 양안 · ROI")
_, 권a, 권폴 = 권유      # ★ 11-07 : 미세조정 뒤 블록은 권유 구성 · 학습률 (`_공통`) · 개발 · 미세조정 전 블록은 그 구성의 개발 모델
if (권a, 권폴) != ("veminimal", FTB): print(f"⚠️ 권유 구성이 S1 + S2 + S3 가 아니다 ({구성글(권a, 권폴)}) — 각주 · 제목 문장을 고칠 것", flush=True)
열 = {g: {} for g in 입력들}
개 = "Development cohort: "
E = {g: 내부교차("mnl", 권a, g) for g in 입력들}
입력블록(열, 개, E)
for g in 입력들:
    v = 내부값("mnl", 권a, g)
    if v is None:
        for t in (12, 11): 열[g][f"{개}AUROC, Hb <{t} g/dL, median (seed range) [95% CI]"] = "··"
        continue
    y = v[v.seed == 1].set_index("key").y.sort_index(); S = -행렬(v, "p").to_numpy(float).T
    for t in (12, 11):
        a, _ = AUROC요약(y.to_numpy() < t, S); 열[g][f"{개}AUROC, Hb <{t} g/dL, median (seed range) [95% CI]"] = AUROC칸(a)
    print(f"  {g:11} {열[g][f'{개}{칸글}']} · AUROC12 {열[g][f'{개}AUROC, Hb <12 g/dL, median (seed range) [95% CI]']}", flush=True)
# 한 눈 — 사진 한 장 = 한 사람 값
E1 = 내부교차("mnl", 권a, "auto_scca", 값="e1"); x = 짝(E1, E["auto_scca"])
for g in 입력들:
    a = g == "auto_scca"
    열[g][f"{개}single eye instead of both-eye average, {칸글}"] = 칸값(E1) if a else "··"
    열[g][f"{개}p value, single eye vs both-eye average"] = p표(x[3]) if a else "··"
print(f"  한 눈 {칸값(E1)} · 차이 {차이글(x)} · p {p표(x[3])}", flush=True)
for site in SITES:
    이 = 영문사이트[site]
    입력블록(열, f"{이}, before fine-tuning: ", {g: 무보정앙상블("mnl", 권a, site, g) for g in 입력들}, 비열등판정=False)
    입력블록(열, f"{이}, after fine-tuning with {구성글(권a, 권폴)}: ", {g: 외부앙상블("mnl", 권a, 권폴, site, 입력=g) for g in 입력들})
T = pd.DataFrame({입력열[g]: v for g, v in 열.items()}); T.index.name = "Measure"; T = T.reset_index()
T, 머리 = 소제목표(T); print(T.to_string(index=False))
엑셀("Table4_input_bilateral_roi", 본문, {"Table 4": T}, 소제목=머리,
    제목=("Table 4. Input region, bilateral averaging and source of the region of interest in the development cohort and at the "
         "two external locations before and after fine-tuning (development model, MobileNetV3-Large with S1 + S2; fine-tuning added S3)"),
    주석=보고각주 + [
      "Columns are the five inputs; SCCA, sclera and conjunctiva; conjunctiva, palpebral conjunctiva; automatic, region from the",
      "  automated segmentation; manual, region of interest drawn by hand. All five use MobileNetV3-Large with final-feature layer",
      "  normalisation (S1) and an adaptation layer (S2), trained identically. Fine-tuning used S1 + S2 + S3, adding the backbone's",
      "  batch-normalisation statistics held during fine-tuning (S3). Before fine-tuning: all participants. After fine-tuning: evaluation participants only.",
      "Development cohort: 762 participants, 1524 images. Single eye: each image is a separate estimate in the same participants.",
      "AUROC: computed for each seed with the negated predicted haemoglobin as the score.",
      "Comparisons are within the same participants. p value vs reference: each input compared with automatic SCCA, two-sided.",
      "Automatic vs manual, same region (manual columns): automatic minus manual error; positive values mean a larger error with the",
      "  automatic region. Non-inferiority margin δ = 0·089 g/dL, the optimal specification for bias derived from biological variation",
      "  (0·125 x sqrt(CVI^2 + CVG^2) at the mean haemoglobin of the cohort). Non-inferior: the upper 95% limit is below δ; not shown:",
      "  it is not; p is the one-sided p for a difference of at least δ. The margin is used for this comparison only.",
      "Results are reported by location and are not pooled.",
    ])
