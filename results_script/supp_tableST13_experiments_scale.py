# -*- coding: utf-8 -*-
"""**Table ST13. 실험 · 규모 · 결과가 실린 곳** (11-10 새로 만듦)

무엇  논문에 보고한 실험마다 설계, 학습 · 미세조정 횟수, Results 절, 본문 표 · 그림, 부록 항목을 한 줄씩.
      횟수는 결과 파일을 직접 세어 확인한다 (설계에서 계산한 값과 다르면 멈춘다).
      논문에서 뺀 실험(보조 손실 anchor* · batchmean, 학습률 2e-5 의 옛 본 분석 판, 다른 백본의 2e-5 판, 2e-5 입력 비교)은 세지 않는다.
1차 자료  main/results/grid/val_pred · {stage5,기전,M빼기LN,최소구성,백본재현}/train · …/external · 임계/외부_안쪽예측
내는 것   results_table_figure_lr1e-4/supplementary/TableST13_experiments_scale.xlsx  (있으면 멈춘다 — 덮어쓰지 않는다)
쓰는 법   (프로젝트 뿌리에서) python -u paperwork/results_script/supp_tableST13_experiments_scale.py
"""
import sys, glob; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

R = 뿌리 / "main/results"
출력 = 부록 / "TableST13_experiments_scale.xlsx"
if 출력.exists(): sys.exit(f"🔴 {출력.name} 가 이미 있다 — 덮어쓰지 않는다. 옛 판을 _previous_results_<날짜>/supplementary 로 먼저 옮긴다")

폴더 = {"mse": "stage5", "normonly": "기전", "veminimalnoln": "M빼기LN", "veminimal": "최소구성"}   # MobileNetV3-Large
구성4 = ["mse", "normonly", "veminimalnoln", "veminimal"]                                         # 없음 · S1 · S2 · S1+S2
다른입력 = ["auto_bc", "auto_bcwb", "human_bc", "human_scca"]
땅 = ["India", "Italy"]


def 개(pat):
    return len(glob.glob(str(R / pat)))


def 확인(이름, 센, 기대):
    print(f"  {'✅' if 센 == 기대 else '🔴'} {이름:52} 센 값 {센:>7,} · 설계 {기대:>7,}")
    if 센 != 기대: sys.exit("🔴 설계와 다르다 — 멈춤")
    return 센


def 학습(d, 이름):        # 개발 코호트 10겹 모델 (seed 10 × 겹 10)
    return 개(f"{d}/train/{이름}_seed*-fold*")


def 미세(d, ft, n, 이름):  # 외부 미세조정 = fold*.csv 하나가 미세조정 한 번
    return sum(개(f"{d}/external/{ft}/{s}/n{n}/rep*/{이름}_seed*/fold*.csv") for s in 땅)


시작("Table ST13 — 실험과 규모 (결과 파일을 세어 확인)")

# ── 개발 코호트 ──
격자 = pd.read_csv(R / "grid/lr_선택.csv"); 격자.columns = [c.lstrip("﻿") for c in 격자.columns]
격자 = 격자[격자["경로"] == "A"]
설정 = int(격자["시도한_lr"].str.split().str.len().sum())
lr범위 = 격자["시도한_lr"].str.split().str.len()
n격자 = 확인("백본 · 학습률 선별 (검증 예측 칸)", len(glob.glob(str(R / "grid/val_pred/*.csv"))), 설정 * 3 * 2)

n개발안전 = 0
for c in 구성4: n개발안전 += 학습(폴더[c], f"*_mnl_{c}_K40_auto_scca")
for b in ["eb0", "mnv2", "mvxx", "enxs"]:
    for c in 구성4: n개발안전 += 학습("백본재현", f"bb_{b}_{c}_K40_auto_scca")
확인("안전장치 · 개발 (백본 5 × S1×S2 4 × 100)", n개발안전, 5 * 4 * 100)

n개발입력 = sum(학습(폴더[c], f"*_mnl_{c}_K40_{i}") for c in ["mse", "veminimal"] for i in 다른입력)
확인("입력 · 개발 (다른 입력 4 × 구성 2 × 100)", n개발입력, 4 * 2 * 100)

# ── 외부 미세조정 ──
n요인 = 0
for c in 구성4:
    for ft in ["few_lr0.0001", "few_lr0.0001_bn"]: n요인 += 미세(폴더[c], ft, 30, f"*_mnl_{c}_K40_auto_scca")
for b in ["eb0", "mnv2", "mvxx"]:
    for c in 구성4:
        for ft in ["few_lr0.0001", "few_lr0.0001_bn"]: n요인 += 미세("백본재현", ft, 30, f"bb_{b}_{c}_K40_auto_scca")
for c in 구성4: n요인 += 미세("백본재현", "few_lr0.0001", 30, f"bb_enxs_{c}_K40_auto_scca")
확인("안전장치 · 외부 (36 조합 × 1000 × 2 기관)", n요인, 36 * 1000 * 2)

n학습률 = 0
for c in 구성4:
    for ft in ["few", "few_bn", "few_lr0.001", "few_lr0.001_bn"]: n학습률 += 미세(폴더[c], ft, 30, f"*_mnl_{c}_K40_auto_scca")
for ft in ["few_lr0.0003_bn", "few_lr0.003_bn"]: n학습률 += 미세("최소구성", ft, 30, "mn_mnl_veminimal_K40_auto_scca")
확인("학습률 민감도 (18 설정 × 1000 × 2)", n학습률, 18 * 1000 * 2)

n인원 = sum(미세("최소구성", "few_lr0.0001_bn", n, "mn_mnl_veminimal_K40_auto_scca") for n in [10, 20, 40, 50])
확인("미세조정 인원 (4 × 1000 × 2)", n인원, 4 * 1000 * 2)

n외부입력 = sum(미세("최소구성", "few_lr0.0001_bn", 30, f"mn_mnl_veminimal_K40_{i}") + 미세("stage5", "few_lr0.0001", 30, f"s5_mnl_mse_K40_{i}") for i in 다른입력)
확인("입력 · 외부 (다른 입력 4 × 구성 2 × 1000 × 2)", n외부입력, 8 * 1000 * 2)

안쪽 = sorted(glob.glob(str(R / "임계/외부_안쪽예측/lr0.0001_bn/*.csv")))
확인("현지 문턱 · 안쪽 3겹 (파일)", len(안쪽), 1000 * 2)
줄 = {len(pd.read_csv(f)) for f in 안쪽[:50]}
if 줄 != {30}: sys.exit(f"🔴 안쪽 예측 파일의 행 수가 30 이 아니다 : {줄}")
n문턱 = len(안쪽) * 3

개발합 = n격자 + n개발안전 + n개발입력
외부합 = n요인 + n학습률 + n인원 + n외부입력 + n문턱
print(f"  ▶ 개발 학습 {개발합:,} 번 · 외부 미세조정 {외부합:,} 번")

f = lambda x: f"{x:,}".replace(",", " ") if x >= 10000 else str(x)   # 11-11 : Lancet 숫자 형식 (3160 · 138 000)
행 = [
  ("1", "Automated segmentation", "Can the eye region be obtained without manual drawing?",
   "SAM 3 with everyday text prompts; 1524 development, 95 Indian and 122 Italian photographs; two readers", "No training",
   "Automated segmentation", "Table 2", "Figure S1 (segmentation examples)"),
  ("2", "Backbone and learning-rate screening (development)", "Which mobile backbones, at which learning rate?",
   f"10 backbones × {lr범위.min()}–{lr범위.max()} learning rates ({설정} settings) × 3 folds × 2 inputs; no safeguards", f"{f(n격자)} training runs",
   "Development: backbone and learning-rate screening", "Figure 4A", "Table S3"),
  ("3", "Safeguards in the development cohort", "Do S1 and S2 change development error?",
   "5 backbones × 4 configurations (S1 × S2) × 10 seeds × 10 folds", f"{f(n개발안전)} training runs",
   "Development: backbone and learning-rate screening", "Table 3", "Tables S4, S8"),
  ("4", "External evaluation before fine-tuning", "Does the model transfer unchanged?",
   "The models of step 3 applied unchanged at both locations; photo-free baseline", "No new training",
   "External evaluation before fine-tuning", "Table 3", "Table S5"),
  ("5", "Safeguards during local fine-tuning (five backbones)", "Does fine-tuning with 30 local participants help, and do the safeguards make it reliable?",
   "36 backbone–configuration combinations × 10 seeds × 10 fold models × 10 draws × 2 locations (learning rate 1 × 10⁻⁴)", f"{f(n요인)} fine-tuning runs",
   "Local fine-tuning; Safeguards in five backbones", "Table 3, Figure 4B", "Tables S7, S8"),
  ("6", "Sensitivity to the fine-tuning learning rate", "Do the findings depend on the learning rate?",
   "MobileNetV3-Large: 8 configurations at 2 × 10⁻⁵ and 1 × 10⁻³; S1+S2+S3 also at 3 × 10⁻⁴ and 3 × 10⁻³; × 1000 × 2 locations", f"{f(n학습률)} fine-tuning runs",
   "Sensitivity analyses", "—", "Table S6"),
  ("7", "Number of fine-tuning participants", "How many local participants are needed?",
   "10, 20, 40 and 50 participants (30 in step 5) × 1000 × 2 locations; same evaluation participants", f"{f(n인원)} fine-tuning runs",
   "Sensitivity analyses", "—", "Figure S2"),
  ("8", "Individual fold models and output recalibration", "Is averaging needed, and is retraining needed at all?",
   "Fold models evaluated individually, without averaging predictions; shift or straight line fitted in the same 30 participants", "No new training (runs of step 5)",
   "Sensitivity analyses", "—", "Tables S5, S6"),
  ("9", "Recommended model", "Which backbone and configuration to recommend?",
   "Rule applied to the 36 combinations of step 5, at both locations and at each alone", "No new training",
   "Recommended model", "Figure 4C", "Table S4"),
  ("10", "Input region and automated versus manual regions", "Is automated SCCA as good as manual delineation, and better than the conjunctiva alone?",
   f"MobileNetV3-Large, 4 further inputs × 2 configurations: development 10 seeds × 10 folds; external × 1000 × 2 locations", f"{f(n개발입력)} training runs; {f(n외부입력)} fine-tuning runs",
   "Input region and automated versus manual regions", "Table 4", "Table S9"),
  ("11", "Decision thresholds", "Does the decision cut-off transfer?",
   "Development cut-off from inner validation sets; local cut-off by 3-fold cross-validation within the 30 fine-tuning participants (refitted in each fold)", f"{f(n문턱)} fine-tuning runs",
   "Decision thresholds", "Table 5", "—"),
  ("", "Total", "", "", f"{f(개발합)} training runs in the development cohort; {f(외부합)} local fine-tuning runs", "", "", ""),
]
T = pd.DataFrame(행, columns=["Step", "Experiment", "Question", "Design", "Training or fine-tuning runs",
                              "Results section", "Main table or figure", "Appendix item"])
print(T[["Step", "Experiment", "Training or fine-tuning runs"]].to_string(index=False))

엑셀("TableST13_experiments_scale", 부록, {"ST13": T},
    제목="Table ST13. Stages of the study, their scale and where their results are reported",
    주석=[
      "Counts were obtained by counting the saved model outputs and checked against the design; experiments not reported in the paper are not counted.",
      "Training run: one model trained in the development cohort (one fold of one seed, or one cell of the learning-rate screen).",
      "Fine-tuning run: one fold model fine-tuned once at one external location with one random draw of local participants.",
      "Steps 5 to 11 used the learning rate of 1 × 10⁻⁴ unless stated; step 6 is a sensitivity analysis, not a selection of the learning rate.",
      "S1, final-feature layer normalisation; S2, adaptation layer; S3, backbone batch-normalisation statistics held during fine-tuning.",
      "Appendix items are numbered in order of first citation in the main text.",
    ])
print(f"  ✅ {출력}")
