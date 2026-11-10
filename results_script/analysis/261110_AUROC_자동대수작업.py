# -*- coding: utf-8 -*-
"""자동 대 수작업 영역의 AUROC 비교 (11-10 · 사용자 질문 "AUROC 에 대한 p value 구해 볼 수 있나?")

무엇  같은 영역에서 자동 분할 대 수작업 분할의 AUROC (Hb <12·0 · <11·0) 와 차이의 양측 p
      영역 : SCCA · 결막 단독
      어디 : 개발 코호트 (MobileNetV3-L · S1+S2 · 10겹 교차검증) · 인도 · 이탈리아 (권유 모델 · 현지 30명 미세조정 뒤)
값    `_공통.AUROC요약` 그대로 — seed 마다 AUROC → 중간값 (seed 범위) [95% CI] · 같은 재표집에서 두 중간값의 차이 → 양측 p (참여자 bootstrap 100,000)
      표 4 · 표 5 의 AUROC 와 같은 점수 (−예측 Hb · 외부는 추첨 평균)
내는 것   results_table_figure_lr1e-4/_심사대응_준비/261110_AUROC_auto_vs_manual.csv  (있으면 멈춘다 — 덮어쓰지 않는다)
쓰는 법   (프로젝트 뿌리에서) python -u paperwork/results_script/analysis/261110_AUROC_자동대수작업.py
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _공통 import *

출력 = 뿌리 / "paperwork/results_table_figure_lr1e-4/_심사대응_준비/261110_AUROC_auto_vs_manual.csv"
if 출력.exists(): sys.exit(f"🔴 {출력.name} 가 이미 있다 — 덮어쓰지 않는다")
_, 권a, 권폴 = 권유
if (권a, 권폴) != ("veminimal", "few_lr0.0001_bn"): sys.exit(f"🔴 권유 구성이 다르다 : {권a} {권폴}")

시작("AUROC — 자동 대 수작업 (같은 영역)")
행 = []
for 영역, (자동, 수작업) in {"SCCA": ("auto_scca", "human_scca"), "Conjunctiva": ("auto_bc", "human_bc")}.items():
    for 어디 in ["Development cohort", "India", "Italy"]:
        if 어디 == "Development cohort":
            A, M = 내부값("mnl", 권a, 자동), 내부값("mnl", 권a, 수작업)
        else:
            A, M = 외부값("mnl", 권a, 권폴, 어디, 입력=자동), 외부값("mnl", 권a, 권폴, 어디, 입력=수작업)
        A, M = 필수(A, f"{어디} {자동}"), 필수(M, f"{어디} {수작업}")
        PA, PM = 행렬(A, "p"), 행렬(M, "p")
        if list(PA.index) != list(PM.index): sys.exit(f"🔴 {어디} {영역} : 자동 · 수작업 참여자가 다르다")
        y = A[A.seed == 1].set_index("key").y.reindex(PA.index).to_numpy(float)
        for t in (12.0, 11.0):
            a, m = AUROC요약(y < t, -PA.to_numpy(float).T, -PM.to_numpy(float).T)
            r = {"Region": 영역, "Cohort": 어디, "Target": f"Hb <{점(t, 1)} g/dL", "Participants": len(y), "Anaemic": int((y < t).sum()),
                 "AUROC automatic": AUROC칸(a), "AUROC manual": AUROC칸(m[:5]), "Difference (median)": 점(a[0] - m[0], 3, 부호=True),
                 "p value (automatic vs manual)": p표(m[5])}
            행.append(r); print(f"  {영역:11} {어디:18} {r['Target']:14} 자동 {r['AUROC automatic']} · 수작업 {r['AUROC manual']} · p {r['p value (automatic vs manual)']}", flush=True)
출력.parent.mkdir(exist_ok=True)
pd.DataFrame(행).to_csv(출력, index=False, encoding="utf-8-sig")
print(f"  ✅ {출력}")
