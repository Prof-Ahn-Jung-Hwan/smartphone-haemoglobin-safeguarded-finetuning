# -*- coding: utf-8 -*-
"""**Table 3. 안전장치 (11-09 번호 바꿈 · 옛 Table 4) · ablation — MobileNetV3-L** (11-04 : 1 seed 10겹 앙상블 · 중간값 (범위) · 표 코드는 `_공통.안전장치표` · 옛 판 _old_261030/)

1차 자료  main/results/{stage5,기전,M빼기LN,최소구성}/{train,external/{zero,few,few_bn}} · main_results/P3_추론비용.csv
내는 것   results_table_figure/main/Table3_ablation_mobilenetv3_10seed.xlsx
쓰는 법   python main_table3_ablation_mobilenetv3_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

시작("Table 3 — 안전장치 ablation (MobileNetV3-L)")
T, 머리 = 안전장치표(["mnl"], 기준선줄=True)
엑셀("Table3_ablation_mobilenetv3_10seed", 본문, {"Table 3": T}, 소제목=머리,
    제목=("Table 3. Ablation of the three safeguards in MobileNetV3-Large: mean absolute error in the development cohort and at "
         "each external location before and after local fine-tuning"),
    주석=안전장치표_각주(True))
