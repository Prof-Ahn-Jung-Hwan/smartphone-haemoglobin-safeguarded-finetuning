# -*- coding: utf-8 -*-
"""**Table ST3. 안전장치 ablation — 다른 네 백본** (11-04 : Table 4 와 같은 단위 · 같은 코드 · 옛 판 _old_261030/)

1차 자료  main/results/{stage5,백본재현}/{train,external/{zero,few,few_bn}} · main_results/P3_추론비용.csv
내는 것   results_table_figure/supplementary/TableST3_ablation_other_backbones_10seed.xlsx
쓰는 법   python supp_tableST3_ablation_other_backbones_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

시작("Table ST3 — 안전장치 ablation (다른 네 백본)")
T, 머리 = 안전장치표([b for b in ["eb0", "enxs", "mnv2", "mvxx"] if b in BB])
엑셀("TableST3_ablation_other_backbones_10seed", 부록, {"ST3": T}, 소제목=머리,
    제목=("Table ST3. Ablation of the three safeguards in the other four mobile backbones: mean absolute error in the development "
         "cohort and at each external location before and after local fine-tuning"),
    주석=안전장치표_각주(False))
