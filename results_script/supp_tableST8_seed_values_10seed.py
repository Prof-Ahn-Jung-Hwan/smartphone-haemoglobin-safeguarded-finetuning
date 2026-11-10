# -*- coding: utf-8 -*-
"""**Table ST8. 권유 모델의 seed 별 MAE** (11-04 새로 · 지침 맨 위 칸 "seed 별 값" · 권유 모델 하나 · 10줄)

내는 것   results_table_figure/supplementary/TableST8_seed_values_10seed.xlsx
쓰는 법   python supp_tableST8_seed_values_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

b, a, 폴더 = 권유; 이름 = f"{표시[b]}, {구성글(a, 폴더)}"
시작(f"Table ST8 — seed 별 MAE ({이름})")
개 = 내부교차(b, a).mean(0); 외 = {s: 외부앙상블(b, a, 폴더, s).mean(0) for s in SITES}
T = pd.DataFrame([{"Seed": s, "Development cohort, MAE, g/dL": 점(float(개[s])),
                   **{f"{영문사이트[x]}, MAE, g/dL": 점(float(외[x][s])) for x in SITES}} for s in S10])
print(T.to_string(index=False))
엑셀("TableST8_seed_values_10seed", 부록, {"ST8": T},
    제목=f"Table ST8. Mean absolute error of the recommended model ({이름}) for each of the 10 seeds",
    주석=["Development cohort: 10-fold cross-validation MAE. External locations: MAE of the 10 fold models fine-tuned on the same 30 local",
         "  participants with predictions averaged, averaged over 10 random draws of participants."])
