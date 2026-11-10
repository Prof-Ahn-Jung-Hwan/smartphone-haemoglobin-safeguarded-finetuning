# -*- coding: utf-8 -*-
"""**Table ST7. 입력 비교 — 안전장치 없는 모델** (11-04 새로 · 지침 §14 · Table 3 과 같은 칸)

칸   개발 (E04b · 자동 SCCA 는 E03) · 기관마다 안전장치 없는 미세조정 (E15 · 자동 SCCA 는 E08). 미세조정 전은 하지 않는다 (§14).
🔵 run_all 뒤에 도는 E04b · E15 가 끝나야 만들어진다. 그 전에는 "아직" 이라 찍고 정상 종료한다.
1차 자료  main/results/stage5/train · stage5/external/few
내는 것   results_table_figure/supplementary/TableST7_inputs_without_safeguards_10seed.xlsx
쓰는 법   python supp_tableST7_inputs_without_safeguards_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

입력들 = list(입력열)
없음 = [g for g in 입력들 if 내부값("mnl", "mse", g) is None] + [f"{g} {s}" for g in 입력들 for s in SITES if 외부값("mnl", "mse", FT, s, 입력=g) is None]
if 없음 and not 빈칸허용:
    print(f"  ⏳ 아직 없다 (E04b · E15 뒤에 다시 돌린다) : {없음[:4]} … 모두 {len(없음)}칸"); sys.exit(0)
시작("Table ST7 — 입력 비교 (안전장치 없는 모델)")
열 = {g: {} for g in 입력들}
입력블록(열, "Development cohort: ", {g: 내부교차("mnl", "mse", g) for g in 입력들})
for site in SITES:
    입력블록(열, f"{영문사이트[site]}, after fine-tuning without safeguards: ", {g: 외부앙상블("mnl", "mse", FT, site, 입력=g) for g in 입력들})
T = pd.DataFrame({입력열[g]: v for g, v in 열.items()}); T.index.name = "Measure"; T = T.reset_index()
T, 머리 = 소제목표(T); print(T.to_string(index=False))
엑셀("TableST7_inputs_without_safeguards_10seed", 부록, {"ST7": T}, 소제목=머리,
    제목="Table ST7. Input comparisons in MobileNetV3-Large without safeguards, in the development cohort and after fine-tuning at the two external locations",
    주석=보고각주 + [
      "Same comparisons as table 4, with the model trained and fine-tuned without safeguards. Columns as in table 4. Non-inferiority margin",
      "  δ = 0·089 g/dL. Input comparisons used the primary backbone. Results are reported by location and are not pooled.",
    ])
