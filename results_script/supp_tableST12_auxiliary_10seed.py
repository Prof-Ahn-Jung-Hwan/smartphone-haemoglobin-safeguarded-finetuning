# -*- coding: utf-8 -*-
"""**Table ST12. 보조 장치 (S4 · S4n · S5 · S4 + S5) 를 더하면** (11-04 : 표 안에서 계산 · 1 seed 10겹 앙상블 · 옛 판 _old_261030/)

내는 것   results_table_figure/supplementary/TableST12_auxiliary_losses_10seed.xlsx
쓰는 법   python supp_tableST12_auxiliary_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

이름 = {"anchor": "S4 (anchor alignment loss)", "anchorshuf": "S4n (shuffled labels, negative control)", "batchmean": "S5 (batch-mean regulariser)", "anchorbm": "S4 + S5"}
절 = {FT: "BN statistics updated (reference S1 + S2)", FTB: "BN statistics frozen (reference S1 + S2 + S3)"}
시작("Table ST12 — 보조 장치")
행 = []; 차 = []
for b in BB:
    for site in SITES:
        for 폴더 in (FT, FTB):
            if b == "enxs" and 폴더 == FTB: continue
            기 = 외부앙상블(b, "veminimal", 폴더, site)
            if 기 is None: continue
            for a in ["anchor", "anchorshuf", "batchmean"] + (["anchorbm"] if b == "mnl" else []):
                v = 외부앙상블(b, a, 폴더, site)
                if v is None: continue
                x = 짝(v, 기); 차.append((a, x[0], x[3]))
                행.append({"Backbone": 표시[b], "Location": 영문사이트[site], "Fine-tuning": 절[폴더], "Auxiliary component added to S1 + S2": 이름[a],
                          f"With the component, {칸글}": 칸값(v), f"Reference, {칸글}": 칸값(기), "p value, with vs without the component": p표(x[3])})
                print(f"  {표시[b]:16} {site:6} {폴더:7} {a:11} {칸값(v)} 기준 {칸값(기)} p {p표(x[3])}", flush=True)
c = [x for x in 차 if x[0] != "anchorshuf"]; 낮 = sum(x[2] < .05 and x[1] < 0 for x in c); 높 = sum(x[2] < .05 and x[1] > 0 for x in c); 큰 = max(abs(x[1]) for x in c)
print(f"  S4n 제외 {len(c)}개 · 낮아짐 {낮} · 높아짐 {높} · |차이| 최대 {큰:.3f}")
엑셀("TableST12_auxiliary_losses_10seed", 부록, {"ST12": pd.DataFrame(행)},
    제목="Table ST12. Adding auxiliary losses to S1 + S2: five backbones, two locations and two fine-tuning procedures",
    주석=보고각주 + [
      "Each row compares the configuration with the component with the same configuration without it (the reference), fine-tuned with the",
      "  same procedure, in the same participants.",
      f"Of {len(c)} comparisons of S4, S5 or S4 + S5, {낮} showed a lower and {높} a higher error (all differences within {점(큰)} g/dL). S4n is a",
      "  negative control for S4. S4 + S5 was evaluated in MobileNetV3-Large. EdgeNeXt-XS contains no batch normalisation.",
      "Results are reported by location and are not pooled.",
    ])
