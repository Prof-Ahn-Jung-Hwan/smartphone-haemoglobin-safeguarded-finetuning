# -*- coding: utf-8 -*-
"""**Table ST4. 안전장치 상호작용 — 다섯 백본** (11-04 : 1 seed 10겹 앙상블 대비의 seed 중간값 · 옛 판 _old_261030/)

무엇  참여자 × seed 마다 대비 → seed 값의 중간값 · 참여자 bootstrap. 주효과는 Table 4 · ST3 (화면에만 찍는다).
내는 것   results_table_figure/supplementary/TableST4_factorial_backbones_10seed.xlsx
쓰는 법   python supp_tableST4_factorial_backbones_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

팔 = {"mse": (0, 0), "normonly": (1, 0), "veminimalnoln": (0, 1), "veminimal": (1, 1)}
시작("Table ST4 — 상호작용 (다섯 백본)")
행 = []
for b in BB:
    for site in SITES:
        K = (0, 1) if b != "enxs" else (0,); g = {}
        for a, (i, j) in 팔.items():
            for k in K: g[(i, j, k)] = 외부앙상블(b, a, FTB if k else FT, site)
        if any(v is None for v in g.values()): continue
        y = {k: v.to_numpy(float) for k, v in g.items()}
        효 = {"S1 x S2 interaction": np.mean([(y[(1, 1, k)] - y[(0, 1, k)]) - (y[(1, 0, k)] - y[(0, 0, k)]) for k in K], axis=0)}
        if b != "enxs":
            효["S1 x S3 interaction"] = np.mean([(y[(1, j, 1)] - y[(0, j, 1)]) - (y[(1, j, 0)] - y[(0, j, 0)]) for j in (0, 1)], axis=0)
            효["S2 x S3 interaction"] = np.mean([(y[(i, 1, 1)] - y[(i, 0, 1)]) - (y[(i, 1, 0)] - y[(i, 0, 0)]) for i in (0, 1)], axis=0)
            효["S1 x S2 x S3 interaction"] = ((y[(1, 1, 1)] - y[(0, 1, 1)]) - (y[(1, 0, 1)] - y[(0, 0, 1)])) - ((y[(1, 1, 0)] - y[(0, 1, 0)]) - (y[(1, 0, 0)] - y[(0, 0, 0)]))
        for k, v in 효.items():
            x = 대비(v)
            행.append({"Backbone": 표시[b], "Location": 영문사이트[site], "Participants": len(v), "Term": k,
                      "Interaction, g/dL, median (95% CI)": 차이글(x), "p value": p표(x[3])})
            print(f"  {표시[b]:16} {site:6} {k:26} {차이글(x)} p {p표(x[3])}", flush=True)
엑셀("TableST4_factorial_backbones_10seed", 부록, {"ST4": pd.DataFrame(행)},
    제목="Table ST4. Interactions between the three safeguards after transfer, five backbones",
    주석=[
      "Interaction: the difference between the conditional effects of the first safeguard with and without the second, in absolute error",
      "  (g/dL), averaged over the remaining safeguard (the three-way term: the S1 x S2 interaction with S3 minus that without S3), for",
      "  each participant and seed, and summarised as the median across 10 seeds. Main effects are in table 3 and the appendix.",
      "S1, final-feature layer normalisation; S2, adaptation layer; S3, backbone batch-normalisation statistics held during fine-tuning.",
      "  EdgeNeXt-XS contains no batch normalisation (2 x 2 design of S1 and S2 only).",
      "Predictions of the 10 fold models fine-tuned on the same 30 local participants were averaged; 10 random draws. 95% CIs and two-sided",
      "  p values (unadjusted) from a participant bootstrap (100,000 resamples). Results are reported by location and are not pooled.",
    ])
