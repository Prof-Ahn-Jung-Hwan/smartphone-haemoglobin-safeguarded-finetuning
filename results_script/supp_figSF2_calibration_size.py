# -*- coding: utf-8 -*-
"""**Supplementary Figure SF2. 미세조정 인원 곡선** — SS (11-04 : 1 seed 10겹 앙상블 · 중간값 · 옛 판 _old_261030/)

무엇  현지 10 · 20 · 30 · 40 · 50명. 점 = seed 중간값 · 막대 = 95 % CI. 후보 50명 앞에서 n 명을 쓰므로 평가 참여자가 같다 (인도 45 · 이탈리아 72).
1차 자료  main/results/최소구성/external/few_bn/{India,Italy}/n{10,20,30,40,50} (E08 · E11)
내는 것   results_table_figure/supplementary/FigSF2_calibration_size.{pdf,svg,png,xlsx}
쓰는 법   python supp_figSF2_calibration_size.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = [10, 20, 30, 40, 50]
시작("Figure SF2 — 미세조정 인원 (S1 + S2 + S3)")
행 = []
for site in SITES:
    값 = {n: 외부앙상블(*권유, site, n=n) for n in NS}      # ★ 11-07 : 권유 구성 · 학습률
    for n in [n for n in NS if 값[n] is not None]:
        m, r1, r2, lo, hi = 요약(값[n]); x = None if n == 30 else 짝(값[n], 값[30])
        행.append({"Location": 영문사이트[site], "Fine-tuning participants": n, "Participants": len(값[n]), 칸글: 칸값(값[n]), "_m": m, "_lo": lo, "_hi": hi,
                  "Difference from 30 participants, g/dL (95% CI)": "Reference" if x is None else 차이글(x), "p": "··" if x is None else p표(x[3])})
        print(f"  {site:6} n{n:2} {칸값(값[n])}" + ("" if x is None else f" · {차이글(x)} p {p표(x[3])}"), flush=True)
T = pd.DataFrame(행)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for site, 색, 표, dx in [("India", "#c0504d", "o", -0.6), ("Italy", "#1f5fa8", "s", 0.6)]:
    g = T[T.Location == 영문사이트[site]]
    ax.errorbar(g["Fine-tuning participants"] + dx, g._m, yerr=[g._m - g._lo, g._hi - g._m], fmt=표 + "-", color=색, ms=4, lw=1, capsize=2, elinewidth=0.8, label=영문사이트[site])
ax.axvline(30, ls=":", lw=0.7, color="k"); ax.set_xticks(NS); ax.set_xlabel("Local participants used for fine-tuning")
ax.set_ylabel("Mean absolute error, g/dL (median, 95% CI)"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("pdf", "svg", "png"): fig.savefig(부록 / f"FigSF2_calibration_size.{ext}", dpi=300, bbox_inches="tight")
엑셀("FigSF2_calibration_size", 부록, {"SF2": T.drop(columns=["_m", "_lo", "_hi"])},
    제목="Figure SF2 data. External error of MobileNetV3-Large with S1 + S2 + S3 by the number of local fine-tuning participants",
    주석=보고각주 + ["The 50 candidates were drawn at random 10 times; the first n were used, so the evaluation",
                  "  participants are the same for every n. Differences are n participants minus 30 participants; positive values mean a larger error."])
