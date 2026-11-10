# -*- coding: utf-8 -*-
"""**Figure 3. 안전장치 없이 미세조정하면 이탈리아에서 나빠지고, SS 로 미세조정하면 두 곳 모두 좋아진다** — 다섯 백본 (11-04 : 새 칸 형식을 읽는다 · 옛 판 _old_261030/)

무엇  두 외부 기관 × 백본 다섯 · 백본마다 화살표 두 개 (미세조정 전 → 안전장치 없이 · 전 → SS). 점 = seed 중간값 · 막대 = 95 % CI.
      새로 계산하지 않는다 — Table 3 (11-09 번호 바꿈 · 옛 Table 4) · ST3 의 칸을 그대로 읽는다.
1차 자료  results_table_figure/main/Table3_ablation_… · supplementary/TableST3_… (두 표를 먼저 만든다)
내는 것   results_table_figure/main/Figure3_calibration_safeguards_10seed.{pdf,svg,png,xlsx}
쓰는 법   python main_figure3_calibration_safeguards_10seed.py
"""
import sys, re; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
import openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

표들 = [본문 / "Table3_ablation_mobilenetv3_10seed.xlsx", 부록 / "TableST3_ablation_other_backbones_10seed.xlsx"]
순서 = ["MobileNetV3-L", "EfficientNet-B0", "MobileNetV2", "MobileViT-XXS", "EdgeNeXt-XS"]
열 = {"India": ("Indian location: before fine-tuning", "Indian location: after fine-tuning", "Indian location: p value"),
     "Italy": ("Italian location: before fine-tuning", "Italian location: after fine-tuning", "Italian location: p value")}


def 수(s):   # "1·45 (1·40–1·49) [1·23–1·67]" → (1.45, 1.23, 1.67) = 중간값 · 95 % CI
    m = re.match(r"([\d·]+) \([\d·]+–[\d·]+\) \[([\d·]+)–([\d·]+)\]", str(s).strip())
    if not m: sys.exit(f"🔴 MAE 칸을 읽지 못했다 : {s!r}")
    return tuple(float(x.replace("·", ".")) for x in m.groups())


def 읽기(f):
    if not f.exists(): sys.exit(f"🔴 {f.name} 가 없다 — 표 스크립트를 먼저 돌린다")
    ws = openpyxl.load_workbook(f).active; 행 = [list(r) for r in ws.iter_rows(values_only=True)]
    머 = next(i for i, r in enumerate(행) if r and r[0] == "Configuration"); 이름 = 행[머]
    블록 = {}; 지금 = None
    for r in 행[머 + 1:]:
        if not r or r[0] is None: continue
        첫 = str(r[0])
        if 첫 == "Notes": break
        b = next((x for x in 순서 if 첫.startswith(x)), None)
        if b and all(v in (None, "") for v in r[1:]): 지금 = b; 블록[b] = {}; continue
        if 지금: 블록[지금][첫] = dict(zip(이름, r))
    return 블록


for ext in ("pdf", "svg", "png", "xlsx"):
    if (본문 / f"Figure3_calibration_safeguards_10seed.{ext}").exists(): sys.exit(f"🔴 Figure3_calibration_safeguards_10seed.{ext} 가 이미 있다 — 덮어쓰지 않는다. 옛 판을 먼저 옮긴다")
시작("Figure 3 — 보정 전후 · 안전장치 없음 vs S1 + S2 + S3 (다섯 백본)")
블록 = {}
for f in 표들: 블록.update(읽기(f))
if set(블록) != set(순서): sys.exit(f"🔴 백본이 모자라다 : {sorted(블록)}")
행 = []
for b in 순서:
    g = 블록[b]; ss = "S1+S2+S3" if "S1+S2+S3" in g else "S1+S2"
    for site, (전, 후, p열) in 열.items():
        for 팔, 줄, 전줄 in [("No safeguards", "No safeguards", "No safeguards"), ("With safeguards", ss, "S1+S2")]:
            a = 수(g[전줄][전]); z = 수(g[줄][후])
            행.append({"Location": 영문사이트[site], "Backbone": b, "Arm": 팔, "Configuration after fine-tuning": 줄,
                      "Before fine-tuning, MAE g/dL": a[0], "Before 95% CI low": a[1], "Before 95% CI high": a[2],
                      "After fine-tuning, MAE g/dL": z[0], "After 95% CI low": z[1], "After 95% CI high": z[2],
                      "Holm-adjusted p, after vs before": g[줄][p열]})
T = pd.DataFrame(행)
for _, r in T.iterrows():
    print(f"  {r.Location:16} {r.Backbone:15} {r.Arm:14} {r['Before fine-tuning, MAE g/dL']:.2f} → {r['After fine-tuning, MAE g/dL']:.2f}  p {r['Holm-adjusted p, after vs before']}")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
색 = {"No safeguards": "#8c8c8c", "With safeguards": "#1f5fa8"}
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
for ax, (site, 글) in zip(axs, [("Indian location", "A"), ("Italian location", "B")]):
    g = T[T.Location == site]
    for i, b in enumerate(순서):
        for 팔, dx in [("No safeguards", -0.16), ("With safeguards", 0.16)]:
            r = g[(g.Backbone == b) & (g.Arm == 팔)].iloc[0]; x0, x1 = i + dx - 0.07, i + dx + 0.07
            for x, m, lo, hi, 채움 in [(x0, r["Before fine-tuning, MAE g/dL"], r["Before 95% CI low"], r["Before 95% CI high"], "white"),
                                      (x1, r["After fine-tuning, MAE g/dL"], r["After 95% CI low"], r["After 95% CI high"], 색[팔])]:
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.6, mfc=채움, mec=색[팔], ecolor=색[팔],
                            elinewidth=0.7, capsize=1.5, zorder=3)
            ax.annotate("", xy=(x1, r["After fine-tuning, MAE g/dL"]), xytext=(x0, r["Before fine-tuning, MAE g/dL"]),
                        arrowprops=dict(arrowstyle="-|>", color=색[팔], lw=0.9, shrinkA=2.5, shrinkB=2.5), zorder=2)
    ax.set_xticks(range(len(순서))); ax.set_xticklabels(["MobileNetV3\n-Large", "EfficientNet\n-B0", "MobileNetV2", "MobileViT\n-XXS", "EdgeNeXt\n-XS"], fontsize=6.5)
    ax.text(-0.02, 1.02, 글, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(0.5, 1.02, {"Indian location": "India", "Italian location": "Italy"}[site], transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
axs[0].set_ylabel("Mean absolute error, g/dL (median, 95% CI)")
from matplotlib.lines import Line2D
leg = [Line2D([], [], color=색["No safeguards"], marker="o", ms=3.6, lw=0.9, label="Fine-tuning without safeguards"),
       Line2D([], [], color=색["With safeguards"], marker="o", ms=3.6, lw=0.9, label="Fine-tuning with S1+S2+S3"),
       Line2D([], [], color="k", marker="o", mfc="white", ms=3.6, lw=0, label="Before fine-tuning (open)"),
       Line2D([], [], color="k", marker="o", ms=3.6, lw=0, label="After fine-tuning (filled)")]
fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("pdf", "svg", "png"):
    fig.savefig(본문 / f"Figure3_calibration_safeguards_10seed.{ext}", dpi=300, bbox_inches="tight")

# 11-10 : 자료 파일의 숫자는 가운뎃점 글자로 (검사기 · 보통 소수점 0)
T글 = T.copy()
for c in [c for c in T.columns if "MAE" in c or "CI" in c]: T글[c] = T[c].map(lambda x: 점(float(x), 2))
엑셀("Figure3_calibration_safeguards_10seed", 본문, {"Figure 3": T글},
    제목="Figure 3 data. External error before and after local fine-tuning, without safeguards and with S1 + S2 + S3, in five backbones",
    주석=[
      "Values are copied from table 3 (MobileNetV3-Large) and Table ST3 (other backbones); nothing is recomputed.",
      "No safeguards: the model without safeguards before and after fine-tuning. With safeguards: the same model with S1 + S2 before fine-tuning,",
      "  and S1 + S2 + S3 after fine-tuning (S1 + S2 for EdgeNeXt-XS, which has no batch normalisation).",
      "MAE: median across 10 seeds, with 95% CI from a paired participant bootstrap (100,000 resamples); predictions of the 10 fold models",
      "  were averaged after fine-tuning. p values are Holm-adjusted within the backbone over the before-versus-after comparisons.",
    ])
print(f"  ✅ {본문}/Figure3_calibration_safeguards_10seed.{{pdf,svg,png,xlsx}}")
