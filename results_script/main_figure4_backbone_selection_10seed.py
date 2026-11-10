# -*- coding: utf-8 -*-
"""**Figure 4. 백본 열 개에서 권유 모델 하나까지** (11-10 새로 만듦)

무엇  A  백본 열 개의 개발 오차 (안전장치 없음) · 다음 단계로 넘긴 다섯 개를 색으로
      B  (11-10 사용자 요청 : 옛 Figure 3 를 그대로) 인도 · 이탈리아 두 칸 · 다섯 백본마다 미세조정 전(빈 점) → 뒤(채운 점) 화살표,
         안전장치 없음 (회색) vs 있음 (파랑, S1+S2+S3 · EdgeNeXt-XS 는 S1+S2). 값은 main/Figure3_calibration_safeguards_10seed.xlsx 에서 읽는다
      C  36개 조합 : 가로 = 추론 비용 (GMACs) · 세로 = 두 기관 중 나쁜 쪽 MAE · 후보 (두 기관 모두 p ≥ 0·05) 강조 · 권유 모델 별표
      새로 계산하지 않는다 — 표 ST2 · ST10 의 칸을 그대로 읽는다.
1차 자료  results_table_figure_lr1e-4/supplementary/TableST2_backbone_screen.xlsx · TableST10_ranking_10seed.xlsx · main/Figure3_calibration_safeguards_10seed.xlsx
내는 것   results_table_figure_lr1e-4/main/Figure4_backbone_selection_10seed.{pdf,svg,png,xlsx}  (옛 Figure 3 는 건드리지 않는다)
쓰는 법   (프로젝트 뿌리에서) python -u paperwork/results_script/main_figure4_backbone_selection_10seed.py
"""
import sys, re; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

이름 = "Figure4_backbone_selection_10seed"
for f in [본문 / f"{이름}.{e}" for e in ("pdf", "svg", "png", "xlsx")]:
    if f.exists(): sys.exit(f"🔴 {f.name} 가 이미 있다 — 덮어쓰지 않는다. 옛 판을 _previous_results_<날짜>/main 으로 먼저 옮긴다")

다섯 = ["MobileNetV3-Large", "EfficientNet-B0", "MobileNetV2", "MobileViT-XXS", "EdgeNeXt-XS"]
짧 = {"MobileNetV3-L": "MobileNetV3-Large"}
색 = {"none": "#8c8c8c", "with": "#1f5fa8", "cand": "#e07b39", "other": "#c8c8c8"}


def 수(s):   # "1·45 (1·40–1·49) [1·23–1·67]" 또는 "1·22 (1·10–1·35)" → 중간값, 아래, 위
    s = str(s).strip()
    m = re.match(r"([\d·]+) \([\d·]+–[\d·]+\) \[([\d·]+)–([\d·]+)\]", s) or re.match(r"([\d·]+) \(([\d·]+)–([\d·]+)\)", s)
    if not m: sys.exit(f"🔴 MAE 칸을 읽지 못했다 : {s!r}")
    return tuple(float(x.replace("·", ".")) for x in m.groups())


def p수(s):  # "best" → 1 · "<0·0001" → 0
    s = str(s).strip()
    if s == "best": return 1.0
    if s.startswith("<"): return 0.0
    return float(s.replace("·", "."))


def 표(f):
    if not f.exists(): sys.exit(f"🔴 {f.name} 가 없다 — 표 스크립트를 먼저 돌린다")
    raw = pd.read_excel(f, header=None)
    머 = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.contains("Backbone").any())
    d = pd.read_excel(f, header=머)
    끝 = d.index[d.iloc[:, 0].isna()]
    return d.loc[: 끝[0] - 1] if len(끝) else d


시작("Figure 4 — 백본 선정 (ST2 · ST10 에서 읽음)")

# ── A : 백본 열 개 ────────────────────────────────────────────
S2 = 표(부록 / "TableST2_backbone_screen.xlsx")
A = pd.DataFrame({"Backbone": S2["Backbone"].str.replace(" (reference)", "", regex=False),
                  "p vs reference": S2["p value vs reference"].astype(str)})
A[["MAE", "lo", "hi"]] = [수(s) for s in S2["Validation MAE, g/dL (95% CI)"]]
A["Carried forward"] = A.Backbone.isin(다섯)
if A["Carried forward"].sum() != 5: sys.exit("🔴 다섯 백본을 ST2 에서 찾지 못했다")

# ── B · C : 36 조합 ──────────────────────────────────────────
T = 표(부록 / "TableST10_ranking_10seed.xlsx")
if len(T) != 36: sys.exit(f"🔴 ST10 조합이 {len(T)}개 (36개여야 한다)")
C = pd.DataFrame({"Backbone": T.Backbone.replace(짧), "Configuration": T.Configuration, "Fine-tuning": T["Fine-tuning"],
                  "GMACs": T["Inference cost, GMACs"].astype(str).str.replace("·", ".").astype(float)})
for site, 열 in [("India", "Indian"), ("Italy", "Italian")]:
    C[[f"{site} MAE", f"{site} lo", f"{site} hi"]] = [수(s) for s in T[f"{열}: MAE, g/dL, median (seed range) [95% CI]"]]
    C[f"{site} p vs best"] = [p수(s) for s in T[f"{열}: p vs best"]]
C["Worse-site MAE"] = C[["India MAE", "Italy MAE"]].max(axis=1)
C["Label"] = [c if f == "BN updated" else ("S3" if c == "No safeguards" else f"{c}+S3") for c, f in zip(C.Configuration, C["Fine-tuning"])]
C["Candidate"] = (C["India p vs best"] >= 0.05) & (C["Italy p vs best"] >= 0.05)
후보 = C[C.Candidate].copy()
가벼움 = 후보.GMACs.min(); 남 = 후보[후보.GMACs == 가벼움]; 고름 = 남["Worse-site MAE"].idxmin()
C["Recommended"] = C.index == 고름
print(f"  후보 {len(후보)}개 :"); print(후보[["Backbone", "Label", "GMACs", "India MAE", "Italy MAE", "Worse-site MAE"]].to_string(index=False))
r = C.loc[고름]; print(f"  권유 : {r.Backbone} {r.Label} (나쁜 쪽 {r['Worse-site MAE']:.2f})")
if not (r.Backbone == "MobileNetV3-Large" and r.Label == "S1+S2+S3"): sys.exit("🔴 권유 모델이 원고와 다르다")

# B — 옛 Figure 3 의 자료 (Table 3 · ST3 에서 옮긴 값 · 새로 계산하지 않음)
B = 표(본문 / "Figure3_calibration_safeguards_10seed.xlsx")
B = B.rename(columns={"Before fine-tuning, MAE g/dL": "Before", "Before 95% CI low": "Before lo", "Before 95% CI high": "Before hi",
                      "After fine-tuning, MAE g/dL": "After", "After 95% CI low": "After lo", "After 95% CI high": "After hi"})
B["Backbone"] = B.Backbone.replace(짧); B["Location"] = B.Location.map({"Indian location": "India", "Italian location": "Italy"})
for c in ["Before", "Before lo", "Before hi", "After", "After lo", "After hi"]: B[c] = B[c].astype(str).str.replace("·", ".").astype(float)   # 11-10 : 가운뎃점 글자도 읽는다
if len(B) != 20 or set(B.Backbone) != set(다섯): sys.exit("🔴 옛 Figure 3 자료가 20행 · 다섯 백본이 아니다")
# 미세조정 뒤 값이 ST10 과 같은지 대조
for _, r in B.iterrows():
    lab = "No safeguards" if r.Arm == "No safeguards" else r["Configuration after fine-tuning"]
    c = C[(C.Backbone == r.Backbone) & (C.Label == lab)][f"{r.Location} MAE"].iloc[0]
    if abs(c - r.After) > 0.005: sys.exit(f"🔴 {r.Backbone} {r.Location} {lab} : Figure 3 {r.After} ≠ ST10 {c}")
print("  ✅ 패널 B 의 미세조정 뒤 값 20개가 ST10 과 같다")

# ── 그림 ──────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
fig = plt.figure(figsize=(7.1, 6.9))
gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.45], height_ratios=[1, 1.1], hspace=0.7, wspace=0.42)
gB = gs[0, 1].subgridspec(1, 2, wspace=0.06)
axA, axB1, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gB[0, 0]), fig.add_subplot(gs[1, :])
axB2 = fig.add_subplot(gB[0, 1], sharey=axB1)
점 = lambda v: f"{v:.2f}".replace(".", "·")


def 글자(ax, k, t):
    ax.text(-0.02, 1.05, k, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    ax.text(0.0, 1.05, " " + t, transform=ax.transAxes, fontsize=7.5, va="bottom")


# A — 백본 열 개
a = A.iloc[::-1].reset_index(drop=True)
for i, x in a.iterrows():
    c = 색["with"] if x["Carried forward"] else 색["other"]
    axA.errorbar(x.MAE, i, xerr=[[x.MAE - x.lo], [x.hi - x.MAE]], fmt="o", ms=3.4, color=c, elinewidth=0.8, capsize=1.5)
    axA.text(1.555, i, x["p vs reference"].replace("··", "ref"), va="center", ha="right", fontsize=6, color="#555555")
axA.text(1.555, len(a) - 0.35, "p", ha="right", fontsize=6.3, color="#555555", style="italic")
axA.set_yticks(range(len(a))); axA.set_yticklabels(a.Backbone, fontsize=6.3)
for t, cf in zip(axA.get_yticklabels(), a["Carried forward"]): t.set_color("black" if cf else "#888888"); t.set_fontweight("bold" if cf else "normal")
axA.set_xlim(1.05, 1.56); axA.set_xticks([1.1, 1.2, 1.3, 1.4]); axA.set_ylim(-0.6, len(a) - 0.1)
axA.set_xlabel("Development MAE, g/dL (95% CI)")
axA.legend(handles=[Line2D([], [], marker="o", color=색["with"], ms=3.4, lw=0.8, label="Carried forward"),
                    Line2D([], [], marker="o", color=색["other"], ms=3.4, lw=0.8, label="Not carried forward")],
           loc="lower left", bbox_to_anchor=(-0.25, -0.42), ncol=2, frameon=False, fontsize=6, handletextpad=0.3)
글자(axA, "A", "Ten backbones (development)")

# B — 다섯 백본 · 미세조정 전 → 뒤 · 안전장치 없음 vs 있음 (옛 Figure 3)
팔색 = {"No safeguards": 색["none"], "With safeguards": 색["with"]}
for ax, site in [(axB1, "India"), (axB2, "Italy")]:
    g = B[B.Location == site]
    for i, b in enumerate(다섯):
        for 팔, dx in [("No safeguards", -0.17), ("With safeguards", 0.17)]:
            r = g[(g.Backbone == b) & (g.Arm == 팔)].iloc[0]; x0, x1 = i + dx - 0.08, i + dx + 0.08
            for x, m, lo, hi, 채움 in [(x0, r.Before, r["Before lo"], r["Before hi"], "white"), (x1, r.After, r["After lo"], r["After hi"], 팔색[팔])]:
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.2, mfc=채움, mec=팔색[팔], ecolor=팔색[팔], elinewidth=0.6, capsize=1.2, zorder=3)
            ax.annotate("", xy=(x1, r.After), xytext=(x0, r.Before),
                        arrowprops=dict(arrowstyle="-|>", color=팔색[팔], lw=0.8, shrinkA=2.2, shrinkB=2.2, mutation_scale=6), zorder=2)
    ax.set_xticks(range(5)); ax.set_xticklabels(다섯, fontsize=5.8, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, 4.5)
    ax.text(0.5, 0.98, site, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, fontweight="bold")
axB1.set_ylabel("External MAE, g/dL (95% CI)")
plt.setp(axB2.get_yticklabels(), visible=False)
axB1.legend(handles=[Line2D([], [], color=색["none"], marker="o", ms=3.2, lw=0.8, label="Without safeguards"),
                     Line2D([], [], color=색["with"], marker="o", ms=3.2, lw=0.8, label="With S1+S2+S3*"),
                     Line2D([], [], color="k", marker="o", mfc="white", ms=3.2, lw=0, label="Before fine-tuning"),
                     Line2D([], [], color="k", marker="o", ms=3.2, lw=0, label="After fine-tuning")],
            loc="upper center", bbox_to_anchor=(1.03, -0.36), ncol=4, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.9)
axB1.text(0.0, -0.5, "*S1+S2 in EdgeNeXt-XS (no batch normalisation); before fine-tuning, the S1+S2 model", transform=axB1.transAxes, fontsize=5.6, color="#555555")
글자(axB1, "B", "Five backbones, before and after local fine-tuning")

# C — 36 조합 (백본을 추론 비용 순서로, 백본 안에서는 구성 순서로)
순구성 = ["No safeguards", "S1", "S2", "S1+S2", "S3", "S1+S3", "S2+S3", "S1+S2+S3"]
백순 = C.groupby("Backbone").GMACs.first().sort_values().index.tolist()
yl = 2.0
for j, b in enumerate(백순):
    g = C[C.Backbone == b]; 있는 = [k for k in 순구성 if k in set(g.Label)]
    off = dict(zip(있는, np.linspace(-0.36, 0.36, len(있는)) if len(있는) > 1 else [0]))
    if j % 2 == 0: axC.axvspan(j - 0.48, j + 0.48, color="#f4f4f4", zorder=0, lw=0)
    for _, x in g.iterrows():
        gx = j + off[x.Label]; y = min(x["Worse-site MAE"], yl)
        if x.Recommended:
            axC.plot(gx, y, "*", ms=12, color=색["cand"], mec="black", mew=0.5, zorder=5)
            axC.annotate(f"{x.Label} (recommended)\nIndia {점(x['India MAE'])}, Italy {점(x['Italy MAE'])}", xy=(gx, y), xytext=(gx + 0.1, y - 0.09),
                         fontsize=6, ha="left", va="center", fontweight="bold", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))
        elif x.Candidate:
            axC.plot(gx, y, "o", ms=5.2, color=색["cand"], mec="black", mew=0.4, zorder=4)
            왼 = b == "MobileNetV3-Large"
            axC.annotate(f"{x.Label}\nIndia {점(x['India MAE'])}, Italy {점(x['Italy MAE'])}", xy=(gx, y), xytext=(gx + (0.0 if 왼 else 0.1), y - (0.13 if 왼 else 0.09)),
                         fontsize=6, ha="center" if 왼 else "left", va="center", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))
        else:
            axC.plot(gx, y, "o" if x["Worse-site MAE"] <= yl else "^", ms=3.8,
                     mfc="white" if x.Configuration == "No safeguards" else 색["other"], mec="#777777", mew=0.6, zorder=3, clip_on=False)
            if x["Worse-site MAE"] > yl:
                axC.text(gx + 0.07, yl, f"{x.Label}: {점(x['Worse-site MAE'])}", fontsize=6, color="#555555", va="center")
axC.set_xticks(range(len(백순)))
axC.set_xticklabels([f"{b}\n{C[C.Backbone == b].GMACs.iloc[0]:.3f} GMACs".replace(".", "·") for b in 백순], fontsize=6.3)
axC.set_xlim(-0.5, len(백순) - 0.5); axC.set_ylim(1.2, yl + 0.03)
axC.set_xlabel("Backbone, in order of inference cost; points within a backbone are its configurations")
axC.set_ylabel("MAE at the less favourable site, g/dL")
axC.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#777777", ms=3.8, label="No safeguards"),
                    Line2D([], [], marker="o", ls="", color=색["other"], mec="#777777", ms=3.8, label="With safeguards, not a candidate"),
                    Line2D([], [], marker="o", ls="", color=색["cand"], mec="black", ms=5.2, label="Candidate: not different from the smallest error at both locations (p≥0·05)"),
                    Line2D([], [], marker="*", ls="", color=색["cand"], mec="black", ms=10, label="Recommended: smallest inference cost, then smaller error at the less favourable site")],
           loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.2)
글자(axC, "C", "36 backbone–configuration combinations, each fine-tuned 1000 times at each location")
for ax in (axA, axB1, axB2, axC):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x" if ax is axA else "y", lw=0.3, color="#e3e3e3"); ax.set_axisbelow(True)

for ext in ("pdf", "svg", "png"):
    fig.savefig(본문 / f"{이름}.{ext}", dpi=300, bbox_inches="tight")

def 칸(d, 판):
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = [("<0·0001" if (c.endswith("p vs best") and v == 0) else ("best" if c.endswith("p vs best") and v == 1 else
                     (f"{v:.3f}" if c == "GMACs" else (f"{v:.4g}" if "p vs" in c else f"{v:.2f}")).replace(".", "·"))) for v in d[c]]
        elif d[c].dtype.kind == "b":
            d[c] = d[c].map({True: c, False: "Not " + c[0].lower() + c[1:]})
    d.insert(0, "Panel", 판); return d


F = pd.concat([칸(A.rename(columns={"MAE": "MAE, g/dL", "lo": "95% CI low", "hi": "95% CI high"}), "A"),
               칸(B.rename(columns={"Before": "Before fine-tuning, MAE g/dL", "Before lo": "Before 95% CI low", "Before hi": "Before 95% CI high",
                                    "After": "After fine-tuning, MAE g/dL", "After lo": "After 95% CI low", "After hi": "After 95% CI high"}), "B"),
               칸(C, "C")], ignore_index=True).fillna("")
엑셀(이름, 본문, {"Figure 4": F},
    제목="Figure 4 data. From ten backbones to one recommended model",
    주석=[
      "Values are copied from Table ST2 (panel A), the Figure 3 data of table 3 and Table ST3 (panel B) and Table ST10 (panel C); nothing is recomputed.",
      "A: development MAE without safeguards (95% CI); p against the backbone with the smallest error. Five were carried forward.",
      "B: external MAE before and after fine-tuning, without safeguards and with S1 + S2 + S3 (S1 + S2 for EdgeNeXt-XS, which has no batch",
      "  normalisation); before fine-tuning, the safeguarded arm is the model with S1 + S2. p values are Holm-adjusted, after vs before.",
      "C: candidate = p >= 0·05 against the smallest median error at both locations; recommended = smallest inference cost among the candidates,",
      "  then the smaller error at the less favourable site. Each combination: 10 seeds x 10 fold models x 10 random draws per location.",
      "MAE: median across 10 seeds, with 95% CI from a paired participant bootstrap (100,000 resamples).",
    ])
print(f"  ✅ {본문}/{이름}.{{pdf,svg,png,xlsx}}")
