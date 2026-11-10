# -*- coding: utf-8 -*-
"""**Table ST10. 모든 조합의 성적 · 권유 규칙** (11-04 : 1 seed 10겹 앙상블 · 중간값 (범위) · 규칙 = 지침 §6-2 · 옛 판 _old_261030/)

규칙 (§6-2)  S1 × S2 × S3 의 36조합에서 ① 미세조정 뒤 외부 MAE ② 두 기관 모두 가장 낮은 MAE 와 차이 없음 (p ≥ 0·05)
             ③ 그중 GMACs 가 가장 낮은 백본 ④ 그 안에서 더 나쁜 기관의 MAE (소수 둘째 자리) 가 가장 낮은 구성.
             같은 규칙을 한 기관에만 적용한 결과도 함께 낸다 (민감도 · 결과와 상관없이 · 싣는지는 단어 수를 보고 정한다).
내는 것   results_table_figure/supplementary/TableST10_ranking_10seed.xlsx · main_results/S20_권유규칙.csv
쓰는 법   python supp_tableST10_ranking_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

# ★ 11-10 : 보조 손실(S4 · S4n · S5 · S4+S5)을 논문에서 뺐다 → 표에도 S1 × S2 × S3 조합만 둔다 (옛 판 = _previous_results_1110/)
구성들 = ["mse", "normonly", "veminimalnoln", "veminimal"]; 핵심 = 구성들[:4]
GM = 읽기("P3_추론비용").set_index("백본")
시작("Table ST10 — 모든 조합 · 권유 규칙")
조합 = [(b, a, 폴, 절) for b in BB for a in 구성들 for 폴, 절 in [(FT, "BN updated"), (FTB, "BN frozen (S3)")] if not (b == "enxs" and 폴 == FTB)]
내 = {(b, a): 내부교차(b, a) for b, a, _, _ in 조합}
외 = {s: {(b, a, 폴): 외부앙상블(b, a, 폴, s) for b, a, 폴, _ in 조합} for s in SITES}
중 = lambda M: float(np.median(M.to_numpy(float).mean(0)))


def 일등대비(D):
    후 = {k: v for k, v in D.items() if v is not None and k[1] != "anchorshuf"}; 일 = min(후, key=lambda k: 중(후[k]))
    return 일, {k: (np.nan if k == 일 or v is None else 짝(v, D[일])[3]) for k, v in D.items()}


내일, 내p = 일등대비(내); 외결 = {s: 일등대비(외[s]) for s in SITES}
행 = []
for b, a, 폴, 절 in 조합:
    if any(외[s][(b, a, 폴)] is None for s in SITES): continue
    r = {"Backbone": 표시[b], "Configuration": 구성이름[a], "Fine-tuning": 절, "Inference cost, GMACs": 점(float(GM.loc[표시[b], "GMACs"]), 3),
         f"Development: {칸글}": 칸값(내[(b, a)]), "Development: p vs best": "best" if (b, a) == 내일 else p표(내p[(b, a)])}
    v = {}
    for s in SITES:
        이 = 영문사이트[s].replace(" location", ""); 일, pp = 외결[s]; v[s] = 중(외[s][(b, a, 폴)])
        r[f"{이}: {칸글}"] = 칸값(외[s][(b, a, 폴)]); r[f"{이}: p vs best"] = "best" if (b, a, 폴) == 일 else p표(pp[(b, a, 폴)])
    r["Worse location, MAE, g/dL"] = 점(max(v.values())); r["Difference between locations, g/dL"] = 점(abs(v["India"] - v["Italy"]))
    행.append(r)
T = pd.DataFrame(행); print(T.to_string(index=False), flush=True)


def 규칙(곳들):
    칸들 = [(b, a, 폴) for b, a, 폴, _ in 조합 if a in 핵심 and all(외[s][(b, a, 폴)] is not None for s in SITES)]
    후보 = set(칸들)
    for s in 곳들:
        D = {k: 외[s][k] for k in 칸들}; 일 = min(D, key=lambda k: round(중(D[k]), 2))
        후보 &= {k for k in 칸들 if k == 일 or round(중(D[k]), 2) == round(중(D[일]), 2) or 짝(D[k], D[일])[3] >= 0.05}
    낮은 = min(float(GM.loc[표시[k[0]], "GMACs"]) for k in 후보)
    남 = [k for k in 후보 if float(GM.loc[표시[k[0]], "GMACs"]) == 낮은]
    끝 = min(남, key=lambda k: (round(max(중(외[s][k]) for s in 곳들), 2), sum(중(외[s][k]) for s in 곳들)))
    return 끝, 후보


이름 = lambda k: f"{표시[k[0]]}, {구성글(k[1], k[2])}"
결 = {"both": 규칙(SITES), "India": 규칙(["India"]), "Italy": 규칙(["Italy"])}
pd.DataFrame([{"규칙": g, "고른것": 이름(v[0]), "후보수": len(v[1]), "후보": " | ".join(sorted(이름(k) for k in v[1]))} for g, v in 결.items()]).to_csv(MR / f"S20_권유규칙{판꼬리}.csv", index=False, encoding="utf-8-sig")
for g, v in 결.items(): print(f"  규칙 {g:6} → {이름(v[0])} · 후보 {len(v[1])}개", flush=True)
if 결["both"][0] != 권유: print(f"  🔴 규칙 결과 {결['both'][0]} 가 `_공통.권유` {권유} 와 다르다 — ST5 · ST8 을 확인할 것")
엑셀("TableST10_ranking_10seed", 부록, {"ST10": T},
    제목="Table ST10. Error of every combination of backbone, configuration and fine-tuning procedure in the development cohort and at each external location",
    주석=보고각주 + [
      "p vs best: two-sided, against the combination with the smallest median error in that setting.",
      "  Development values do not depend on the fine-tuning procedure and are repeated in both rows.",
      "S1, final-feature layer normalisation; S2, adaptation layer; S3, backbone batch-normalisation statistics frozen during fine-tuning.",
      "  EdgeNeXt-XS contains no batch normalisation.",
      "Recommendation rule (36 combinations of S1 x S2 x S3): not different from the lowest error at both locations (p >= 0·05), then the lowest",
      f"  inference cost, then the lowest error at the worse location (second decimal) selected {이름(결['both'][0])}. The same rule applied at one",
      f"  location alone selected {이름(결['India'][0])} (Indian location) and {이름(결['Italy'][0])} (Italian location).",
      "Results are reported by location and are not pooled.",
    ])
