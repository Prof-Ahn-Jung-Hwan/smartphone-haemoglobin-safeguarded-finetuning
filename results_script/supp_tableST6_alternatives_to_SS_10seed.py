# -*- coding: utf-8 -*-
"""**Table ST6. 미세조정 학습률 민감도 · 출력만 재보정** (★ 11-07 : 기준 = 권유 구성 · 본분석 학습률 1e-4 · 옛 판 = 백업 · _old_261030/)

칸   ① 여덟 구성 × 다른 학습률 (2e-5 : E08 · 1e-3 : E12 · E12b · E12c) · S1 + S2 + S3 의 3e-4 · 3e-3 (E12b)
     ② 출력만 재보정 (recalibration) : 없음 · 절편 · 직선 — 10겹 모델 예측 평균에 대해 30명으로 맞춘다
1차 자료  main/results/{stage5,기전,M빼기LN,최소구성}/external/{zero,few,few_lr<개발 lr>,few_bn} · external/splits · grid/lr_선택.csv
내는 것   results_table_figure/supplementary/TableST6_alternatives_to_SS_10seed.xlsx
쓰는 법   python supp_tableST6_alternatives_to_SS_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

SP = 결과 / "external/splits"
_선 = pd.read_csv(결과 / "grid/lr_선택.csv", encoding="utf-8-sig")
개발lr = float(_선[(_선.백본 == "mnl") & (_선.경로 == "A")].lr.iloc[0]); 개발폴더 = f"few_lr{개발lr:g}"
def _지수(x):
    m, e = f"{x:.0e}".split("e"); return f"{int(m)} × 10" + str(int(e)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
개발lr글 = _지수(개발lr)


def 출력보정(site):
    """방식마다 참여자 × seed 표 — seed 의 10겹 예측 평균으로 30명에서 맞추고, 그 추첨의 평가 참여자에 쓴다."""
    z = 무보정값("mnl", 권유[1], site); 방 = {"none": {}, "intercept": {}, "linear": {}}
    for s in S10:
        Z = z[z.seed == s].set_index("key"); 급 = {k: [] for k in 방}
        for r in range(10):
            cal = Z.loc[pd.read_csv(SP / site / f"rep{r}" / "calib_random.csv").key[:30]]; ev = Z.loc[pd.read_csv(SP / site / f"rep{r}" / "eval.csv").key]
            b1, b0 = np.polyfit(cal.p, cal.y, 1)
            for k, p in [("none", ev.p), ("intercept", ev.p + (cal.y - cal.p).mean()), ("linear", b0 + b1 * ev.p)]: 급[k].append((p - ev.y).abs())
        for k in 방: 방[k][s] = pd.concat(급[k]).groupby(level=0).mean()
    return {k: pd.DataFrame(v).sort_index() for k, v in 방.items()}


시작("Table ST6 — 학습률 민감도 · 출력만 재보정")
# ★ 11-07 : 기준 = 권유 구성 · 본분석 학습률 (`_공통`). 줄 = 여덟 구성 × 다른 학습률 (2e-5 · 1e-4 · 개발 학습률 중 본분석이 아닌 것)
#           + S1 + S2 + S3 의 학습률 곡선 나머지 (3e-4 · 3e-3 · E12b) + 출력만 재보정 셋. p 는 기준과 짝지은 양측 (보정 전 · 옛 판과 같다)
기b, 기a, 기폴 = 권유; 기준 = {s: 외부앙상블(기b, 기a, 기폴, s) for s in SITES}; 출 = {s: 출력보정(s) for s in SITES}
줄 = [(f"Reference: {구성글(기a, 기폴)}, fine-tuning learning rate {학습률글} (main analysis)", lambda s: 기준[s])]
학습률들 = [(L, ("few", "few_bn") if float(L) == 2e-5 else (f"few_lr{float(L):g}", f"few_lr{float(L):g}_bn")) for L in ("2e-5", "1e-4", f"{개발lr:g}")]
for L, (갱, 고) in 학습률들:
    if float(L) == float(보정lr): continue
    표L = _지수(float(L)) + (" (development learning rate)" if float(L) == 개발lr else "")
    for a in ("mse", "normonly", "veminimalnoln", "veminimal"):
        for 폴더 in (갱, 고):
            줄.append((f"Other learning rate {표L}: {구성글(a, 폴더)}", lambda s, a=a, 폴더=폴더: 외부앙상블("mnl", a, 폴더, s)))
for L in ("3e-4", "3e-3"):
    줄.append((f"Other learning rate {_지수(float(L))}: S1 + S2 + S3", lambda s, L=L: 외부앙상블("mnl", "veminimal", f"few_lr{float(L):g}_bn", s)))
for k, 이름 in [("none", "no recalibration (model before fine-tuning)"), ("intercept", "shift by the average error of the 30 participants"),
               ("linear", "straight-line recalibration fitted in the 30 participants")]:
    줄.append((f"Recalibrating only the output, without retraining: {이름}", lambda s, k=k: 출[s][k]))
행 = []
for 이름, 값 in 줄:
    r = {"Method": 이름}
    for s in SITES:
        v = 값(s); r[f"{영문사이트[s]}: {칸글}"] = 칸값(v)
        r[f"{영문사이트[s]}: p value vs reference"] = "··" if 이름.startswith("Reference") or v is None else p표(짝(v, 기준[s])[3])
    행.append(r); print(f"  {이름[:80]:80} " + " | ".join(str(v) for c, v in r.items() if ":" in c), flush=True)
T, 머리 = 소제목표(pd.DataFrame(행), 첫열="Method")
엑셀("TableST6_alternatives_to_SS_10seed", 부록, {"ST6": T}, 소제목=머리,
    제목=("Table ST6. Sensitivity to the fine-tuning learning rate and recalibration of the output only, using the same 30 local "
         "participants (MobileNetV3-Large)"),
    주석=보고각주 + [
      "Every row uses the same 30 participants and is evaluated in the same other participants; p values compare each row with the reference.",
      f"Reference: the recommended configuration fine-tuned at the learning rate of the main analysis ({학습률글.replace('×', 'x')}).",
      f"Other learning rates: every configuration at the other learning rates, including the development learning rate ({개발lr글.replace('×', 'x')});",
      "  S1 + S2 + S3 was also fine-tuned at 3 x 10⁻⁴ and 3 x 10⁻³.",
      "Recalibrating only the output: predictions of the 10 fold models before fine-tuning are averaged; shift moves every prediction by the",
      "  average error of the 30 participants; straight-line recalibration fits laboratory haemoglobin = a + b x predicted haemoglobin in",
      "  the 30 participants and applies it to the others.",
      "S1, final-feature layer normalisation; S2, adaptation layer; S3, backbone batch-normalisation statistics held during fine-tuning",
      "  (otherwise they are updated). Results are reported by location and are not pooled.",
    ])
