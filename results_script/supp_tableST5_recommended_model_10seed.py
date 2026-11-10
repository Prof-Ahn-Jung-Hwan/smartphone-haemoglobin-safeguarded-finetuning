# -*- coding: utf-8 -*-
"""**Table ST5. 권유 모델 요약** (11-08 : 앙상블 기울기 두 줄 더함 · 11-04 새로 · 지침 §5-1 표 형식 · §7 기준선 · 맨 위 칸의 평균 치우침 · 모델 1개 한 줄)

칸   개발 · 기관마다 : 권유 모델 · 사진 없는 기준선 · 모델 1개 (앙상블 아님) · 미세조정 전 · 평균 치우침 (실측 − 예측) 전 · 후
     권유 모델은 `_공통.권유` — §6-2 규칙 결과로 확정한다.
1차 자료  권유 모델의 train · external/{zero,few_bn} · main/results/external/splits
내는 것   results_table_figure/supplementary/TableST5_recommended_model_10seed.xlsx
쓰는 법   python supp_tableST5_recommended_model_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
from _공통 import _청크

b, a, 폴더 = 권유; 이름 = f"{표시[b]}, {구성글(a, 폴더)}"
시작(f"Table ST5 — 권유 모델 요약 ({이름})")


def 줄(곳, 무엇, M, 비교=None):
    m, r1, r2, lo, hi = 요약(M); x = 짝(비교, M) if 비교 is not None else None
    출 = {"Setting": f"{곳}: {무엇}", "Median (seed range), g/dL": f"{점(m)} ({구간(r1, r2)})", "95% CI": 구간(lo, hi),
         "Difference from the model that used no photograph, g/dL (95% CI)": 차이글(x) if x else "··", "p value": p표(x[3]) if x else "··"}
    print(f"  {곳:18} {무엇:58} {출['Median (seed range), g/dL']} [{구간(lo, hi)}]" + (f" · 차이 {차이글(x)} p {p표(x[3])}" if x else ""), flush=True)
    return 출


def 기울기(D):
    """앙상블 기울기 (실측 ~ 예측) — seed 마다 참여자 사이 회귀 기울기 · seed 중간값 (범위) · CI = 참여자 재표집 (MAE 와 같은 재표집)."""
    if D is None: return None
    P = D.pivot(index="key", columns="seed", values="p").sort_index(); y = D.groupby("key").y.first().loc[P.index].to_numpy(float)
    P = P.to_numpy(float); n = len(y)
    v = [np.polyfit(P[:, j], y, 1)[0] for j in range(P.shape[1])]
    W = 재표집(n); 중 = []
    for i in range(0, len(W), _청크):
        w = W[i:i + _청크].astype(float) / n
        mp, my = w @ P, w @ y; cov = w @ (P * y[:, None]) - mp * my[:, None]; var = w @ (P ** 2) - mp ** 2
        중.append(np.median(cov / var, axis=1))
    중 = np.concatenate(중)
    return float(np.median(v)), float(min(v)), float(max(v)), float(np.percentile(중, 2.5)), float(np.percentile(중, 97.5))


def 기울기줄(곳, 무엇, D):
    r = 필수(기울기(D), f"{곳} {무엇}")
    if r is None: return None
    m, r1, r2, lo, hi = r
    print(f"  {곳:18} {무엇:58} {점(m)} ({구간(r1, r2)}) [{구간(lo, hi)}]", flush=True)
    return {"Setting": f"{곳}: {무엇}", "Median (seed range), g/dL": f"{점(m)} ({구간(r1, r2)})", "95% CI": 구간(lo, hi),
            "Difference from the model that used no photograph, g/dL (95% CI)": "··", "p value": "··"}


행 = [줄("Development cohort", "MAE, cross-validation (762 participants)", 내부교차(b, a))]
for s in SITES:
    이 = 영문사이트[s]; 기 = 기준선(s)
    행 += [줄(이, "MAE, recommended model", 외부앙상블(b, a, 폴더, s), 기),
          줄(이, "MAE, photo-free baseline (mean haemoglobin of the 30 local participants)", 기),
          줄(이, "MAE, individual fold models without averaging (not an ensemble)", 외부앙상블(b, a, 폴더, s, 값="e1"), 기),
          줄(이, "MAE, before fine-tuning", 무보정앙상블(b, a, s), 기),
          줄(이, "mean error (measured minus predicted), before fine-tuning", 무보정앙상블(b, a, s, 값="d")),
          줄(이, "mean error (measured minus predicted), after fine-tuning", 외부앙상블(b, a, 폴더, s, 값="d")),
          기울기줄(이, "calibration slope (measured on predicted), before fine-tuning", 무보정값(b, a, s)),
          기울기줄(이, "calibration slope (measured on predicted), after fine-tuning", 외부값(b, a, 폴더, s))]
행 = [x for x in 행 if x is not None]
T, 머리 = 소제목표(pd.DataFrame(행), 첫열="Setting")
T = T.rename(columns={"Difference from the model that used no photograph, g/dL (95% CI)": "Baseline minus this row, g/dL (95% CI)"})
엑셀("TableST5_recommended_model_10seed", 부록, {"ST5": T}, 소제목=머리,
    제목=f"Table ST5. Performance of the recommended model ({이름}) against a photo-free baseline",
    주석=보고각주 + [
      "Photo-free baseline: the mean haemoglobin of the same 30 fine-tuning participants predicted for every evaluation participant of",
      "  that draw. Baseline minus this row: positive values mean the row has a lower error than the baseline.",
      "Single fold model: absolute errors of each fold model fine-tuned once, without averaging predictions between fold models.",
      "Mean error: laboratory minus predicted haemoglobin; 0 means no average over- or under-estimation.",
      "Calibration slope: slope of laboratory on predicted haemoglobin across participants, for the same ensemble prediction as the MAE",
      "  (computed for each seed; median and range across seeds; 95% CI from the same participant bootstrap). 1 means that the spread of",
      "  predictions matches that of laboratory values; above 1 means predictions are compressed towards the mean. Values have no unit.",
      "Before fine-tuning and the baseline are compared in the same participants: each participant's baseline error is averaged over the",
      "  draws in which that participant was evaluated, and the 10 draws together include every participant.",
      "Results are reported by location and are not pooled.",
    ])
