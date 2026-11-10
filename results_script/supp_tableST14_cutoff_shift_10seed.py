# -*- coding: utf-8 -*-
"""**Table ST14. 현지 문턱이 위로 옮겨진 이유 — 예측이 현지 평균 쪽으로 당겨진다** (11-10 새로 만듦)

무엇  권유 모델 (MobileNetV3-L · S1+S2+S3 · 자동 SCCA · 현지 30명) 의 외부 예측에서
      ① 빈혈 (Hb <12·0) · 비빈혈 참여자의 검사실 Hb 평균과 예측값 평균
      ② 예측값을 검사실 Hb 에 대해 직선으로 맞춘 기울기 (1 이면 당겨짐 없음)
      ③ 빈혈인데 예측값이 12·0 보다 높은 비율
      ④ 현지 문턱 (Hb <12·0 · 표 5 와 같은 규칙과 자료) 의 평균
      ⑤ 현지 문턱에서 빈혈로 분류되었지만 빈혈이 아닌 참여자의 검사실 Hb 구간 (12·0–12·9 · 13·0–13·9 · ≥14·0)
      — Discussion 의 "anaemic participants in Italy had a mean laboratory haemoglobin of 10·1 g/dL and a mean predicted value of 12·5 g/dL" 의 출처.
값    seed 마다 계산 → 중간값 (seed 범위) [95% CI] · CI = 참여자 bootstrap 100,000 (재표집마다 seed 중간값)
      예측값 = 추첨마다 10겹 모델 평균 → 그 참여자가 평가된 추첨들의 평균 (`_공통.외부값`, 표 3 · 4 와 같음)
      ⑤ 는 표 5 와 같이 추첨마다 그 추첨의 현지 문턱으로 분류 → 참여자마다 양성 추첨 비율 q → seed 마다 기대 수
1차 자료  main/results/최소구성/external/few_lr0.0001_bn · main/results/임계/외부_안쪽예측/lr0.0001_bn
내는 것   results_table_figure_lr1e-4/supplementary/TableST14_cutoff_shift_10seed.xlsx  (있으면 멈춘다 — 덮어쓰지 않는다)
쓰는 법   (프로젝트 뿌리에서) python -u paperwork/results_script/supp_tableST14_cutoff_shift_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

출력 = 부록 / "TableST14_cutoff_shift_10seed.xlsx"
if 출력.exists(): sys.exit(f"🔴 {출력.name} 가 이미 있다 — 덮어쓰지 않는다. 옛 판을 _previous_results_<날짜>/supplementary 로 먼저 옮긴다")

src = (뿌리 / "main/analysis/260929_임계선정.py").read_text(encoding="utf-8")
ns = {"__name__": "임계선정", "__file__": str(뿌리 / "main/analysis/260929_임계선정.py")}
exec(compile(src[:src.index("ap = argparse.ArgumentParser()")], "260929_임계선정.py", "exec"), ns)
임계고르기 = ns["임계고르기"]

TH = 12.0
_, 권a, 권폴 = 권유
if (권a, 권폴) != ("veminimal", "few_lr0.0001_bn"): sys.exit(f"🔴 권유 구성이 {구성글(권a, 권폴)} · {권폴} — 이 표는 S1+S2+S3 · 1 × 10⁻⁴ 기준이다")
권뿌, 권틀 = 칸찾기("mnl", 권a); 안태그 = "lr0.0001_bn"
구간들 = [("12·0–12·9", 12.0, 13.0), ("13·0–13·9", 13.0, 14.0), ("≥14·0", 14.0, 99.0)]


def 분류(site, 키):
    """표 5 의 현지 문턱 · 추첨마다 → q (seed × 참여자 · 평가 안 된 칸 nan) · 문턱들."""
    위 = {k: i for i, k in enumerate(키)}; A = np.zeros((len(S10), len(키))); C = np.zeros_like(A); 문 = []
    for si, s in enumerate(S10):
        for r in range(10):
            ev = pd.concat([pd.read_csv(권뿌 / "external" / 권폴 / site / "n30" / f"rep{r}" / 권틀.format(s=s) / f"fold{f}.csv", usecols=["key", "pred"]) for f in range(CV)]).groupby("key").pred.mean()
            안 = pd.concat([pd.read_csv(결과 / "임계/외부_안쪽예측" / 안태그 / f"{권틀.format(s=s)}_fold{f}_{site}_rep{r}.csv") for f in range(CV)]).groupby("key").agg(y=("y", "first"), pred=("pred", "mean"))
            t = 임계고르기(안.y, 안.pred, TH)
            if np.isnan(t): continue                       # 그 추첨의 30명에 빈혈이 없으면 고를 수 없다 (표 5 와 같다)
            문.append(t); i = np.array([위[k] for k in ev.index]); A[si, i] += (ev.to_numpy() < t); C[si, i] += 1
    return np.where(C > 0, A / np.maximum(C, 1), np.nan), 문


def 칸(v, b, 자릿=2):
    return f"{점(float(np.median(v)), 자릿)} ({구간(float(v.min()), float(v.max()), 자릿)}) [{구간(*np.percentile(b, [2.5, 97.5]), 자릿)}]"


def 비율(num, den):
    """num, den : 참여자 × seed 가중치 행렬 → seed 값 · 재표집마다 seed 중간값."""
    W = 재표집(len(num)); v = num.sum(0) / den.sum(0); b = []
    for i in range(0, len(W), 2000):
        w = W[i:i + 2000].astype(float); b.append(np.nanmedian((w @ num) / (w @ den), axis=1))
    return v, np.concatenate(b)


def 기울기(Y, P):
    """참여자 × seed 의 검사실 Y · 예측 P → seed 마다 P 를 Y 에 대해 맞춘 기울기 · 재표집마다 seed 중간값."""
    v = np.array([np.polyfit(Y[:, j], P[:, j], 1)[0] for j in range(P.shape[1])]); W = 재표집(len(Y)); b = []
    for i in range(0, len(W), 2000):
        w = W[i:i + 2000].astype(float); n = w.sum(1, keepdims=True)
        mx = (w @ Y) / n; my = (w @ P) / n
        b.append(np.median(((w @ (Y * P)) / n - mx * my) / ((w @ (Y * Y)) / n - mx ** 2), axis=1))
    return v, np.concatenate(b)


시작("Table ST14 — 현지 문턱이 옮겨진 이유 (예측의 당겨짐)")
열 = {}
for site in SITES:
    P = 외부앙상블("mnl", 권a, 권폴, site, 값="p")                # 참여자 × seed
    키 = list(P.index); v = 외부값("mnl", 권a, 권폴, site)
    y = v[v.seed == 1].set_index("key").y.reindex(키).to_numpy(float); Y = np.repeat(y[:, None], len(S10), 1)
    Pm = P.to_numpy(float); 빈 = y < TH; 아님 = ~빈; 하나 = np.ones_like(Pm)
    q, 문 = 분류(site, 키); q0 = np.nan_to_num(q).T                # 참여자 × seed
    c = {}
    c["Participants, n"] = str(len(y)); c["Participants with haemoglobin <12·0 g/dL, n"] = str(int(빈.sum()))
    c["Laboratory haemoglobin, anaemic participants, mean, g/dL"] = 점(float(y[빈].mean()), 2)
    c["Predicted haemoglobin, anaemic participants, mean, g/dL"] = 칸값(Pm[빈], 2)
    c["Laboratory haemoglobin, participants without anaemia, mean, g/dL"] = 점(float(y[아님].mean()), 2)
    c["Predicted haemoglobin, participants without anaemia, mean, g/dL"] = 칸값(Pm[아님], 2)
    s_, sb = 기울기(Y, Pm); c["Slope of predicted on laboratory haemoglobin (1 = no shrinkage)"] = 칸(s_, sb, 2)
    a_, ab = 비율((Pm[빈] > TH).astype(float), 하나[빈]); c["Anaemic participants predicted above 12·0 g/dL, proportion"] = 칸(a_, ab, 3)
    c["Local cut-off for haemoglobin <12·0 g/dL, predicted haemoglobin, mean of draws, g/dL"] = 점(float(np.mean(문)), 1)
    fp = q0 * 아님[:, None]
    for 이름, lo, hi in 구간들:
        m = ((y >= lo) & (y < hi))[:, None]
        r_, rb = 비율(fp * m, fp); c[f"Classified anaemic at the local cut-off without anaemia: laboratory haemoglobin {이름} g/dL, proportion"] = 칸(r_, rb, 3)
    열[영문사이트[site]] = c
    print(f"  {site}: 빈혈 {빈.sum()} · 검사실 {y[빈].mean():.2f} · 예측 {np.median(Pm[빈].mean(0)):.2f} · 기울기 {np.median(s_):.2f} · 문턱 {np.mean(문):.2f} · 거짓양성 12–13 비율 {np.median(비율(fp * ((y >= 12) & (y < 13))[:, None], fp)[0]):.3f}", flush=True)

T = pd.DataFrame(열); T.index.name = "Measure"; T = T.reset_index()
엑셀("TableST14_cutoff_shift_10seed", 부록, {"ST14": T},
    제목="Table ST14. Why the local cut-off moved: predictions after fine-tuning were drawn towards the local mean",
    주석=[
      "Recommended model: MobileNetV3-Large with S1+S2+S3 and automated SCCA, fine-tuned with 30 local participants (learning rate 1 × 10⁻⁴).",
      "Values are median (range across 10 seeds) [95% CI]; CIs from 100 000 participant bootstrap resamples, taking the seed median in each resample.",
      "Predicted haemoglobin: for each draw, the 10 fold models were averaged for each participant, and then averaged over the draws in which the participant was evaluated.",
      "Slope: predicted haemoglobin fitted by a straight line on laboratory haemoglobin; a slope below 1 means that predictions are drawn towards the mean.",
      "Local cut-off: the same rule and data as table 5 (highest specificity among cut-offs with a sensitivity of at least 0·70 in the 30 fine-tuning participants).",
      "Last three rows: among participants classified as anaemic at the local cut-off of each draw but with laboratory haemoglobin ≥12·0 g/dL,",
      "  the proportion in each band of laboratory haemoglobin (expected counts over draws). WHO 2024 cut-offs: 12·0 g/dL in women, 13·0 g/dL in men; sex was not recorded.",
    ])
print(f"  ✅ {출력}")
