# -*- coding: utf-8 -*-
"""**Table 5. 빈혈 판정 — 고정 · 개발 · 현지 문턱** (11-04 : 1 seed 10겹 앙상블 · 중간값 (범위) · 옛 판 _old_261030/)

절차
  개발  MobileNetV3-L · S1 + S2 · seed 1–10 × 10겹 교차검증 (앙상블 아님) · 겹마다 안쪽 검증 참여자로 문턱 → 그 겹의 시험 참여자 (두 눈 평균)
  외부  SS · seed 마다 : 추첨마다 10겹 모델의 예측을 사람별로 평균 (앙상블) → 문턱과 비교
        고정 = 정의값 · 개발 = `L1_임계_내부.csv` 의 그 seed 값 · 현지 = 보정 30명 안 3겹 안쪽 예측을 10겹 모델끼리 평균해 고른 문턱
  문턱 고르기 : `main/analysis/260929_임계선정.py` 의 `임계고르기` (민감도 ≥ 0·70 중 특이도 최대)
값    참여자마다 "양성으로 판정된 추첨의 비율" q → seed 마다 기대 수로 민감도 · 특이도 · PPV · NPV · 정확도 · F1 → seed 중간값 (범위)
      CI · p : 참여자 bootstrap 100,000 (재표집마다 seed 중간값 · 현지 대 개발은 같은 재표집의 차이)
      AUROC : seed 마다 참여자 점수 (−예측 Hb · 추첨 평균) · 미세조정 전은 같은 모델의 10겹 예측 평균
1차 자료  main/results/{최소구성/train, 임계/내부_검증예측, 최소구성/external/{zero,few_bn}, 임계/외부_안쪽예측/lr2e-05_bn} · main_results/L1_임계_내부.csv (E13)
내는 것   results_table_figure/main/Table5_thresholds_10seed.xlsx
쓰는 법   python main_table5_thresholds_10seed.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)   # 양성 판정이 0 이면 PPV 가 빈 값이다 (칸에는 ·· 로 나온다)

임뿌 = Path(os.environ.get("표_임계뿌리", 결과 / "임계"))   # 시험 때만 바꾼다
src = (뿌리 / "main/analysis/260929_임계선정.py").read_text(encoding="utf-8")
ns = {"__name__": "임계선정", "__file__": str(뿌리 / "main/analysis/260929_임계선정.py")}
exec(compile(src[:src.index("ap = argparse.ArgumentParser()")], "260929_임계선정.py", "exec"), ns)
임계고르기 = ns["임계고르기"]

목표 = (12.0, 11.0); CELL = "mn_mnl_veminimal_K40_auto_scca_seed{s}"; SP = 결과 / "external/splits"
# ★ 11-07 : 외부 = 권유 구성 · 학습률 (`_공통`) · 현지 문턱 = 임계/외부_안쪽예측/<태그> (L13) · 개발 문턱은 S1 + S2 개발 모델 그대로 (E13)
_, 권a, 권폴 = 권유
if 권a != "veminimal": sys.exit(f"🔴 권유 구성 {구성글(권a, 권폴)} — 개발 문턱 (L1_임계_내부.csv · 내부_검증예측) 이 S1 + S2 개발 모델 기준이라 그 구성으로 다시 만들어야 한다")
권뿌, 권틀 = 칸찾기("mnl", 권a); 안태그 = f"lr{float(보정lr):g}" + ("_bn" if 권폴.endswith("_bn") else "")
L1 = pd.read_csv(Path(os.environ.get("표_L1", MR / "L1_임계_내부.csv")))
명 = pd.read_csv(자료 / "internal/명부.csv", usecols=["stem", "number"]); 번호 = dict(zip(명.stem, 명.number))


def 개발():
    """→ {th: (y, q(seed×참여자), 문턱들)} 자료 기반 · 고정 · 점수 행렬 (seed × 참여자)."""
    판 = {th: {} for th in 목표}; 고 = {th: {} for th in 목표}; 문 = {th: [] for th in 목표}; y = {}; 점수 = {}
    for s in S10:
        점수[s] = {}
        for f in range(CV):
            V = pd.read_csv(임뿌 / "내부_검증예측" / f"seed{s}_fold{f}.csv"); V["n"] = V.key.map(lambda k: 번호.get(k))
            검 = V.dropna(subset=["n"]).groupby("n").agg(y=("y", "first"), p=("pred", "mean"))
            D = pd.read_csv(결과 / "최소구성/train" / f"{CELL.format(s=s)}-fold{f}" / "test_pred.csv"); D["n"] = D.file_path.map(lambda x: 번호[Path(x).stem])
            시 = D.groupby("n").agg(y=("y", "first"), p=("pred_best", "mean"))
            for th in 목표:
                t = 임계고르기(검.y, 검.p, th)
                if np.isnan(t): sys.exit(f"🔴 개발 seed{s} fold{f} Hb<{th} 문턱 없음")
                문[th].append(t)
            for n, r in 시.iterrows():
                y[n] = r.y; 점수[s][n] = -r.p
                for i, th in enumerate(목표):
                    고[th].setdefault((s, n), int(r.p < th)); 판[th].setdefault((s, n), int(r.p < 문[th][-1]))
    키 = sorted(y); yy = np.array([y[n] for n in 키])
    q = lambda g: np.array([[g[(s, n)] for n in 키] for s in S10], float)
    S = np.array([[점수[s][n] for n in 키] for s in S10])
    return {th: (yy, q(판[th]), 문[th]) for th in 목표}, {th: (yy, q(고[th]), [th]) for th in 목표}, S


def 외부_(site):
    """→ {(방식, th): (y, q(seed×참여자 · 평가되지 않은 칸은 nan), 문턱들)} · 점수 (후 · 전)."""
    v = 외부값("mnl", 권a, 권폴, site); 키 = sorted(v.key.unique()); 위 = {k: i for i, k in enumerate(키)}
    y = v[v.seed == 1].set_index("key").y.reindex(키).to_numpy(float); 판, 수, 문 = {}, {}, {}
    for si, s in enumerate(S10):
        for r in range(10):
            ev = pd.concat([pd.read_csv(권뿌 / "external" / 권폴 / site / "n30" / f"rep{r}" / 권틀.format(s=s) / f"fold{f}.csv", usecols=["key", "y", "pred"]) for f in range(CV)]).groupby("key").pred.mean()
            안 = pd.concat([pd.read_csv(임뿌 / "외부_안쪽예측" / 안태그 / f"{권틀.format(s=s)}_fold{f}_{site}_rep{r}.csv") for f in range(CV)]).groupby("key").agg(y=("y", "first"), pred=("pred", "mean"))
            i = np.array([위[k] for k in ev.index])
            for th in 목표:
                for 방식, t in [("고정", th), ("개발", float(L1[(L1.seed == s) & (L1.목표Hb == th)].임계평균.iloc[0])), ("현지", 임계고르기(안.y, 안.pred, th))]:
                    if np.isnan(t): continue                                  # 그 추첨의 30명에 빈혈이 없으면 현지 문턱을 고를 수 없다
                    문.setdefault((방식, th), []).append(t)
                    A = 판.setdefault((방식, th), np.zeros((len(S10), len(키)))); C = 수.setdefault((방식, th), np.zeros((len(S10), len(키))))
                    A[si, i] += (ev.to_numpy() < t); C[si, i] += 1
    q = {m: (y, np.where(수[m] > 0, 판[m] / np.maximum(수[m], 1), np.nan), 문[m]) for m in 판}
    return q, -행렬(v, "p").to_numpy(float).T, -행렬(무보정값("mnl", 권a, site), "p").to_numpy(float).T


def _지표(TP, FN, FP, TN):
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(sens=TP / (TP + FN), spec=TN / (TN + FP), ppv=TP / (TP + FP), npv=TN / (TN + FN), acc=(TP + TN) / (TP + FN + FP + TN), f1=2 * TP / (2 * TP + FP + FN))


def 점표(pos, q):
    """seed 마다 지표 (점 추정)."""
    u = ~np.isnan(q); q0 = np.nan_to_num(q)
    return _지표((q0 * pos).sum(1), ((1 - q0) * pos * u).sum(1), (q0 * ~pos).sum(1), ((1 - q0) * ~pos * u).sum(1))


def 재표집지표(pos, q):
    """재표집마다 seed 중간값 → {지표: 길이 BOOT}."""
    u = ~np.isnan(q); q0 = np.nan_to_num(q); n = len(pos); W = 재표집(n)
    A = [(q0 * pos).T, ((1 - q0) * pos * u).T, (q0 * ~pos).T, ((1 - q0) * ~pos * u).T]; 모 = {}
    for i in range(0, len(W), 2000):
        w = W[i:i + 2000]; m = _지표(*[w @ a for a in A])
        for k, v in m.items(): 모.setdefault(k, []).append(np.nanmedian(v, axis=1))
    return {k: np.concatenate(v) for k, v in 모.items()}


def 칸(v, b, 자릿=3):
    return f"{점(float(np.nanmedian(v)), 자릿)} ({구간(float(np.nanmin(v)), float(np.nanmax(v)), 자릿)}) [{구간(*np.nanpercentile(b, [2.5, 97.5]), 자릿)}]"


시작("Table 5 — 판정 문턱 (고정 · 개발 · 현지)")
D, DF, Sd = 개발(); E = {s: 외부_(s) for s in SITES}
행 = []; 값글 = "median (seed range) [95% CI]"
for th in 목표:
    정의 = f"Haemoglobin <{점(th, 1)} g/dL"; 고정이름 = "fixed (predicted haemoglobin below the definition)"
    줄 = [("Development cohort", 고정이름, *DF[th], Sd, None, None), ("Development cohort", "development, data-driven (reference)", *D[th], Sd, None, None)]
    for s in SITES:
        이 = 영문사이트[s]; q, S, S0 = E[s]
        줄 += [(이, 고정이름, *q[("고정", th)], S, S0, None), (이, "development, data-driven, applied unchanged (reference)", *q[("개발", th)], S, S0, None),
              (이, "local, data-driven in the 30 fine-tuning participants", *q[("현지", th)], S, S0, q[("개발", th)][1])]
    a_캐시 = {}
    for 어디, 이름, y, q, 문, S, S0, 기준q in 줄:
        pos = y < th; m = 점표(pos, q); b = 재표집지표(pos, q)
        if 어디 not in a_캐시: a_캐시[어디] = AUROC요약(pos, S, S0)
        a, a0 = a_캐시[어디]
        r = {"Measure": f"{정의}: {어디}, {이름} cut-off", "Decision cut-off, predicted haemoglobin, g/dL": 점(float(np.mean(문)), 1),
             "Participants, n": len(y), "Anaemic participants, n": int(pos.sum())}
        for 키, 열 in [("sens", "Sensitivity"), ("spec", "Specificity"), ("ppv", "PPV"), ("npv", "NPV")]: r[f"{열}, {값글}"] = 칸(m[키], b[키])
        r[f"AUROC, {값글}"] = AUROC칸(a); r[f"AUROC before fine-tuning, {값글}"] = AUROC칸(a0) if a0 else "··"
        r["p value, AUROC, after vs before fine-tuning"] = p표(a0[5]) if a0 else "··"
        r[f"Accuracy, {값글}"] = 칸(m["acc"], b["acc"]); r[f"F1 score, {값글}"] = 칸(m["f1"], b["f1"])
        for 키, 열 in [("sens", "sensitivity"), ("spec", "specificity")]:
            if 기준q is None: r[f"p value, {열}, local vs development cut-off"] = "··"; continue
            d = b[키] - 재표집지표(pos, 기준q)[키]
            r[f"p value, {열}, local vs development cut-off"] = p표(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
        행.append(r)
        print(f"  Hb<{th:.0f} {어디:18} {이름[:12]:12} 문턱 {np.mean(문):.2f} · 민감 {r[f'Sensitivity, {값글}']} · 특이 {r[f'Specificity, {값글}']} · AUROC {r[f'AUROC, {값글}']}", flush=True)
T, 머리 = 소제목표(pd.DataFrame(행))
엑셀("Table5_thresholds_10seed", 본문, {"Table 5": T}, 소제목=머리,
    제목=("Table 5. Anaemia classification with the decision cut-off from the development cohort applied unchanged and with a "
         "cut-off derived from the 30 local fine-tuning participants"),
    주석=보고각주 + [
      "Three cut-offs on the predicted haemoglobin. Fixed: the anaemia definition itself (predicted haemoglobin below 12·0 or 11·0 g/dL).",
      "  Development, data-driven: chosen in the development cohort as the cut-off with the highest specificity among those with a sensitivity",
      "  of at least 0·70, and applied unchanged at the external locations. Local, data-driven: chosen by the same rule using only the 30 local",
      "  participants used for fine-tuning, and applied to the other participants of that location. Cut-offs are shown as the mean.",
      "Development cohort: MobileNetV3-Large with S1 + S2 (762 participants). External locations: the same model fine-tuned with S1 + S2 + S3.",
      "Sensitivity, specificity, PPV, NPV, accuracy and F1 score are computed for each seed from each participant's proportion of draws in",
      "  which they were classified as anaemic. AUROC does not depend on the cut-off and is the same in every row of a location; AUROC before",
      "  fine-tuning uses the same model before fine-tuning in the same participants.",
      "PPV, positive predictive value; NPV, negative predictive value; AUROC, area under the receiver operating characteristic curve.",
      "Haemoglobin <12·0 g/dL is the lower of the two WHO 2024 adult cut-offs (women 12·0, men 13·0 g/dL) and <11·0 g/dL marks moderate or severe",
      "  anaemia in both sexes; sex was not recorded. Results are reported by location and are not pooled.",
    ])
