# -*- coding: utf-8 -*-
"""**Table 2. 자동분할 품질** — 사이트별 판독 · 2단계 · 실패 탐지 · 판독자 일치도.  2026-10-05

왜  주장 ③(사람 손 없이)의 근거. 사람이 아무것도 그리지 않고 사진 전체에서 낱말로 분할했을 때
    얼마나 맞는지, 실패를 사람 없이 알아내는지, 판정을 두 사람이 얼마나 같게 하는지를 사이트별로 보인다.

분모  모델에 쓴 사진 — 내부 1,524 · 인도 95 · 이탈리아 122 (판독 전수 1,733 과 섞지 않는다)
      ★ 10-25 이탈리아 107 → 122 : 추가 15장 (53 · 59 · 60 · 69–80번) 의 두 판독 · 단계 (first_pass.csv) 를 더한다
      ★ 10-25 2차 판독은 기준을 다시 설명한 뒤 1차를 보지 않고 전체를 다시 한 판 (`…_2차판정다시.csv`) 으로 바꿨다
        (옛 2차 판독은 usb_판정/260919_2nd_판독_1st_old/ · 옛 스크립트 _이전판/main_table2_segmentation_전_261025.py)

1차 자료
  판독 1차  usb_판정/_판정끝난뒤_열기/교수님판정_{안쪽,인도,이탈리아}.csv
  판독 1차  + usb_판정/usb2_이탈리아추가15_판정_1판독자.csv (이탈리아 추가 15장)
  판독 2차  usb_판정/260919_2nd_판독/usb2_{안쪽,인도,이탈리아}_판정_2차판정다시.csv
            + usb_판정/260919_2nd_판독/usb2_이탈리아추가15_2판독자_판정다시.csv
  단계      ../260822_…/results/260909_final3/장별_{안쪽,바깥쪽}.csv  (주 1,479 · 외부 202)
            segmentation/results/260909_final3/장별_안쪽.csv            (되살린 42)
            segmentation/cand_260911/results/260909_final3/장별_안쪽.csv (후보 359)
            🔴 세 내부 파일을 **모두** 합쳐야 1,524장이 다 찬다 (하나라도 빼면 1,491장 · 66장이 된다)
  실패 탐지 segmentation/results/261005_first_pass_failure/first_pass.csv
            (`segmentation/analysis/261005_first_pass_failure.py` 가 만든다 — 먼저 돌린다)
            정답 = 최종 파이프라인 1단계 공막이 사람이 그린 공막을 덮은 몫 < 0.5 · 점수 = `공막몫`

🔵 **모든 칸의 분모가 연구 대상 그대로다** — 내부 1,524 · 인도 95 · 이탈리아 122. 실패 탐지 AUC 도 같다.
   (예전 값 0.985 · 0.980 은 09-06 분할 판의 내부 1,479장 — 최종 1,524장과 1,411장만 겹친다 — 에서 잰 것이다)

내는 것  results_table_figure/main/Table2_segmentation.xlsx
쓰는 법  python main_table2_segmentation.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *
from sklearn.metrics import roc_auc_score, cohen_kappa_score

옛 = 뿌리 / "segmentation/results"   # ★ 10-28 : 옛 프로젝트 260822_… 의 results/260909_final3/장별_{안쪽,바깥쪽}.csv 두 파일을 새 폴더 segmentation/results/260822판_260909_final3/ 로 복사해 읽는다 (옛 프로젝트는 ssd3 로 옮김)
판독 = 뿌리 / "usb_판정"
사이트 = [("안쪽", "Development cohort"), ("인도", "Indian location"), ("이탈리아", "Italian location")]


def 읽csv(f):
    d = pd.read_csv(f); d.columns = [str(c).lstrip("﻿") for c in d.columns]
    if "stem" in d.columns: d["stem"] = d.stem.astype(str)
    return d


def 무리(stem):
    return "인도" if stem.startswith("i_") else "이탈리아"


시작("Table 2 — 자동분할 품질")
# ── 판독 ──────────────────────────────────────────────────────────────────
판 = {k: 읽csv(판독 / "_판정끝난뒤_열기" / f"교수님판정_{k}.csv") for k, _ in 사이트}
판2 = {k: 읽csv(판독 / "260919_2nd_판독" / f"usb2_{k}_판정_2차판정다시.csv") for k, _ in 사이트}
판["이탈리아"] = pd.concat([판["이탈리아"], 읽csv(판독 / "usb2_이탈리아추가15_판정_1판독자.csv")], ignore_index=True)
판2["이탈리아"] = pd.concat([판2["이탈리아"], 읽csv(판독 / "260919_2nd_판독" / "usb2_이탈리아추가15_2판독자_판정다시.csv")], ignore_index=True)
for k, _ in 사이트:
    if 판[k].stem.duplicated().any() or set(판[k].stem) != set(판2[k].stem): sys.exit(f"🔴 {k} 두 판독의 장이 다르다")

# ── 단계 ──────────────────────────────────────────────────────────────────
명 = 읽csv(자료 / "internal/명부.csv"); 내부stem = set(명.stem)
내단계 = pd.concat([읽csv(옛 / "260822판_260909_final3/장별_안쪽.csv"),
                    읽csv(뿌리 / "segmentation/results/260909_final3/장별_안쪽.csv"),
                    읽csv(뿌리 / "segmentation/cand_260911/results/260909_final3/장별_안쪽.csv")]).drop_duplicates("stem", keep="last")
내단계 = 내단계[내단계.stem.isin(내부stem)]
if len(내단계) != len(내부stem):
    sys.exit(f"🔴 내부 단계 정보가 {len(내단계)}장뿐이다 — 1,524장이 다 차야 한다")
외단계 = 읽csv(옛 / "260822판_260909_final3/장별_바깥쪽.csv"); 외단계["무리"] = 외단계.stem.map(무리)
단계 = {"안쪽": 내단계, "인도": 외단계[외단계.무리 == "인도"], "이탈리아": 외단계[외단계.무리 == "이탈리아"]}

# ── 실패 탐지 (연구 대상 전부) ────────────────────────────────────────────
FP = 뿌리 / "segmentation/results/261005_first_pass_failure/first_pass.csv"
if not FP.exists(): sys.exit("🔴 먼저 python segmentation/analysis/261005_first_pass_failure.py")
fp = 읽csv(FP); fp["사이트"] = fp["사이트"].replace({"내부": "안쪽"})
추가 = fp[(fp["사이트"] == "이탈리아") & ~fp.stem.isin(단계["이탈리아"].stem)][["stem", "단계"]]   # 추가 15장의 단계 (최종 파이프라인 그대로)
단계["이탈리아"] = pd.concat([단계["이탈리아"], 추가], ignore_index=True)
for k in ["인도", "이탈리아"]:                     # 내부 판독 파일은 가명 stem (P…) 이라 장 수로만 맞춘다 (위 1,524 확인)
    if set(단계[k].stem) != set(판[k].stem): sys.exit(f"🔴 {k} 단계 정보 {len(단계[k])}장 — 판독 {len(판[k])}장과 다르다")

# ★ 10-27 (사용자) : 기관마다 SCCA · 결막 단독 두 열 · 두 판독자의 적절 비율 모두 · 오류 종류는 1차 판독 (더 엄격 · 보수적)
#   2단계 · 실패 탐지는 사진 단위라 SCCA 열에만 (결막 열 ··) · 옛 판 _이전판/main_table2_segmentation_전_결막열_261027.py
오류 = {"scca_code": [("extended upward", ["OVER_UP"]),
                     ("extended onto the lower-lid skin", ["OVER_DOWN"]), ("incomplete", ["UNDER_BC", "UNDER_WC"]),
                     ("region not found", ["MISS"]), ("other", ["OVER_BOTH", "SCATTER"])],
       "bc_code":   [("extended upward", ["OVER_UP"]),
                     ("extended onto the lower-lid skin", ["OVER_DOWN"]), ("incomplete", ["PART_INCOMP"]),
                     ("region not found", ["MISS"]), ("other", ["OVER_BOTH", "SCATTER"])]}
칸 = {}
for k, 이 in 사이트:
    a = 판[k]; n = len(a)
    s_ = 단계[k]; s2 = int((s_["단계"] == 2).sum())
    m = fp[fp["사이트"] == k]
    if len(m) != n: sys.exit(f"🔴 {k} 실패 탐지 자료가 {len(m)}장 — 판독 {n}장과 다르다")
    y = m["실패"].astype(int)
    auc = roc_auc_score(y, -m.공막몫) if y.nunique() == 2 else np.nan
    건짐 = int(((m["단계"] == 2) & (y == 1)).sum())
    b = a.merge(판2[k], on="stem", suffixes=("_1", "_2"))
    for 코드, 영역 in [("scca_code", "SCCA"), ("bc_code", "conjunctiva alone")]:
        c1 = a[코드].value_counts(); ok1 = int(c1.get("OK", 0)); ok2 = int((판2[k][코드] == "OK").sum())
        o1, o2 = (b[f"{코드}_1"] == "OK").astype(int), (b[f"{코드}_2"] == "OK").astype(int)
        모른 = set(c1.index) - {"OK"} - {x for _, xs in 오류[코드] for x in xs}
        if 모른: sys.exit(f"🔴 {k} {코드} 모르는 판정 {모른}")
        sc = 코드 == "scca_code"
        r = {"Images, n": n,
             "Region judged adequate, reader 1, n (%)": f"{ok1} ({점(ok1/n*100, 1)})",
             "Region judged adequate, reader 2, n (%)": f"{ok2} ({점(ok2/n*100, 1)})"}
        for 이름, xs in 오류[코드]:
            r[f"Errors judged by reader 1: {이름}, n"] = int(sum(c1.get(x, 0) for x in xs))
        r["Two readers: agreement on adequacy, %"] = 점(float((o1 == o2).mean() * 100), 1)
        r["Two readers: kappa for adequacy"] = 점(float(cohen_kappa_score(o1, o2)), 2)
        r["Photograph routed to the second pass, n (%)"] = f"{s2} ({점(s2/n*100, 1)})" if sc else "··"
        r["Detection of first-pass failure without human review, AUROC"] = 점(float(auc), 3) if sc else "··"
        r["First-pass failures routed to the second pass, n / N"] = f"{건짐} / {int(y.sum())}" if sc else "··"
        칸[f"{이}: {영역}"] = r
        print(f"  {이:20} {영역:18} 적절 1차 {ok1}/{n} · 2차 {ok2} · κ {r['Two readers: kappa for adequacy']}")
T = pd.DataFrame(칸); T.index.name = "Characteristic"; T = T.reset_index()
T, 머리 = 소제목표(T, 첫열="Characteristic")
엑셀("Table2_segmentation", 본문, {"Table 2": T}, 소제목=머리,
    제목="Table 2. Automated segmentation of the sclera–conjunctiva co-analysis (SCCA) region and of the conjunctiva alone without manual delineation, by site",
    주석=[
      "The region was obtained from the whole photograph with everyday-language prompts; no box or outline was drawn.",
      "Denominators are the images used for modelling: 1524 in the development cohort, 95 and 122 at the external",
      "  locations (one image per participant externally).",
      "Second pass: a first pass whose sclera, after removing the conjunctiva, was less than 0·30 of the conjunctiva in area",
      "  was judged to have failed and the region was re-derived from the start.",
      "Detection of first-pass failure: the first pass was defined as failed when its sclera covered less than half of the manually annotated",
      "  sclera; the score was the ratio above, computed from the image alone. Evaluated in all study images.",
      "Both regions were graded by two clinicians for every image. Adequate: no error; error types are those judged by reader 1,",
      "  who graded more strictly. Extended upward: beyond the region into the cornea or iris (SCCA) or into the sclera",
      "  (conjunctiva alone). All images were retained in every analysis, whatever their grades.",
      "Reader agreement: simple agreement and Cohen's kappa for adequacy (adequate versus any error). The second reader",
      "  graded every image independently, without access to the first reader's grades, after a standardised briefing on",
      "  the grading criteria.",
      "The second pass and its detection concern the photograph as a whole and are shown once, in the SCCA column (··).",
    ])
print(T.to_string(index=False))
