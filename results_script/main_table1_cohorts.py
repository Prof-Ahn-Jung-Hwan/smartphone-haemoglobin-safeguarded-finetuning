# -*- coding: utf-8 -*-
"""**10-20 — 부록 표 번호를 한 칸 밀었다** (Supplementary Table 1 = 선행 연구 요약이 새로 들어감). 이 설명 안의 표 번호 · 파일 이름도 새 번호로 바꿨다 — 옛 판 스크립트는 `_이전판/` 에 옛 번호 그대로 있다.

**10-17 판** — 개발 코호트의 조명 칸을 \"Standardised, single protocol\" → \"Ordinary indoor lighting; no attachment; single photography protocol\" 로 (Methods 와 맞춤 · 교수님 10-17). 외부 칸에 매크로 렌즈를 넣었다. 원본 `main_table1_cohorts.py` 는 그대로 둔다.

**Table 1. 코호트 특성** — 개발 코호트와 외부 두 모집 지역.  2026-10-02

1차 자료  data/internal/명부.csv · main/external/data/arm_ext_auto_scca.csv
내는 것    results_table_figure/main/Table1_cohorts.xlsx

🔵 나이·성별은 **개발 코호트에서 수집하지 않았다**. 외부 자료에는 있으므로 그대로 적고,
   입력으로 쓸 수 없다는 점은 Methods 에 쓴다. "인종" 은 쓰지 않는다.
🔵 외부 두 지역은 **같은 기기·LED·거리**다(Dimauro 2023). 그래서 '기기'는 개발 대비로만 적는다.

쓰는 법  python main_table1_cohorts.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _공통 import *

시작("Table 1 — 코호트 특성")
내 = 내부Hb(); 외 = 외부명부()
명부 = pd.read_csv(자료 / "internal/명부.csv")

열 = ["Development cohort (Republic of Korea)", "External: Indian location", "External: Italian location"]
집 = {열[0]: 내, 열[1]: 외[외.Country == "India"].Hb, 열[2]: 외[외.Country == "Italy"].Hb}

행 = []
def 더(항, f):   행.append({"Characteristic": 항, **{k: f(v, k) for k, v in 집.items()}})

더("Participants, n", lambda v, k: len(v))
더("Images, n", lambda v, k: len(명부) if k == 열[0] else len(v))   # 외부는 1인 1장 (10-05 고침 · 전에는 ×2)
더("Haemoglobin, g/dL, mean (SD)", lambda v, k: f"{점(v.mean(), 1)} ({점(v.std(), 1)})")
더("Haemoglobin, g/dL, median (IQR)",
   lambda v, k: f"{점(v.median(), 1)} ({점(v.quantile(.25), 1)}–{점(v.quantile(.75), 1)})")
더("Haemoglobin, g/dL, range", lambda v, k: f"{점(v.min(), 1)}–{점(v.max(), 1)}")
for 문, 설 in [(12.0, "Haemoglobin <12·0 g/dL, n (%)"), (11.0, "Haemoglobin <11·0 g/dL, n (%)"),
              (10.0, "Haemoglobin <10·0 g/dL, n (%)"), (8.0, "Haemoglobin <8·0 g/dL, n (%)")]:
    더(설, lambda v, k, 문=문: f"{int((v < 문).sum())} ({점((v < 문).mean()*100, 1)})")
더("Haemoglobin ≥14·0 g/dL, n (%)", lambda v, k: f"{int((v >= 14).sum())} ({점((v >= 14).mean()*100, 1)})")

# 🔵 나이·성별은 표에 넣지 않는다 (2026-10-02 교수님 결정) — Methods 본문에 한 줄로 적는다.
#    개발 코호트에서 수집하지 않아 입력으로 쓸 수 없고, 표에 "Not collected" 를 세 번 적을 이유가 없다.

# 🔴 개발 코호트는 **두 기종**이다 (EXIF 실측) — "single smartphone" 은 사실과 다르다.
#    SM-G991N(Galaxy S21) 1,166장 · SM-N971N(Galaxy Note10 5G) 276장 · 기종 불명 82장
def 기기(v, k):
    if k != 열[0]: return "Samsung S6 (identical at both locations)"   # 11-10 : 원문(Dimauro 2023) 표기
    x = pd.read_csv(자료 / "_내부용_EXIF.csv")   # internal file, not included in this repository
    x.columns = [c.strip().lstrip("\ufeff") for c in x.columns]
    c = x.Model.value_counts(); 전 = len(명부)
    보기 = {"SM-G991N": "Galaxy S21", "SM-N971N": "Galaxy Note10 5G"}
    조각 = [f"{보기.get(m, m)} {n} ({점(n/전*100, 1)})" for m, n in c.items()]   # 10-20 : % 빼기 (각주 "n (%)")
    남 = 전 - int(c.sum())
    if 남: 조각.append(f"model not in metadata {남} ({점(남/전*100, 1)})")   # 10-20 : 가운데 점 (보통 소수점이었다)
    return "; ".join(조각)
더("Smartphone", 기기)
더("Illumination, attachment and distance", lambda v, k: "Ordinary indoor lighting; no attachment; single photography protocol"
   if k == 열[0] else "Macro lens and white LED, fixed distance (identical at both locations)")

T = pd.DataFrame(행)[["Characteristic"] + 열]

엑셀("Table1_cohorts", 본문, {"Table 1": T},
    제목="Table 1. Characteristics of the development cohort and the two external recruitment locations",
    주석=[
      "Data are n (%) unless stated otherwise. IQR, interquartile range.",
      "Haemoglobin <12 and <11 g/dL are the detection targets used throughout. Sex was not recorded in the development",
      "  cohort, so sex-specific WHO definitions of anaemia (WHO 2024) were not applied.",
      "Age and sex were not collected in the development cohort and could not be used as model inputs; they are",
      "  addressed in the Methods and are not tabulated here.",
      "Race and ethnicity were not collected; cohort differences are described in geographical and technical terms.",
      "Development images were taken with two consumer smartphone models under ordinary indoor lighting, with no",
      "  colour chart, ring light or macro attachment; both models differ from the one used in the external dataset.",
      "Both external locations were photographed with the same smartphone, the same white LED illumination and the",
      "  same distance (Dimauro et al. 2023); they used the same acquisition hardware but differ in population.",
    ])
print(T.to_string(index=False))
