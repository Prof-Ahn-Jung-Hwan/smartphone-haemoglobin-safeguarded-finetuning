# -*- coding: utf-8 -*-
"""표 생성 스크립트들의 **공통 바탕**.  2026-10-02 · ★ 11-04 : 보고 단위를 1 seed 10겹 앙상블 · 중간값 (범위) 으로 바꿈 (옛 판 _old_261030/_공통.py)

왜  표 하나에 스크립트 하나로 가면 경로·부트스트랩·엑셀 서식이 스크립트마다 흩어진다.
    그 셋만 여기에 모아 두고, 각 표 스크립트는 **자기 표의 내용만** 담는다.

규칙
  · 본문 표는  paperwork/results_table_figure/main/
  · 부록 표는  paperwork/results_table_figure/supplementary/
  · 표 내용은 **영문** (투고 원고에 그대로 들어간다) · 스크립트 설명은 한국어
  · 수치는 **반올림해서 문자열로** 넣지 않는다 — 숫자는 숫자로 넣고 엑셀 서식으로 자릿수를 맞춘다

쓰는 법 (각 표 스크립트에서)
    from _공통 import *
    엑셀("Table3_external_backbones", 본문, {"Table 3": df}, 제목="...", 주석=[...])
"""
import os, re, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

# ── 경로 ───────────────────────────────────────────────────────────────────────
여기 = Path(__file__).resolve().parent
뿌리 = 여기.parent.parent                      # …/260910_eClinicalMedicine_filtered_2_autosegmentation
자료 = 뿌리 / "data"
결과 = 뿌리 / "main" / "results"
MR   = 결과 / "main_results"
# ★ 11-07 : 미세조정 학습률 본분석 2e-5 → 1e-4 (방법론 문서 §8-8). 표_보정lr 로 고른다 (기본 1e-4).
#   1e-4 → 폴더 few_lr0.0001 · few_lr0.0001_bn · 표는 results_table_figure_lr1e-4/ (옛 2e-5 표를 덮어쓰지 않는다)
#   2e-5 → 옛 폴더 few · few_bn · 옛 표 자리 results_table_figure/ (다시 만들 때만 · 표_보정lr=2e-5)
보정lr = os.environ.get("표_보정lr", "1e-4")
if float(보정lr) == 2e-5: FT, FTB, 판꼬리 = "few", "few_bn", ""
else: FT, FTB, 판꼬리 = f"few_lr{float(보정lr):g}", f"few_lr{float(보정lr):g}_bn", f"_lr{보정lr}"
본문  = 뿌리 / "paperwork" / f"results_table_figure{판꼬리}" / "main"
부록  = 뿌리 / "paperwork" / f"results_table_figure{판꼬리}" / "supplementary"
for d in (본문, 부록, MR, MR / "_앙상블값"): d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(뿌리 / "code"))

# ── 칸 표 · 이름 ──
ROOT = 뿌리
R = 뿌리 / "main" / "results"
팔들 = ["mse", "normonly", "veminimal", "veminimalnoln", "anchor", "anchorshuf", "batchmean"]
이름 = {"mse": "안전장치없음", "normonly": "F 정규화만", "veminimal": "M", "veminimalnoln": "M−LayerNorm",
       "anchor": "M+S4", "anchorshuf": "M+S4n", "batchmean": "M+S5"}
BB = ["mnl", "eb0", "enxs", "mnv2", "mvxx"]
표시 = {"mnl": "MobileNetV3-L", "eb0": "EfficientNet-B0", "enxs": "EdgeNeXt-XS",
       "mnv2": "MobileNetV2", "mvxx": "MobileViT-XXS"}
BN층 = {"mnl": 46, "eb0": 49, "enxs": 0, "mnv2": 52, "mvxx": 32}     # 실측 (torchvision/timm 모듈 수)
BN보유 = [b for b in BB if BN층[b] > 0]
칸 = {"mnl": {"mse": (R/"stage5", "s5_mnl_mse_K40_auto_scca_seed{s}"),
             "normonly": (R/"기전", "mx_mnl_normonly_K40_auto_scca_seed{s}"),
             "veminimal": (R/"최소구성", "mn_mnl_veminimal_K40_auto_scca_seed{s}"),
             "veminimalnoln": (R/"M빼기LN", "ml_mnl_veminimalnoln_K40_auto_scca_seed{s}"),
             "anchor": (R/"anchor", "an_mnl_anchor_K40_auto_scca_seed{s}"),
             "anchorshuf": (R/"anchor", "an_mnl_anchorshuf_K40_auto_scca_seed{s}"),
             "batchmean": (R/"anchor", "an_mnl_batchmean_K40_auto_scca_seed{s}")}}
for b in BB[1:]:
    칸[b] = {a: (R/"백본재현", f"bb_{b}_{a}_K40_auto_scca_seed{{s}}") for a in 팔들}
CV = 10; SITES = ["India", "Italy"]


def boot(v, B_=100000, seed=0):
    """환자 재표집 95 % CI 와 양측 p (bootstrap 분포에서 0 의 양쪽 비율 중 작은 쪽 × 2).
    ★ 10-18 통계 상의 : 재표집 2,000 → **100,000회** (p 0·0001 을 가린다 · 양측 p 의 최소 단위 2 / B = 0·00002).
    이전 값으로 돌리려면 B_=2000 (백업 previous_strategy_notNow/260930_요인분석_before_B100000_backup.py)."""
    rng = np.random.RandomState(seed); n = len(v)
    m = np.array([v[rng.randint(0, n, n)].mean() for _ in range(B_)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), \
           float(min(1.0, 2*min((m <= 0).mean(), (m >= 0).mean())))


def 대응(A, B, **k):
    c = A.index.intersection(B.index)
    return boot((B[c] - A[c]).to_numpy(float), **k)


영문사이트 = {"India": "Indian location", "Italy": "Italian location"}


# ── 숫자 표기 ─────────────────────────────────────────────────────────────────
def p꼴(p):
    """p 값을 학술지 관행대로. 0.001 미만은 '<0.001'."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def 효과(x, 자릿=3):
    """(차이, 하한, 상한, p) → 'd (lo to hi); p=…' — 표 한 칸에 들어가는 꼴."""
    d, lo, hi, p = x
    return f"{d:+.{자릿}f} ({lo:+.{자릿}f} to {hi:+.{자릿}f}); p={p꼴(p)}"


def CI(x, 자릿=3):
    """(값, 하한, 상한, …) → '값 (하한 to 상한)'."""
    return f"{x[0]:.{자릿}f} ({x[1]:.{자릿}f} to {x[2]:.{자릿}f})"


# ── 10-19 표기 규칙 (원고 표 · Lancet / eClinicalMedicine · 결정 문서 paperwork/261019_결정_원고틀C_표기.md §5) ──
#   ① 가운데 점 · 음수는 "−" · 자리 수 : Hb 값 1 · MAE · 차이 2 (기본) · AUROC · 민감도 · 특이도 3 → 칸마다 자릿= 로
#   ② 95 % CI (Lancet) : 두 끝이 모두 0 이상이면 en dash "1·23–1·36", 하나라도 음수면 "−0·12 to −0·02"
#      CI 끝이 반올림으로 0 이 되어도 자릿수는 그대로 ("0·00") — 판정은 p (10-25 · 옛 규칙 '한 자리 더' 는 버림)
#   ③ p : 유효숫자 둘 · 0·0001 미만은 "<0·0001" · 재표집 100,000회
#   ④ 추정치 칸은 "값 (CI)" · 비교는 차이 | 95 % CI | p 를 따로
def 점(x, 자릿=2, 부호=False):
    """숫자 → 가운데 점 · 빼기 기호. 부호=True 면 양수에 + 를 붙인다."""
    if x is None or x != x: return "··"
    if round(x, 자릿) == 0: x, 부호 = 0.0, False          # "−0·00" · "+0·00" 대신 "0·00"
    s = f"{x:+.{자릿}f}" if 부호 else f"{x:.{자릿}f}"
    return s.replace("-", "−").replace(".", "·")


def 구간(lo, hi, 자릿=2):
    """95 % CI 두 끝 → '1·23–1·36' 또는 '−0·12 to 0·03' (괄호 없음 · 값CI 가 괄호를 붙인다)."""
    if lo != lo or hi != hi: return "··"
    # ★ 10-25 (사용자) : 끝이 반올림으로 0 이 되어도 자릿수를 늘리지 않는다 — 표 전체를 같은 자리로 ("0·00").
    #   판정은 반올림 전 값의 p 로 하고, 그 원칙은 Methods 통계 절에 한 번 적는다 (표마다 각주를 달지 않는다).
    #   옛 판 (한 자리 더 · 최대 넷째) : _이전판/_공통_전_CI자릿수고정_261025.py
    음수 = round(lo, 자릿) < 0 or round(hi, 자릿) < 0       # ★ 10-25 : 보이는 값 기준 (−0·001 은 "0·00" 으로 보이므로 en dash)
    return f"{점(lo, 자릿)} to {점(hi, 자릿)}" if 음수 else f"{점(lo, 자릿)}–{점(hi, 자릿)}"


def 값CI(v, lo, hi, 자릿=2):
    """추정치 칸 '1·29 (1·23–1·36)'."""
    return f"{점(v, 자릿)} ({구간(lo, hi, 자릿)})"


def p표(p):
    """p → 유효숫자 둘 · 0·0001 미만은 '<0·0001' (가운데 점)."""
    if p != p: return "··"
    if p < 0.0001: return "<0·0001"
    return ("%#.2g" % p).rstrip(".").replace(".", "·")


def p글(p):
    """'(p<0·0001)' · '(p=0·98)' 꼴 — 칸 안에서 판정 뒤에 붙인다 (10-26)."""
    t = p표(p); return f"p{t}" if t.startswith("<") else f"p={t}"


def 홀름(ps):
    """Holm 보정 p (같은 순서로 돌려준다)."""
    ps = np.asarray(ps, float); o = np.argsort(ps); k = len(ps); a = np.empty(k); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (k - i) * ps[j]); a[j] = min(1.0, run)
    return a


S10 = tuple(range(1, 11))
X2칸 = (뿌리 / "main/results/anchor", "an_mnl_anchorbm_K40_auto_scca_seed{s}")   # MobileNetV3-L S1 + S2 + S4 + S5


def 칸찾기(b, a):
    return X2칸 if (b, a) == ("mnl", "anchorbm") else 칸[b][a]


구성이름 = {"mse": "No safeguards", "normonly": "S1", "veminimalnoln": "S2", "veminimal": "S1 + S2",
         "anchor": "S1 + S2 + S4", "anchorshuf": "S1 + S2 + S4n (shuffled labels, negative control)",
         "batchmean": "S1 + S2 + S5", "anchorbm": "S1 + S2 + S4 + S5"}


# ── 엑셀 쓰기 ─────────────────────────────────────────────────────────────────
def 소제목표(T, 첫열="Measure"):
    """★ 10-26 (사용자) : 행 이름 "Indian location: MAE …" 을 소제목 줄 "Indian location" + 그 아래 "MAE …" 로 편다.
    ':' 가 없는 행은 그대로 둔다. 돌려주는 것 = (새 표, 소제목 이름들) — 엑셀(…, 소제목=이름들) 이 굵게 · 아래 줄 들여쓰기."""
    행, 머리, 지금 = [], [], None
    for _, r in T.iterrows():
        k = str(r[첫열])
        if ": " in k:
            h, rest = k.split(": ", 1)
            if h != 지금:
                지금 = h; 머리.append(h); 행.append({c: (h if c == 첫열 else "") for c in T.columns})
            r = r.copy(); r[첫열] = rest if rest == "p" or rest.startswith("p ") else rest[:1].upper() + rest[1:]   # p 는 소문자 그대로
        행.append(dict(r))
    return pd.DataFrame(행, columns=T.columns), 머리


def 엑셀(파일명, 어디, 시트들, 제목=None, 주석=None, 자릿=None, 소제목=None):
    """표를 .xlsx 로 쓴다.

    파일명 : 확장자 없이. 'Table3_external_backbones'
    어디   : 본문 또는 부록
    시트들 : {"Table 3": DataFrame, …}  — 순서가 시트 순서다
    제목   : 1행에 들어가는 표 제목 (영문)
    주석   : 표 아래에 붙는 각주 줄들 (영문 리스트)
    자릿   : {열이름: 소수 자릿수} — 지정한 열만 엑셀 서식으로 자릿수를 맞춘다
    소제목 : 첫 열의 소제목 이름들 (`소제목표` 가 돌려준다) — 그 아래 줄 들여쓰기 (10-26 · 굵게 없음)
    """
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    길 = Path(어디) / f"{파일명}.xlsx"
    # 11-09 : 안전장치 조합 붙여 쓰기 (S1 + S2 + S3 → S1+S2+S3) — 모든 칸 · 열 이름 · 제목 · 각주 · 소제목
    시트들 = {k: df.rename(columns=붙여).map(붙여) for k, df in 시트들.items()}
    제목 = 붙여(제목); 주석 = [붙여(x) for x in 주석] if 주석 else 주석; 소제목 = [붙여(x) for x in 소제목] if 소제목 else 소제목
    with pd.ExcelWriter(길, engine="openpyxl") as w:
        for 시트, df in 시트들.items():
            머리 = 1 + (2 if 제목 else 0)
            df.to_excel(w, sheet_name=시트[:31], index=False, startrow=머리)
            ws = w.sheets[시트[:31]]

            if 제목:
                ws.cell(1, 1, 제목).font = Font(bold=True, size=12)
                ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(df.columns), 2))

            얇 = Side(style="thin", color="000000")
            for j, c in enumerate(df.columns, 1):
                h = ws.cell(머리 + 1, j)
                # ★ 10-26 (사용자) : 굵게 · 회색 칠 없음 (학술지가 다시 조판한다 · 꾸밈 없는 표) — 위아래 선만
                h.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
                h.border = Border(top=얇, bottom=얇)
                폭 = max([len(str(c))] + [len(str(v)) for v in df[c].head(60)])
                ws.column_dimensions[get_column_letter(j)].width = min(max(폭 + 3, 11), 46)
                if 자릿 and c in 자릿:
                    for i in range(len(df)):
                        ws.cell(머리 + 2 + i, j).number_format = "0." + "0" * 자릿[c]

            if 소제목:                                       # 소제목 줄은 들여쓰지 않고 · 그 아래 줄만 첫 칸 들여쓰기 (굵게 없음 · 10-26)
                안 = False
                for i in range(len(df)):
                    c0 = ws.cell(머리 + 2 + i, 1)
                    if c0.value in 소제목: 안 = True
                    elif 안: c0.alignment = Alignment(indent=1)

            for j in range(1, len(df.columns) + 1):
                ws.cell(머리 + 1 + len(df), j).border = Border(bottom=얇)

            if 주석:
                r = 머리 + len(df) + 3
                ws.cell(r, 1, "Notes").font = Font(bold=True)
                for k, 줄 in enumerate(주석, 1):
                    ws.cell(r + k, 1, 줄).alignment = Alignment(wrap_text=False)
            ws.freeze_panes = ws.cell(머리 + 2, 1)
    print(f"  ✅ {길.relative_to(뿌리)}")
    return 길


def 읽기(이름):
    """main_results 의 CSV 를 BOM·공백 정리해서 읽는다. 없으면 분명하게 멈춘다."""
    길 = MR / (이름 if 이름.endswith(".csv") else 이름 + ".csv")
    if not 길.exists(): sys.exit(f"🔴 1차 자료가 없다 : {길.relative_to(뿌리)}")
    d = pd.read_csv(길)
    d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].astype(str).str.strip()
    return d


def 바꿔(d, 이름표, 남길=None):
    """열 이름을 영문으로 바꾼다. 이름표에 없는 열이 있으면 멈춘다(조용히 빠지지 않게)."""
    모름 = [c for c in d.columns if c not in 이름표]
    if 모름: sys.exit(f"🔴 영문 이름을 안 준 열이 있다 : {모름}")
    out = d.rename(columns=이름표)
    return out[[이름표[c] for c in (남길 or d.columns)]]


def 내부Hb():
    m = pd.read_csv(자료 / "internal/명부.csv")
    return m.groupby("number").Hb.first()


def 외부명부():
    return pd.read_csv(뿌리 / "main/external/data/arm_ext_auto_scca.csv")


def 시작(무엇):
    print(f"■ {무엇}", flush=True)

# ══ 11-04 보고 단위 (지침 맨 위 칸 · §5-1) ═══════════════════════════════════════════════
#  외부 : ① 같은 30명으로 미세조정한 10겹 모델의 예측을 사람마다 평균 (앙상블) → ② 절대오차 → ③ 그 사람이 평가된 추첨끼리 평균
#         → ④ 참여자 평균 = seed 의 MAE 하나 → ⑤ seed 10개의 중간값 (최소–최대)
#  개발 : 앙상블이 아니라 seed 마다 10겹 교차검증 MAE (두 눈 예측 평균) · 같은 형식으로 요약
#  CI · p : 참여자 단위 짝지은 bootstrap 100,000 — 뽑힌 사람으로 seed MAE 10개를 다시 내어 중간값 → 두 중간값의 차이
BOOT = int(os.environ.get("표_BOOT", 100000)); _청크 = 2000
빈칸허용 = os.environ.get("표_빈칸허용") == "1"     # 미리 보기 전용 : 끝나지 않은 칸을 건너뛴다 (실제 실행에서는 켜지 않는다)
if os.environ.get("표_백본"): BB = [b for b in BB if b in os.environ["표_백본"].split()]
# 🔵 권유 모델 — 2e-5 : SS (옛 판) · 1e-4 : `main_results/구성선택_lr1e-4.csv` 의 1위 (여덟 구성 중 두 기관 중 나쁜 쪽 MAE 최소 · 11-07)
#    ST10 이 §6-2 규칙 결과를 찍어 이 값과 같은지 확인한다
if FTB == "few_bn": 권유 = ("mnl", "veminimal", "few_bn")
elif (MR / "구성선택_lr1e-4.csv").exists():
    _선 = pd.read_csv(MR / "구성선택_lr1e-4.csv", encoding="utf-8-sig").iloc[0]; 권유 = ("mnl", str(_선.HEAD), FTB if int(_선.BN) else FT)
else:
    권유 = ("mnl", "veminimal", FTB); print("⚠️ 구성선택_lr1e-4.csv 가 아직 없다 — 권유를 S1 + S2 + S3 로 두고 간다 (L08 뒤 자동으로 생긴다)", flush=True)
def 구성글(a, 폴더): return 구성이름[a] + (" + S3" if str(폴더).endswith("_bn") else "")


새번호표 = {"ST1": "S1", "ST13": "S2", "ST2": "S3", "ST10": "S4", "ST5": "S5", "ST6": "S6", "ST4": "S7", "ST3": "S8", "ST7": "S9", "ST8": "S10", "SF2": "S2"}


def 붙여(x):
    """11-09 사용자 : 안전장치 조합은 붙여 쓴다 (S1 + S2 + S3 → S1+S2+S3 · 단어 수). 표의 모든 글자에 엑셀() 이 적용한다.
    ★ 11-11 : 부록 번호를 원고의 새 번호 (처음 인용 순서 · S1–S10 · 그림 S2) 로 바꾼다. 파일 이름은 그대로. ST12 · ST14 는 원고에 없어 그대로."""
    if not isinstance(x, str): return x
    x = re.sub(r"(S\d)\s\+\s(?=S\d)", r"\1+", x)
    return re.sub(r"\b(ST1[03]|ST[1-8]|SF2)\b", lambda m: 새번호표[m.group(1)], x)
학습률글 = {"2e-5": "2 × 10⁻⁵", "1e-4": "1 × 10⁻⁴"}.get(보정lr, 보정lr)
칸글 = "MAE, g/dL, median (seed range) [95% CI]"
_캐시 = MR / "_앙상블값"


def _틀(b, a, 입력):
    뿌_, 틀 = 칸찾기(b, a); return 뿌_, 틀.replace("auto_scca", 입력)


def _읽어쌓기(길, 만들기):
    """캐시 csv 가 있으면 읽고, 없으면 만들어 저장한다 (만들기 → None 이면 파일이 아직 다 없다 · 저장하지 않는다)."""
    if 길.exists(): return pd.read_csv(길)
    d = 만들기()
    if d is not None: d.to_csv(길, index=False)
    return d


def 외부값(b, a, 폴더, site, n=30, 입력="auto_scca"):
    """긴 표 (seed · key · y · e 앙상블 절대오차 · d 실측 − 앙상블 예측 · p 앙상블 예측 · e1 모델 1개 절대오차). 하나라도 없으면 None."""
    뿌_, 틀 = _틀(b, a, 입력); 길 = _캐시 / f"외부__{뿌_.name}__{틀.format(s='')}__{폴더}__{site}__n{n}.csv"

    def 만들기():
        쌓 = []
        for s in S10:
            for r in range(10):
                fs = [뿌_ / "external" / 폴더 / site / f"n{n}" / f"rep{r}" / 틀.format(s=s) / f"fold{f}.csv" for f in range(CV)]
                if not all(x.exists() for x in fs): return None
                d = pd.concat([pd.read_csv(x, usecols=["key", "y", "pred"]) for x in fs]); d["e1"] = (d.pred - d.y).abs()
                g = d.groupby("key").agg(y=("y", "first"), p=("pred", "mean"), e1=("e1", "mean"), m=("pred", "size"))
                if (g.m != CV).any(): sys.exit(f"🔴 앙상블 겹 수가 10 이 아니다 : {틀.format(s=s)} {site} rep{r}")
                g["seed"] = s; g["e"] = (g.p - g.y).abs(); g["d"] = g.y - g.p; 쌓.append(g.reset_index())
        return (pd.concat(쌓).groupby(["seed", "key"]).agg(y=("y", "first"), e=("e", "mean"), d=("d", "mean"), p=("p", "mean"), e1=("e1", "mean"))
                .reset_index())
    return _읽어쌓기(길, 만들기)


def 무보정값(b, a, site, 입력="auto_scca"):
    """미세조정 전 — 10겹 모델 예측 평균 (seed · key · y · e · d · p · e1). 그 기관 전원."""
    뿌_, 틀 = _틀(b, a, 입력); 길 = _캐시 / f"전__{뿌_.name}__{틀.format(s='')}__{site}.csv"

    def 만들기():
        쌓 = []
        for s in S10:
            fs = [뿌_ / "external/zero" / 틀.format(s=s) / f"fold{f}.csv" for f in range(CV)]
            if not all(x.exists() for x in fs): return None
            d = pd.concat([pd.read_csv(x) for x in fs]); d = d[d.Country == site]; d["e1"] = (d.pred - d.y).abs()
            g = d.groupby("key").agg(y=("y", "first"), p=("pred", "mean"), e1=("e1", "mean"))
            g["seed"] = s; g["e"] = (g.p - g.y).abs(); g["d"] = g.y - g.p; 쌓.append(g.reset_index())
        return pd.concat(쌓)[["seed", "key", "y", "e", "d", "p", "e1"]]
    return _읽어쌓기(길, 만들기)


_번호 = None
def 내부값(b, a, 입력="auto_scca"):
    """개발 교차검증 (seed · key=연구 번호 · y · e 두 눈 평균 절대오차 · p · e1 한 눈). 🔒 연구 번호만 저장한다."""
    global _번호
    뿌_, 틀 = _틀(b, a, 입력); 길 = _캐시 / f"개발__{뿌_.name}__{틀.format(s='')}.csv"

    def 만들기():
        global _번호
        if _번호 is None:
            m = pd.read_csv(자료 / "internal/명부.csv", usecols=["stem", "number"]); _번호 = dict(zip(m.stem, m.number))
        쌓 = []
        for s in S10:
            fs = [뿌_ / "train" / f"{틀.format(s=s)}-fold{f}" / "test_pred.csv" for f in range(CV)]
            if not all(x.exists() for x in fs): return None
            d = pd.concat([pd.read_csv(x) for x in fs]); d["key"] = d.file_path.map(lambda x: _번호.get(Path(x).stem))
            d = d.dropna(subset=["key"]); d["e1"] = (d.pred_best - d.y).abs()
            g = d.groupby("key").agg(y=("y", "first"), p=("pred_best", "mean"), e1=("e1", "mean"))
            g["seed"] = s; g["e"] = (g.p - g.y).abs(); g["d"] = g.y - g.p; 쌓.append(g.reset_index())
        return pd.concat(쌓)[["seed", "key", "y", "e", "d", "p", "e1"]]
    return _읽어쌓기(길, 만들기)


def 행렬(d, 값="e"):
    """긴 표 → 참여자 × seed 표. seed 10개가 다 있어야 한다. d 가 None 이면 None."""
    if d is None: return None
    M = d.pivot(index="key", columns="seed", values=값).sort_index()
    if M.shape[1] != len(S10) or M.isna().any().any(): sys.exit(f"🔴 참여자 × seed 표에 빈칸 : {M.shape}")
    return M


def 필수(v, 무엇):
    """없으면 멈춘다 (미리 보기에서는 그대로 None)."""
    if v is None and not 빈칸허용: sys.exit(f"🔴 빈칸 : {무엇}")
    return v


def 외부앙상블(b, a, 폴더, site, n=30, 입력="auto_scca", 값="e"):
    return 필수(행렬(외부값(b, a, 폴더, site, n, 입력), 값), f"{b} {a} {입력} {폴더} {site} n{n}")


def 무보정앙상블(b, a, site, 입력="auto_scca", 값="e"):
    return 필수(행렬(무보정값(b, a, site, 입력), 값), f"{b} {a} {입력} {site} 미세조정 전")


def 내부교차(b, a, 입력="auto_scca", 값="e"):
    return 필수(행렬(내부값(b, a, 입력), 값), f"{b} {a} {입력} 개발")


def 기준선(site):
    """사진 없는 기준선 — 추첨마다 현지 30명의 평균 Hb 를 평가 참여자 모두에게 예측. 참여자 × seed 표 (seed 와 무관하므로 열이 모두 같다)."""
    SP = 결과 / "external/splits"; z = 무보정값("mnl", "veminimal", site)
    hb = z[z.seed == 1].set_index("key").y; 오 = {}
    for r in range(10):
        cal = pd.read_csv(SP / site / f"rep{r}" / "calib_random.csv").key[:30]; ev = pd.read_csv(SP / site / f"rep{r}" / "eval.csv").key
        c = float(hb.loc[cal].mean())
        for k in ev: 오.setdefault(k, []).append(abs(c - float(hb.loc[k])))
    v = pd.Series({k: float(np.mean(x)) for k, x in 오.items()}).sort_index()
    return pd.DataFrame({s: v for s in S10})


_W = {}
def 재표집(n):
    """참여자 재표집 횟수 표 (B × n) · seed 0 고정 · 같은 n 이면 모든 비교가 같은 재표집을 쓴다 (짝)."""
    if n not in _W:
        rng = np.random.default_rng(0)
        _W[n] = np.vstack([rng.multinomial(n, np.full(n, 1 / n), size=min(_청크, BOOT - i)).astype(np.uint16) for i in range(0, BOOT, _청크)])
    return _W[n]


def 분포(M):
    """참여자 × seed 값 → 재표집마다 seed 값들의 평균 (B × seed)."""
    M = np.asarray(M, float); n = len(M); W = 재표집(n)
    return np.vstack([W[i:i + _청크] @ M / n for i in range(0, len(W), _청크)])


def 요약(M):
    """(중간값, seed 최소, seed 최대, CI 하한, CI 상한). CI = 재표집마다 낸 seed 중간값의 2·5–97·5 백분위."""
    v = np.asarray(M, float).mean(0); b = np.median(분포(M), axis=1)
    return float(np.median(v)), float(v.min()), float(v.max()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def 칸값(M, 자릿=2):
    """'1·45 (1·40–1·49) [1·23–1·67]' = 중간값 (seed 범위) [95 % CI]. 없으면 '··'."""
    if M is None: return "··"
    m, a, b, lo, hi = 요약(M)
    return f"{점(m, 자릿)} ({구간(a, b, 자릿)}) [{구간(lo, hi, 자릿)}]"


def 짝(A, B, 마진=None):
    """A − B : 두 중간값의 차이 · 95 % CI · 양측 p · (마진이 있으면) 단측 p = 차이 ≥ 마진 인 비율. 같은 참여자만."""
    c = A.index.intersection(B.index)
    if len(c) != len(A) or len(c) != len(B): sys.exit(f"🔴 짝 비교의 참여자가 다르다 : {len(A)} · {len(B)} · 공통 {len(c)}")
    X, Y = A.loc[c].to_numpy(float), B.loc[c].to_numpy(float)
    d = np.median(분포(X), axis=1) - np.median(분포(Y), axis=1)
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    점추정 = float(np.median(X.mean(0)) - np.median(Y.mean(0)))
    출 = (점추정, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p)
    return 출 + (float((d >= 마진).mean()),) if 마진 is not None else 출


def 대비(C):
    """참여자 × seed 대비 값 (주효과 · 상호작용) → seed 값의 중간값 · 95 % CI · 양측 p."""
    C = np.asarray(C, float); d = np.median(분포(C), axis=1)
    return float(np.median(C.mean(0))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def 차이글(x):
    return f"{점(x[0], 부호=True)} ({구간(x[1], x[2])})"


def AUROC요약(pos, S, S0=None):
    """seed 마다 AUROC → 중간값 · 범위 · 참여자 bootstrap CI. S0 (미세조정 전 점수) 가 있으면 그 요약과 후 − 전 의 양측 p 도."""
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool); n = len(pos); rng = np.random.default_rng(0)

    def a(M, i):
        y = pos[i]; r = rankdata(M[:, i], axis=1); n1 = y.sum(); n0 = n - n1
        return (r[:, y].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    m1, m0 = np.empty(BOOT), np.empty(BOOT)
    for k in range(BOOT):
        i = rng.integers(0, n, n); m1[k] = np.median(a(S, i))
        if S0 is not None: m0[k] = np.median(a(S0, i))
    i0 = np.arange(n); v = a(S, i0); 출 = (float(np.median(v)), float(v.min()), float(v.max()), *map(float, np.percentile(m1, [2.5, 97.5])))
    if S0 is None: return 출, None
    v0 = a(S0, i0); d = m1 - m0
    return 출, (float(np.median(v0)), float(v0.min()), float(v0.max()), *map(float, np.percentile(m0, [2.5, 97.5])), float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def AUROC칸(x, 자릿=3):
    return f"{점(x[0], 자릿)} ({구간(x[1], x[2], 자릿)}) [{구간(x[3], x[4], 자릿)}]"


보고각주 = [
    "Values are median (range across 10 seeds) [95% CI].",
    "External locations: for each seed, the 10 fold models were fine-tuned on the same local participants and their predictions were",
    "  averaged for each participant; the absolute error was averaged over the draws in which that participant was evaluated,",
    "  and then over participants, to give one MAE per seed. Development cohort: 10-fold cross-validation MAE per seed, grouped by",
    "  participant, with each participant's two eye predictions averaged; it is not an ensemble.",
    "95% CIs and two-sided p values are from a paired participant bootstrap (100,000 resamples): participants were resampled, the 10",
    "  seed MAEs were recomputed and summarised as the median, and medians were compared within the same resample.",
]


# ── 입력 비교 칸 (Table 4 · ST7 공통 · 11-09 번호 바꿈) ─────────────────────────────────────────────
입력열 = {"auto_scca": "SCCA, automatic (reference)", "human_scca": "SCCA, manual", "auto_bc": "Conjunctiva, automatic",
         "human_bc": "Conjunctiva, manual", "auto_bcwb": "Conjunctiva (white-balanced), automatic"}
짝사람 = {"human_scca": "auto_scca", "human_bc": "auto_bc"}
마진 = 0.089
비열등줄 = "automatic vs manual, same region: non-inferiority (δ = 0·089 g/dL)"


def 입력블록(열, 묶음, D, 비열등판정=True):
    """열 = {입력: {줄: 칸}} 에 한 블록을 채운다. D = {입력: 참여자 × seed}. 비열등은 개발 · 미세조정 뒤에만 (§14)."""
    기 = D["auto_scca"]
    for g in 열:
        열[g][f"{묶음}{칸글}"] = 칸값(D[g])
        열[g][f"{묶음}p value vs reference"] = "··" if g == "auto_scca" or D[g] is None or 기 is None else p표(짝(D[g], 기)[3])
        if not 비열등판정: continue
        차, 판 = "··", "··"
        if g in 짝사람 and D[g] is not None and D[짝사람[g]] is not None:
            x = 짝(D[짝사람[g]], D[g], 마진); 차 = 차이글(x)              # 자동 − 사람
            판 = f"{'Non-inferior' if x[2] < 마진 else 'Not shown'} ({p글(x[4])})"
            print(f"    비열등 {묶음}{g} : {차} · 단측 p {p표(x[4])}", flush=True)
        열[g][f"{묶음}automatic minus manual, same region, g/dL (95% CI)"] = 차; 열[g][f"{묶음}{비열등줄}"] = 판


# ── 안전장치 ablation 표 (Table 3 = MobileNetV3-L (11-09 번호 바꿈) · ST3 = 다른 넷 · 같은 모양) ─────────────
def 안전장치표(순서, 기준선줄=False):
    팔 = {"mse": (0, 0), "normonly": (1, 0), "veminimalnoln": (0, 1), "veminimal": (1, 1)}

    def 구성명(i, j, k):
        on = [n for n, v in zip(["S1", "S2", "S3"], (i, j, k)) if v]
        return "No safeguards" if not on else " + ".join(on)
    P3 = 읽기("P3_추론비용").set_index("백본"); 이 = {s: 영문사이트[s] for s in SITES}
    빈 = lambda: {"S1": "", "S2": "", "S3": "", "Development cohort": "··",
                 **{f"{이[s]}: {c}": "··" for s in SITES for c in ("before fine-tuning", "after fine-tuning", "p value")}}
    행, 머리 = [], []
    for b in 순서:
        K = (0,) if BN층[b] == 0 else (0, 1)
        제 = (f"{표시[b]} ({BN층[b]} batch-normalisation layers; {점(float(P3.loc[표시[b], '배포파라미터']) / 1e6, 2)} million deployed "
             f"parameters; {점(float(P3.loc[표시[b], 'GMACs']), 3)} GMACs per image)")
        if BN층[b] == 0: 제 = 제.replace("(0 batch-normalisation layers;", "(no batch normalisation;")
        머리.append(제); 행.append({"Configuration": 제})
        if 기준선줄:   # 11-09 공저자 합의 : C1 (사진 없는 기준선) 을 본문 표에서 보인다 · 같은 평가 참여자 · ST5 와 같은 값
            r = {"Configuration": "Photo-free baseline (mean haemoglobin of the 30 local fine-tuning participants)", **빈()}
            for s in SITES: r[f"{이[s]}: after fine-tuning"] = 칸값(기준선(s))
            행.append(r)
        y = {s: {} for s in SITES}; 개발값 = {}; 구p = []
        for a, (i, j) in 팔.items():
            개발값[a] = 내부교차(b, a)
            z = {s: 무보정앙상블(b, a, s) for s in SITES}
            for k in K:
                r = {"Configuration": 구성명(i, j, k), "S1": "O" if i else "", "S2": "O" if j else "", "S3": "O" if k else "",
                     "Development cohort": "··" if k else 칸값(개발값[a])}
                for s in SITES:
                    u = 외부앙상블(b, a, FTB if k else FT, s); y[s][(i, j, k)] = u
                    r[f"{이[s]}: before fine-tuning"] = "··" if k else 칸값(z[s])
                    r[f"{이[s]}: after fine-tuning"] = 칸값(u)
                    구p.append((r, f"{이[s]}: p value", 짝(u, z[s])[3]))
                행.append(r)
                print(f"  {표시[b]:16} {r['Configuration']:20} " + " | ".join(str(v) for c, v in r.items() if ":" in c), flush=True)
        for (rr, 열, _), hp in zip(구p, 홀름([x[2] for x in 구p])): rr[열] = p표(hp)
        효 = {}
        for s in SITES:
            Y = {key: v.to_numpy(float) for key, v in y[s].items()}
            효[(s, "S1")] = 대비(np.mean([Y[(1, j, k)] - Y[(0, j, k)] for j in (0, 1) for k in K], axis=0))
            효[(s, "S2")] = 대비(np.mean([Y[(i, 1, k)] - Y[(i, 0, k)] for i in (0, 1) for k in K], axis=0))
            if len(K) == 2: 효[(s, "S3")] = 대비(np.mean([Y[(i, j, 1)] - Y[(i, j, 0)] for i in (0, 1) for j in (0, 1)], axis=0))
        h = dict(zip(효, 홀름([v[3] for v in 효.values()])))
        for 요인, 설명 in [("S1", "final-feature normalisation"), ("S2", "adaptation layer"), ("S3", "batch-normalisation statistics held")]:
            if (SITES[0], 요인) not in 효: continue
            r = {"Configuration": f"Main effect of {요인} ({설명}), on vs off: change in MAE, g/dL (95% CI); Holm-adjusted p value (unadjusted p value)", **빈()}
            for s in SITES:
                r[f"{이[s]}: after fine-tuning"] = 차이글(효[(s, 요인)]); r[f"{이[s]}: p value"] = f"{p표(h[(s, 요인)])} ({p표(효[(s, 요인)][3])})"   # 11-08 사용자 : Holm 과 보정 전 p 를 함께
            행.append(r)
            print(f"  {표시[b]:16} 주효과 {요인} " + " | ".join(f"{s} {효[(s, 요인)][0]:+.3f} Holm {h[(s, 요인)]:.2g} (보정 전 {효[(s, 요인)][3]:.2g})" for s in SITES), flush=True)
        r = {"Configuration": "p value, S1 + S2 vs no safeguards (development cohort)", **빈()}
        r["Development cohort"] = p표(짝(개발값["veminimal"], 개발값["mse"])[3]); 행.append(r)
        ss = "S1 + S2 + S3" if len(K) == 2 else "S1 + S2"
        r = {"Configuration": f"p value, {ss} vs no safeguards, both after fine-tuning", **빈()}
        for s in SITES: r[f"{이[s]}: p value"] = p표(짝(y[s][(1, 1, K[-1])], y[s][(0, 0, 0)])[3])
        행.append(r)
    return pd.DataFrame(행).fillna(""), 머리


def 안전장치표_각주(MobileNet만):
    return 보고각주 + [
      "Configuration rows: the p value compares after with before fine-tuning of the same model, Holm-adjusted within the backbone over",
      "  these comparisons (16 tests; 8 for EdgeNeXt-XS)." if not MobileNet만 else "  these comparisons (16 tests).",
      "Main-effect rows: the change in error when the safeguard is switched on, averaged over every combination of the other safeguards",
      "  for each participant and seed, and summarised as the median across seeds; negative values mean a lower error. p values are",
      "  Holm-adjusted over the main-effect tests of each backbone (three safeguards x two locations"
      + ("" if MobileNet만 else "; four tests for EdgeNeXt-XS") + "); the unadjusted p value is shown in brackets. Interactions: appendix.",
      "Before fine-tuning: all participants (Indian 95, Italian 122). After fine-tuning: the 30 fine-tuning participants were drawn at random",
      "  10 times from 50 candidates, and each time the participants not among the 50 candidates were evaluated (Indian 45, Italian 72).",
      "S1, final-feature layer normalisation; S2, adaptation layer; S3, the backbone's batch-normalisation statistics held at their",
      "  development values during fine-tuning (otherwise they are updated). S3 acts only during fine-tuning, so",
      "  development and before-fine-tuning values of a configuration with S3 are those of the row above (··)."
      + ("" if MobileNet만 else " EdgeNeXt-XS contains no batch normalisation, so S3 does not apply."),
      *(["Photo-free baseline: the mean haemoglobin of the same 30 fine-tuning participants predicted for every evaluation participant of",
         "  that draw, evaluated in the same participants as after fine-tuning (shown in the after-fine-tuning column; comparisons in table ST5).",
         "Values for each of the ten seeds of the recommended model are given in table ST8."]
        if MobileNet만 else []),
      "Deployed parameters exclude unused classifier layers and fold the adaptation layer into the regression head; GMACs,",
      "  multiply-accumulate operations for one 224 x 224 image.",
      ("The other four backbones are shown in the appendix in the same format." if MobileNet만 else "MobileNetV3-Large is shown in table 3."),
      "Results are reported by location and are not pooled.",
    ]
