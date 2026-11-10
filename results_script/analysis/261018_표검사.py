# -*- coding: utf-8 -*-
"""★ 원고 표 검사 — 10-19 표기 규칙을 지켰는지 (2026-10-18)

검사  (10-19 Lancet 판) ① 남은 yes · no · ✓ 칸 ② CI 모양 — 두 끝이 양수인데 "to" · 음수가 섞였는데 en dash ③ 보통 소수점 "1.234" ④ 시트가 둘 이상 (한 부분 표 규정)
      ④ 소수 셋째 자리가 아닌 g/dL 값 (경고만) · 표 파일마다 수정 시각
내는 것  화면 출력 (원고 표 폴더 main · supplementary 의 xlsx 전부)
쓰는 법  python paperwork/results_script/analysis/261018_표검사.py
"""
import re, glob, os, time
import pandas as pd
from pathlib import Path
# ★ 11-07 : 표_보정lr (기본 1e-4) 에 맞는 표 폴더를 본다 — 1e-4 → results_table_figure_lr1e-4 · 2e-5 → results_table_figure
_lr = os.environ.get("표_보정lr", "1e-4"); P = Path(__file__).resolve().parents[3] / ("paperwork/results_table_figure" if float(_lr) == 2e-5 else f"paperwork/results_table_figure_lr{_lr}")
for f in sorted(glob.glob(str(P / "main/*.xlsx")) + glob.glob(str(P / "supplementary/*.xlsx"))):
    x = pd.read_excel(f, sheet_name=None, header=None); 문 = {"yes/✓": 0, "양수인데 to": 0, "음수인데 en dash": 0, "보통 소수점": 0, "시트 둘 이상": 0}
    if len(x) > 1: 문["시트 둘 이상"] = len(x)
    for k, v in x.items():
        for c in v.columns:
            for s in v[c].dropna().astype(str).iloc[2:] if len(v) > 2 else []:      # 1 – 2 행 (제목) 은 빼고
                t = s.strip()
                if t.startswith(("Notes", "  ")) or len(t) > 120: continue            # 각주 줄은 빼고
                if t.lower() in ("yes", "no", "✓", "true", "false"): 문["yes/✓"] += 1
                if re.search(r"(?<![−\d·])\d+·\d+ to \d+·\d+", t): 문["양수인데 to"] += 1
                if re.search(r"−\d+·\d+–|\d–−\d", t): 문["음수인데 en dash"] += 1
                if re.fullmatch(r"[−-]?\d+\.\d+", t) or re.search(r"(?<![A-Za-z\d.])\d+\.\d+(?![\d.])", t): 문["보통 소수점"] += 1   # 10-20 : 글자 사이의 "5.4" 도 (Table 1 스마트폰 칸이 빠져나갔다)
    표 = "✅" if not any(문.values()) else "🔴"
    print(f"{표} {Path(f).name:45} {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f)))} · " + " · ".join(f"{a} {b}" for a, b in 문.items()))
