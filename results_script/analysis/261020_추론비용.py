# -*- coding: utf-8 -*-
"""★ 다섯 백본의 추론 비용 — 배포 파라미터 · GMACs (224 × 224 한 장) · CPU 한 장 지연  2026-10-20

한 목적  Table 4 의 "Deployed parameters" · "Inference cost, GMACs" 줄과 Table ST10 의 GMACs 열.
원래     main/analysis/261004_선택논리.py 의 `추론비용` 하위 명령 (그 파일은 옛 51조합 선택 넷 + 이것 = 목적 다섯이라
         10-20 에 이것만 떼어 냈다 · 원본은 main/analysis/_이전판/). 계산은 글자 그대로 옮겼다.
원자료   각 백본의 seed1-fold0 config.yaml (모델 구조만 · 가중치는 읽지 않는다)
내는 것  main/results/main_results/P3_추론비용.csv
         🔵 배포파라미터 · GMACs 는 결정적이다. CPU1장_ms 는 돌릴 때마다 조금 다르다 (표에 쓰지 않는다).
쓰는 법  python paperwork/results_script/analysis/261020_추론비용.py   (CPU · 1분)
"""
import sys, glob, time
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parents[3]; MAIN = P / "main"
sys.path.insert(0, str(P / "paperwork/results_script")); sys.path.insert(0, str(P / "code"))
from _공통 import BB, 표시, MR   # noqa: E402

import torch, yaml
from types import SimpleNamespace
from torch.utils.flop_counter import FlopCounterMode
from model.backbone import create_backbone

torch.set_num_threads(1); rows = []
for b in BB:
    f = glob.glob(str(MAIN / f"results/{'최소구성' if b == 'mnl' else '백본재현'}/train/*_{b}_veminimal_K40_auto_scca_seed1-fold0/config.yaml"))[0]
    g = SimpleNamespace(**yaml.safe_load(open(f))); m, dim = create_backbone(g); m.eval(); x = torch.zeros(1, 3, 224, 224)
    쓴 = set(); hs = [mod.register_forward_hook(lambda mod, i, o: 쓴.add(id(mod)))
                      for mod in m.modules() if len(list(mod.children())) == 0]
    with FlopCounterMode(display=False) as fc, torch.no_grad(): m(x)
    for h in hs: h.remove()
    백 = sum(p.numel() for mod in m.modules() if id(mod) in 쓴 for p in mod.parameters(recurse=False))
    with torch.no_grad():
        for _ in range(5): m(x)
        t = time.perf_counter(); [m(x) for _ in range(30)]; ms = (time.perf_counter() - t) / 30 * 1000
    rows.append(dict(백본=표시[b], 배포파라미터=백 + 2 * dim + dim + 1, GMACs=round(fc.get_total_flops() / 2 / 1e9, 3),
                     CPU1장_ms=round(ms, 1)))
T = pd.DataFrame(rows); T.to_csv(MR / "P3_추론비용.csv", index=False, encoding="utf-8-sig"); print(T.to_string(index=False))
print("  ✅ main/results/main_results/P3_추론비용.csv")
