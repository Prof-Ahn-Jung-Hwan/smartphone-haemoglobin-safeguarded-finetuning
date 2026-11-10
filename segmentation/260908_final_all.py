# -*- coding: utf-8 -*-
"""★★ 확정 파이프라인을 **전수**(내부 1,479 + 외부 202)에 돌린다.  2026-09-08

무엇을 하나
    1단계  지금까지 쓰던 것 — BC 합집합 + 흰눈동자(또는 밝은영역) + 손질 넷
    가르기  1단계가 공막을 못 잡았는지 **사람 자를 안 쓰고** 판정한다
    2단계  못 잡은 장만 **처음부터 다시** — 낱말 `the eye surface next to the iris`
    다듬기  BC 와 영역 둘 다 **타원 1.06** 안으로 (목자르기는 뺐다 · 아래 참조)

왜 목자르기를 뺐나 (09-07 저녁 전수 확인)
    한 장에서 SCCA 의 **64 %** 를 날렸다. 공막은 홍채 좌우 두 덩어리인데 둘은 좁은 다리로만
    이어진다. 목자르기가 그 다리를 혹의 목으로 착각해 **한쪽 흰자를 통째로** 버린다.
    정작 이득은 그 장에서도 크기비 2.11 -> 2.00 뿐이었다. 흥정이 나쁘다.

SAM 을 장당 **두 번**만 부른다
    `the eye surface next to the iris` (2단계 영역) · `skin` (빗장 1)
    BC·홍채·윗속눈썹·동공은 260904_bc4 · 260903_final_v3 에 저장된 것을 읽는다 (전수 다 있다)

결과
    results/260908_final_all/
        mask/*.npz     ★ Hb 모델에 넣을 ROI 가 여기 있다
        장별_{안쪽,바깥쪽}.csv
        panel/*.jpg
        {reader}/review_{안쪽,바깥쪽}.html    ← 판독자별 폴더·저장키 (눈가림)

사용법
    conda activate torch271
    python analysis/260908_final_all.py --set internal
    python analysis/260908_final_all.py --set external
    # 다른 판독자용 페이지만 다시 만들 때 (SAM 안 돌림)
    python analysis/260908_final_all.py --set internal --reader secondary --pageonly
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from render_bc_overlay import polygon_mask, read_raw
from metrics_seg import bottom_line, below_frac
from build_review_final import BC_AXIS, SCCA_AXIS, TAGS          # ★ 판정 코드 단일 정의
from importlib import import_module
PF = import_module("260903_pipeline_final")
R4 = import_module("260904_bc_final_review")
SM = import_module("260907_smooth")
tint, outline, _font = R4.tint, R4.outline, R4._font

ROOT = HERE.parent
OLD  = Path("PATH/TO/PROJECT/2608_eClinicalMedicine_SCCA_imageOnly_filtered_2")
MASTER = Path("PATH/TO/PROJECT/external_validation_master.csv")
OUT  = ROOT / "results/260908_final_all"
MODEL_ID = "facebook/sam3"

P_REG, P_SKIN = "the eye surface next to the iris", "skin"
TAU   = 0.05        # 빗장 1 — 살갗이 BC 를 이보다 많이 물면 그 장에선 살갗을 안 쓴다
GATE  = 0.90        # 가르기 — 1단계 공막 넓이 / BC 넓이 가 이보다 작으면 2단계로
GROW  = 1.06        # 다듬기 — 맞춘 타원의 배수
SEED  = 20260908
READERS = {"primary":   dict(dir="primary",   key="final_all_260908"),
           "secondary": dict(dir="secondary", key="final_all_260908_2nd")}
C_BC, C_SC, C_CUT = (120, 255, 180), (0, 200, 255), (255, 60, 60)


def unpack(z, k, H, W):
    return np.unpackbits(z[k])[:H*W].reshape(H, W).astype(bool)


def build_one(z, z3, m2r, mas, H, W):
    """확정 셈틀. 사람 주석을 하나도 안 쓴다."""
    U0 = unpack(z, "bc_1", H, W) | unpack(z, "bc_2", H, W)      # BC = 붉은안쪽 ∪ 분홍안쪽
    w1 = unpack(z3, "w_fin", H, W)                               # 1단계 공막
    ir = unpack(z3, "iris", H, W); el = unpack(z3, "el_up", H, W)
    pu = unpack(z3, "pupil", H, W)
    yy = np.arange(H)[:, None]

    # ── 바닥 : BC 합집합의 열별 아래 margin (안전망 B — BC 가 홍채 위에 뜨면 안 건다)
    skipB = False
    if U0.sum() > 100 and ir.sum() > 100:
        skipB = (U0 & (yy < np.nonzero(ir.any(1))[0].min())).sum() > 0.5 * U0.sum()
    Lb = None if (U0.sum() <= 100 or skipB) else bottom_line(U0, W)
    fl = (yy <= Lb[None, :]) if Lb is not None else np.ones((H, W), bool)
    # ── 천장 : 홍채 윗끝 위는 부스러기라도 버린다 (빗장 2)
    ty = None if ir.sum() < 20 else float(np.nonzero(ir.any(1))[0].min())
    ce = np.ones((H, W), bool) if ty is None else (yy >= ty)
    # ── 빗장 1 : 살갗이 결막을 물면 그 장에선 살갗을 안 쓴다
    침범 = float((mas & U0).sum() / max(U0.sum(), 1))
    살갗씀 = 침범 <= TAU

    RAW = ((((m2r & ~ir) & ~el) & ~pu) & fl) & ce
    core = (RAW & ~mas) if 살갗씀 else RAW
    reg2 = PF.drop_small(core)
    reg2 = keep_touching(reg2, ir | U0, W)
    firedA = reg2.sum() < 0.05 * max(RAW.sum(), 1)                # 안전망 A
    if firedA:
        reg2 = PF.drop_small(RAW)

    # ── 🔴 1단계 결과에도 **똑같은 바닥·천장·닿은것만**을 건다
    #    v3 의 w_fin 은 낱말 하나(`b1`)의 바닥선으로 잘린 것이라
    #    **BC 합집합의 아래 margin** 이 안 걸려 있었다 (평균 2.1 % · 최대 25.4 % 가 그 아래).
    reg1 = keep_touching(PF.drop_small((((w1 & ~pu) & fl) & ce)), ir | U0, W)
    if reg1.sum() < 0.05 * max(w1.sum(), 1):                      # 안전망 — 무너지면 되돌린다
        reg1 = PF.drop_small(w1 & fl)

    # ── 가르기 : 1단계 공막이 BC 에 견주어 너무 작으면 2단계를 쓴다 (사람 자 안 씀)
    공막몫 = float(reg1.sum() / max(U0.sum(), 1))
    쓴단계 = 2 if 공막몫 < GATE else 1
    reg = reg2 if 쓴단계 == 2 else reg1

    # ── 다듬기 : BC 와 영역 둘 다 타원 안으로
    #    ★ 안전망 — 타원이 BC 를 20 % 넘게 깎으면 그 장에선 타원을 안 쓴다
    E = fit_ellipse_all(PF.drop_small(U0 | reg), H, W, GROW)
    타원씀 = (U0 & E).sum() >= 0.8 * max(U0.sum(), 1)
    if not 타원씀:
        E = np.ones((H, W), bool)
    U, R = U0 & E, reg & E
    # ── 빈 띠 메우기 : SCCA = 구멍메우기(BC ∪ 공막)  ← 08-30 설계
    scca0 = PF.drop_small(U | R)
    ys_, xs_ = np.nonzero(scca0) if scca0.any() else (np.array([0]), np.array([0]))
    eyeW = max(20, xs_.max() - xs_.min() + 1)
    br = bridge_columns(U, PF.drop_small(R & ~U), H, W) & ~ir      # ★ 열별 세로 잇기
    scca, 띠 = fill_gap(scca0, pu, eyeW, extra=br)
    return dict(BC=U, REG=R, SCCA=scca, SCLERA=PF.drop_small(scca & ~U), 띠=띠,
                BC0=U0, REG1=reg1, REG2=reg2, CUT=(U0 | reg) & ~E, IRIS=ir,
                쓴단계=쓴단계, 공막몫=공막몫, 살갗침범=침범, 살갗씀=bool(살갗씀),
                안전망A=bool(firedA), 안전망B=bool(skipB), 타원씀=bool(타원씀),
                띠몫=float(띠.sum()/max(scca.sum(), 1)))


def fit_ellipse_all(m, H, W, grow=1.06):
    """마스크 **전체**에 타원 하나를 맞춘다.

    🔴 260907_smooth 의 fit_ellipse 는 **제일 큰 덩어리 하나**에만 맞췄다.
       결막과 공막이 떨어져 있으면 한쪽에만 맞아 다른 쪽을 통째로 지운다
       (바깥쪽 맛보기에서 BC 덮음 0.901 -> 0.071). 그래서 여기서는 **모든 점**을 쓴다.
    """
    if m.sum() < 200:
        return np.ones((H, W), bool)
    # 🔴 fitEllipse 는 점들이 **타원 위**에 있다고 보고 맞춘다.
    #    속을 채운 점을 다 주면 타원이 덩어리보다 훨씬 작게 나온다. **테두리 점**을 줘야 한다.
    ct, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ct:
        return np.ones((H, W), bool)
    pts = np.vstack(ct)                                  # 모든 덩어리의 테두리를 다 쓴다
    if len(pts) < 5:
        return np.ones((H, W), bool)
    (cx, cy), (aw, ah), ang = cv2.fitEllipse(pts)
    o = np.zeros((H, W), np.uint8)
    cv2.ellipse(o, ((cx, cy), (aw*grow, ah*grow), ang), 1, -1)
    return o.astype(bool)


def fill_gap(m, pu, eyeW, extra=None, frac=0.02):
    """★ 공막과 결막 사이의 **빈 띠를 메운다** (08-30 설계 `SCCA = 구멍메우기(BC ∪ WC)`).

    사방이 눈으로 둘러싸인 자리는 피부일 수 없다. 안쪽 10.7 % · 바깥쪽 15.8 % 의 장에 생기고,
    그 띠의 28.7 % / 38.4 % 가 **진짜 사람 결막**이었다.

    🔴 `binary_fill_holes` 만으로는 **한 장도 안 메워진다** (전수 확인 0.0000).
       그 띠는 **양옆이 트여 있어 구멍이 아니기** 때문이다. 그래서 **닫기(closing)** 를 쓴다.
       닫기는 커널보다 좁은 오목한 자리를 메우므로 트인 띠도 메운다.
       커널 = 눈 너비의 2 %. 그때 더해진 것의 **64.5 %가 사람이 그린 눈 안**이었다
       (사람 주석이 조각이라 눈 전체를 안 덮는 걸 감안하면 강한 증거다).

    🔴 빗장 — 공막이 홍채 좌우를 감싸고 BC 가 아래를 받치면 **홍채가 갇힌 자리**가 된다.
       그걸 메우면 눈동자가 SCCA 에 들어온다. 그래서 **동공을 품은 조각은 안 메운다.**
    """
    if m.sum() == 0:
        return m, np.zeros_like(m)
    if extra is None:
        extra = np.zeros_like(m)
    k = max(3, int(round(eyeW * frac)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cl = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    hole = ((cl | ndi.binary_fill_holes(cl)) | extra) & ~m
    if not hole.any():
        return m, hole
    n, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8))
    keep = np.zeros_like(m)
    for v in range(1, n):
        c = (lab == v)
        if pu.sum() > 20 and (c & pu).any():        # 동공을 품었다 = 홍채다 -> 안 메운다
            continue
        keep |= c
    return m | keep, keep


def bridge_columns(U, R, H, W, maxfrac=0.30):
    """★ 열별 세로 잇기 — 같은 세로줄에서 **공막의 아랫면**과 **결막의 윗면** 사이가
    비어 있으면 그 사이를 채운다.

    🔴 닫기(closing)는 **커널보다 넓은 틈은 못 메운다.** 바깥쪽은 틈이 넓어 안 메워졌다.
       이 규칙은 틈 너비와 상관없이 **위아래로 마주보고 있으면** 잇는다.
       그 자리는 위가 공막이고 아래가 결막이므로 **눈일 수밖에 없다.**

    빗장 : 틈이 눈 높이의 maxfrac 을 넘으면 잇지 않는다 (마주본 게 아니라 딴 것이다)
    """
    band = np.zeros((H, W), bool)
    if U.sum() < 50 or R.sum() < 50:
        return band
    ys_u, xs_u = np.nonzero(U); ys_r, xs_r = np.nonzero(R)
    tu = np.full(W, 1e9); np.minimum.at(tu, xs_u, ys_u.astype(float))   # 결막 윗면
    br = np.full(W, -1.0);  np.maximum.at(br, xs_r, ys_r.astype(float)) # 공막 아랫면
    both = np.nonzero(U | R)[0] if False else None
    ys_a = np.nonzero((U | R).any(1))[0]
    eyeH = max(20, ys_a.max() - ys_a.min() + 1)
    ok = (tu < 1e9) & (br >= 0) & (tu > br + 1) & ((tu - br) <= maxfrac * eyeH)
    if not ok.any():
        return band
    yy = np.arange(H)[:, None]
    band = (yy > br[None, :]) & (yy < tu[None, :]) & ok[None, :]
    return band


def keep_touching(m, seed, W):
    """홍채·BC 에 닿은 덩어리만 남긴다. 떨어진 것은 공막도 결막도 아니다."""
    if m.sum() == 0 or seed.sum() == 0:
        return m
    k = max(3, int(round(W * 0.006)) * 2 + 1)
    sd = cv2.dilate(seed.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = np.zeros_like(m)
    for v in range(1, n):
        c = (lab == v)
        if (c & sd).any():
            out |= c
    return out


def page(recs, label, key, dst):
    """판정 화면 — 08-25 에 정한 두 축(BC·SCCA)과 태그를 그대로 쓴다."""
    def bar(items, cls, fn):
        return "".join(
            f'<button class="{cls}" data-c="{c}" style="--k:{col}" onclick="{fn}(\'{c}\')">'
            f'<b>{k.upper()}</b> {c}<small>{ko}</small></button>' for k, c, ko, col in items)
    tagbar = "".join(f'<button class="tg" id="tg_{c}" onclick="tog(\'{c}\')">'
                     f'<b>{k.upper()}</b> {ko}</button>' for k, c, ko in TAGS)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{label} · {len(recs)}장</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#131313;color:#e8e8e8;font-family:system-ui,'Noto Sans KR',sans-serif}}
header{{position:sticky;top:0;background:#1c1c1c;padding:6px 12px;border-bottom:1px solid #333;z-index:9}}
h1{{font-size:14px;margin:0}} .sub{{font-size:12px;color:#9a9a9a}}
#bar{{height:3px;background:#2a2a2a;margin-top:5px}} #barf{{height:100%;background:#4caf50;width:0}}
#wrap{{padding:6px 12px 178px;text-align:center}}
img{{max-width:100%;max-height:66vh;border-radius:3px;background:#000}}
body.zoom img{{max-height:none;width:100%}}
footer{{position:fixed;bottom:0;left:0;right:0;background:#1c1c1c;border-top:1px solid #333;padding:6px 12px}}
.row{{display:flex;align-items:center;gap:6px;margin:3px 0;flex-wrap:wrap}}
.tit{{font-size:12px;color:#777;width:46px;flex:none}}
.b1,.b2{{background:#262626;border:1px solid #3a3a3a;border-left:4px solid var(--k);color:#e8e8e8;
   padding:4px 6px;border-radius:4px;cursor:pointer;font-size:11.5px;white-space:nowrap}}
.b1:hover,.b2:hover{{background:#333}}
.b1.on,.b2.on{{background:var(--k);color:#fff;border-color:var(--k)}}
.b1 b,.b2 b{{color:var(--k);margin-right:4px}} .b1.on b,.b2.on b{{color:#fff}}
.b1 small,.b2 small{{color:#8f8f8f;margin-left:4px}} .b1.on small,.b2.on small{{color:#eee}}
.tg{{background:#242424;border:1px solid #3a3a3a;color:#bbb;padding:4px 7px;border-radius:4px;
   cursor:pointer;font-size:11.5px}}
.tg.on{{background:#ffab40;color:#222;border-color:#ffab40;font-weight:600}}
input#note{{background:#222;border:1px solid #3a3a3a;color:#eee;padding:6px;border-radius:4px;
   width:100%;font-size:13px}}
input#note:focus{{border-color:#ffab40;outline:none}}
button.x{{background:#333;border:1px solid #4a4a4a;color:#ddd;padding:5px 10px;border-radius:4px;
   cursor:pointer;margin-left:5px;font-size:12px}}
#st{{font-size:12px}} .ok{{color:#4caf50}} .no{{color:#ff7043}}
</style></head><body>
<header>
 <h1>{label} <span style="color:#777;font-size:12px">· {len(recs)}장 · 무작위 차례(눈가림)</span></h1>
 <div class="sub"><span id="st"></span> &nbsp;|&nbsp;
   <b>숫자</b>=BC · <b>QWERTYUIO</b>=SCCA · <b>ASDFZX</b>=태그 &nbsp;|&nbsp;
   <b>/</b> 메모 · <b>Enter</b> 다음 · <b>←→</b> 이동 · <b>Backspace</b> 지우기 · <b>+</b> 확대</div>
 <div id="bar"><div id="barf"></div></div>
</header>
<div id="wrap"><img id="im" src=""></div>
<footer>
  <div class="row"><span class="tit">BC</span>{bar(BC_AXIS, "b1", "pickBC")}</div>
  <div class="row"><span class="tit">SCCA</span>{bar(SCCA_AXIS, "b2", "pickWC")}</div>
  <div class="row"><span class="tit">태그</span>{tagbar}
      <button class="x" onclick="exportCSV()">CSV 내보내기</button>
      <button class="x" onclick="jumpTodo()">미판정으로</button></div>
  <div class="row"><span class="tit">메모</span>
      <input id="note" placeholder="특징을 자유롭게 (/ 로 들어오고 Enter 로 나갑니다)"></div>
</footer>
<script>
const R = {json.dumps(recs, ensure_ascii=False)};
const KEY = "{key}";
const MBC = {json.dumps({k: c for k, c, _, _ in BC_AXIS})};
const MWC = {json.dumps({k: c for k, c, _, _ in SCCA_AXIS})};
const MTG = {json.dumps({k: c for k, c, _ in TAGS})};
let S = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let p = 0;
function save(){{ localStorage.setItem(KEY, JSON.stringify(S)); }}
function cur(){{ const f=R[p].f; if(!S[f]) S[f]={{bc:"",wc:"",tags:[],note:""}}; return S[f]; }}
function ready(s){{ return s && s.bc && s.wc; }}
function draw(){{
  const r = R[p], s = cur();
  document.getElementById("im").src = r.img;
  document.querySelectorAll(".b1").forEach(b=>b.classList.toggle("on", b.dataset.c===s.bc));
  document.querySelectorAll(".b2").forEach(b=>b.classList.toggle("on", b.dataset.c===s.wc));
  Object.values(MTG).forEach(c=>{{
    const el=document.getElementById("tg_"+c); if(el) el.classList.toggle("on", s.tags.includes(c)); }});
  document.getElementById("note").value = s.note || "";
  const done = Object.values(S).filter(ready).length;
  document.getElementById("barf").style.width = (100*done/R.length)+"%";
  document.getElementById("st").innerHTML =
    `[${{p+1}}/${{R.length}}] ${{r.f}}`
    + ` &nbsp; BC <span class="${{s.bc?'ok':'no'}}">${{s.bc||'미판정'}}</span>`
    + ` &nbsp; SCCA <span class="${{s.wc?'ok':'no'}}">${{s.wc||'미판정'}}</span>`
    + ` &nbsp; 완료 ${{done}}/${{R.length}}`;
  document.title = `[${{done}}/${{R.length}}] {label}`;
}}
function nextIfReady(){{ if(ready(cur())&&p<R.length-1){{p++;}} draw(); }}
function pickBC(c){{ cur().bc=c; save(); nextIfReady(); }}
function pickWC(c){{ cur().wc=c; save(); nextIfReady(); }}
function tog(c){{ const s=cur(); const i=s.tags.indexOf(c);
  if(i<0) s.tags.push(c); else s.tags.splice(i,1); save(); draw(); }}
function jumpTodo(){{
  for(let k=0;k<R.length;k++){{ const q=(p+1+k)%R.length; const s=S[R[q].f];
    if(!ready(s)){{ p=q; draw(); return; }} }}
  alert("전부 판정했습니다. CSV 를 내보내십시오.");
}}
function exportCSV(){{
  let rows=[["stem","bc_code","scca_code","tags","note","무리","쓴단계"]];
  R.forEach(r=>{{ const s=S[r.f]; if(s&&(s.bc||s.wc)) rows.push(
    [r.f,s.bc,s.wc,(s.tags||[]).join("|"),
     (s.note||"").replace(/[",\\n]/g," "),r.co,r.stg]); }});
  const csv="\\uFEFF"+rows.map(a=>a.join(",")).join("\\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));
  a.download=KEY+"_판정.csv"; a.click();
}}
const note=document.getElementById("note");
note.addEventListener("input",()=>{{ cur().note=note.value; save(); }});
document.addEventListener("keydown", e=>{{
  if(e.target.id==="note"){{ if(e.key==="Enter"||e.key==="Escape"){{ note.blur(); e.preventDefault(); }} return; }}
  if(e.key==="/"){{ note.focus(); e.preventDefault(); return; }}
  const k=e.key.toLowerCase();
  if(MBC[e.key]){{ pickBC(MBC[e.key]); e.preventDefault(); return; }}
  if(MWC[k]){{ pickWC(MWC[k]); e.preventDefault(); return; }}
  if(MTG[k]){{ tog(MTG[k]); e.preventDefault(); return; }}
  if(e.key==="Enter"&&p<R.length-1){{ p++; draw(); e.preventDefault(); return; }}
  if(e.key==="ArrowRight"&&p<R.length-1){{ p++; draw(); }}
  if(e.key==="ArrowLeft" &&p>0)         {{ p--; draw(); }}
  if(e.key==="Backspace"){{ delete S[R[p].f]; save(); draw(); e.preventDefault(); }}
  if(e.key==="+"||e.key==="="){{ document.body.classList.toggle("zoom"); e.preventDefault(); }}
}});
draw();
</script></body></html>"""
    dst.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["internal", "external"], default="internal")
    ap.add_argument("--reader", choices=list(READERS), default="primary")
    ap.add_argument("--device", default="0"); ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pageonly", action="store_true", help="SAM 안 돌리고 판정 화면만 다시 만든다")
    a = ap.parse_args()
    EXT = a.set == "external"; NM = "바깥쪽" if EXT else "안쪽"
    bc4 = ROOT / ("results/260904_bc4_ext" if EXT else "results/260904_bc4")
    v3  = ROOT / ("results/260903_final_v3_ext" if EXT else "results/260903_final_v3")
    pj = json.load(open(OLD / "autosegmentation/external/results/external_index.json" if EXT
                        else "PATH/TO/PROJECT/260910_eClinicalMedicine_filtered_2_autosegmentation/results/extra42_index.json"))
    exif = pj["apply_exif"]
    if EXT:
        it = pj["items"]
        poly = {Path(k).stem: v["polygons"] for k, v in it.items()}
        raws = {Path(k).stem: v["raw"] for k, v in it.items()}
    else:
        full = json.load(open(ROOT / "results/polygon_index_full.json"))
        poly = {s: v["polygons"] for s, v in full.items()}
        raws = {s: pj["items"][s+".jpg"]["raw"] for s in full if (s+".jpg") in pj["items"]}
    keys = sorted(p.stem for p in (bc4/"mask").glob("*.npz")
                  if (v3/"mask"/f"{p.stem}.npz").exists() and p.stem in raws)
    if a.limit: keys = keys[:a.limit]
    RD = READERS[a.reader]
    (OUT/"panel").mkdir(parents=True, exist_ok=True); (OUT/"mask").mkdir(exist_ok=True)
    (OUT/RD["dir"]).mkdir(exist_ok=True)

    if not a.pageonly:
        print(f"[{NM}] {len(keys)}장 × SAM 2회 = {len(keys)*2}회", flush=True)
        dev = f"cuda:{a.device}"
        model = Sam3Model.from_pretrained(MODEL_ID).to(dev).eval()
        proc = Sam3Processor.from_pretrained(MODEL_ID)

        @torch.no_grad()
        def seg(pil, text, shape):
            z = np.zeros(shape, bool)
            inp = proc(images=pil, text=text, return_tensors="pt").to(dev)
            R = proc.post_process_instance_segmentation(
                model(**inp), threshold=a.threshold, mask_threshold=0.5,
                target_sizes=inp.get("original_sizes").tolist())[0]
            ms = R["masks"].cpu().numpy() > 0
            if not len(ms): return z, z
            o = np.argsort(-R["scores"].cpu().numpy()); ms = ms[o]
            return ms[:2].any(axis=0), ms.any(axis=0)

        rows = []
        for i, s in enumerate(keys):
            z, z3 = np.load(bc4/"mask"/f"{s}.npz"), np.load(v3/"mask"/f"{s}.npz")
            H, W = z["shape"]
            img = read_raw(raws[s], exif); pil = Image.fromarray(img)
            m2r, _ = seg(pil, P_REG, (H, W))
            _, mas = seg(pil, P_SKIN, (H, W))
            M = build_one(z, z3, m2r, mas, H, W)

            np.savez_compressed(OUT/"mask"/f"{s}.npz", shape=np.array([H, W]),
                                bc=np.packbits(M["BC"]), sclera=np.packbits(M["SCLERA"]),
                                scca=np.packbits(M["SCCA"]), reg2=np.packbits(M["REG2"]),
                                reg1=np.packbits(M["REG1"]), iris=np.packbits(M["IRIS"]))
            r = dict(stem=s, 무리=NM, 쓴단계=M["쓴단계"], 공막몫=M["공막몫"],
                     살갗침범=M["살갗침범"], 살갗씀=M["살갗씀"],
                     안전망A=M["안전망A"], 안전망B=M["안전망B"], 타원씀=M["타원씀"], 띠몫=M["띠몫"],
                     SCCA화소=int(M["SCCA"].sum()), BC화소=int(M["BC"].sum()),
                     공막화소=int(M["SCLERA"].sum()))
            P = poly.get(s) or {}
            if P.get("WCBC") and P.get("BC"):
                gw = polygon_mask(P["WCBC"], (W, H)); gb = polygon_mask(P["BC"], (W, H))
                gs = gw & ~gb
                if gb.sum() > 100 and gs.sum() > 100:
                    lw = bottom_line(gw, W)
                    r.update(SCCA덮음=float((M["SCCA"] & gw).sum()/gw.sum()),
                             SCCA크기비=float(M["SCCA"].sum()/gw.sum()),
                             SCCA아래삐짐=below_frac(M["SCCA"], lw, H),
                             BC덮음=float((M["BC"] & gb).sum()/gb.sum()),
                             공막덮음=float((M["SCLERA"] & gs).sum()/gs.sum()),
                             띠가눈이었나=float((M["띠"] & gw).sum()/max(M["띠"].sum(), 1)) if M["띠"].any() else np.nan)
            rows.append(r)

            # ── 판정 그림 넉 칸 (사람 주석을 안 그린다 — 판독자가 편향되지 않게)
            base_m = M["SCCA"] | M["IRIS"]
            if base_m.sum() < 100:
                base_m = np.ones((H, W), bool)
            ys, xs = np.nonzero(base_m)
            ph, pw = int((ys.max()-ys.min())*0.9)+20, int((xs.max()-xs.min())*0.35)+20
            cy0, cy1 = max(0, ys.min()-ph), min(H, ys.max()+ph)
            cx0, cx1 = max(0, xs.min()-pw), min(W, xs.max()+pw)
            bs = img[cy0:cy1, cx0:cx1]; cm = lambda m: m[cy0:cy1, cx0:cx1]
            # 🔴 outline() 은 준 배열에 **그 자리에서** 그린다. np.ascontiguousarray 는
            #    이미 연속이면 **같은 배열을 돌려주므로** ①과 ④가 한 배열이 되어 버린다. 반드시 copy()
            T = [("① 원본", bs.copy()),
                 ("② 결막 BC", np.ascontiguousarray(tint(bs, cm(M["BC"]), C_BC, 0.45))),
                 ("③ SCCA — 결막 민트 · 공막 하늘",
                  np.ascontiguousarray(tint(tint(bs, cm(M["BC"]), C_BC, 0.45),
                                            cm(M["SCLERA"]), C_SC, 0.45))),
                 ("④ 테두리만",
                  outline(outline(bs.copy(), cm(M["BC"]), C_BC), cm(M["SCLERA"]), C_SC)),
                 (f"⑤ ★이어 메운 띠(노랑) · SCCA 의 {M['띠몫']:.1%}",
                  np.ascontiguousarray(
                      tint(tint(tint(bs, cm(M["BC"]), C_BC, 0.35), cm(M["SCLERA"]), C_SC, 0.35),
                           cm(M["띠"]), (255, 210, 0), 0.75)))]
            th = 34; hh, ww = T[0][1].shape[:2]
            cv_ = Image.new("RGB", (ww*5, hh+th), (18, 18, 18)); dr = ImageDraw.Draw(cv_)
            f2 = _font(max(14, ww//28))
            for k, (cap, im) in enumerate(T):
                cv_.paste(Image.fromarray(im), (k*ww, th)); dr.text((k*ww+8, 6), cap, font=f2, fill=(255, 230, 140))
            cv_.save(OUT/"panel"/f"{s}.jpg", quality=84)
            if (i+1) % 50 == 0: print(f"  [{i+1}/{len(keys)}]", flush=True)

        D = pd.DataFrame(rows); D.to_csv(OUT/f"장별_{NM}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===== [{NM}] {len(D)}장 =====")
        print(f"  2단계를 쓴 장 {int((D.쓴단계==2).sum())} ({100*(D.쓴단계==2).mean():.1f} %) · "
              f"살갗 쓴 장 {int(D.살갗씀.sum())} · 타원 쓴 장 {int(D.타원씀.sum())} · "
              f"안전망 A {int(D.안전망A.sum())} · B {int(D.안전망B.sum())}")
        print(f"  빈 띠를 메운 장 {int((D.띠몫>0.001).sum())} · 메운 몫 평균 {D.띠몫.mean():.4f}"
              + (f" · 메운 것이 눈이었던 몫 {D.띠가눈이었나.mean():.3f}" if "띠가눈이었나" in D else ""))
        for c in ("SCCA덮음", "SCCA크기비", "SCCA아래삐짐", "BC덮음", "공막덮음"):
            if c in D: print(f"  {c:<12}{D[c].mean():.3f}")
    else:
        D = pd.read_csv(OUT/f"장별_{NM}.csv"); D.stem = D.stem.astype(str)

    recs = [dict(f=str(r.stem), img=f"../panel/{r.stem}.jpg", co=NM, stg=int(r.쓴단계))
            for r in D.itertuples() if (OUT/"panel"/f"{r.stem}.jpg").exists()]

    # ★ 바깥쪽은 **인도와 이탈리아를 나눈다** (stem 모양으로 가른다: i_* = 인도 · 숫자 = 이탈리아)
    #   🔴 화면마다 저장 열쇠가 달라야 한다. 예전에 두 곳이 한 열쇠를 써서 판정이 섞인 적이 있다.
    if EXT:
        그룹 = [("인도", [r for r in recs if r["f"].startswith("i_")]),
              ("이탈리아", [r for r in recs if not r["f"].startswith("i_")])]
    else:
        그룹 = [(NM, recs)]
    for 이름, rs in 그룹:
        if not rs:
            continue
        for r in rs:
            r["co"] = 이름
        random.Random(SEED).shuffle(rs)                            # ★ 눈가림 — 차례를 섞는다
        page(rs, f"자동분할 최종 판정 · {이름}", f"{RD['key']}_{이름}",
             OUT/RD["dir"]/f"review_{이름}.html")
        print(f"  {OUT/RD['dir']}/review_{이름}.html  ({len(rs)}장 · 저장키 {RD['key']}_{이름})")


if __name__ == "__main__":
    main()
