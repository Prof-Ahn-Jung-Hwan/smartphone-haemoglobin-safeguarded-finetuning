"""Pre-push privacy check for this repository. Run from the repository root:  python tools/check_no_private.py
Fails (exit 1) if any file contains machine paths, internal file names, chart-number-like identifiers,
or if an image still carries EXIF metadata. Development-cohort data must never be in this repository."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".sh", ".md", ".yaml", ".yml", ".txt", ".csv", ".json", ".cff", ".toml", ".cfg"}
RULES = [
    ("machine path", re.compile(r"/home/|/mnt/|erdrajh(?!@ajou\.ac\.kr|@naver\.com)")),   # 공개 연락처 이메일만 허용
    ("internal file name", re.compile(r"_내부용_[^\s\"'`]*\.(csv|xlsx)\b(?!.*(not included|포함하지))")),
    ("development-cohort image stem (date_number_eye_ID)", re.compile(r"20\d{6}_\d+_\w*?_+\d{5,}")),
    ("7-digit number (possible chart number)", re.compile(r"(?<![\d.\-·])\d{7}(?![\d.])")),
]
ALLOW_7 = {"2974833"}            # deployed parameter count
bad = 0
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or ".git" in p.parts or "work" in p.parts or p.name == Path(__file__).name: continue
    rel = p.relative_to(ROOT)
    if p.stat().st_size > 50 * 2**20: print(f"🔴 {rel}: larger than 50 MB"); bad += 1
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        from PIL import Image
        if Image.open(p).getexif(): print(f"🔴 {rel}: image has EXIF metadata"); bad += 1
        continue
    if p.suffix.lower() in {".xlsx", ".npz", ".ckpt", ".pt", ".pth"}:
        print(f"🔴 {rel}: binary data/weights file — confirm it may be published"); bad += 1; continue
    if p.suffix.lower() not in TEXT: continue
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        for name, rx in RULES:
            for m in rx.finditer(line):
                if name.startswith("7-digit") and m.group(0) in ALLOW_7: continue
                print(f"🔴 {rel}:{i}: {name}: {line.strip()[:100]}"); bad += 1
print("✅ nothing found" if not bad else f"🔴 {bad} finding(s) — fix before pushing"); sys.exit(1 if bad else 0)
