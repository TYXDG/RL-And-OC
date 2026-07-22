#!/usr/bin/env python3
"""Extract and clean chapter text from Bertsekas RL&OC PDF (via pdftotext output)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "Reinforcement learning and optimal control.pdf"
OUT_DIR = ROOT / "source"
RAW = OUT_DIR / "full.txt"

# Line ranges in pdftotext output (1-based, inclusive start, exclusive end).
CHAPTER_RANGES = {
    1: ("Exact Dynamic Programming", 754, 3935),
    2: ("Approximation in Value Space", 4095, 7984),
    3: ("Parametric Approximation", 8047, 10529),
    4: ("Infinite Horizon Reinforcement Learning", 10607, 17236),
    5: (
        "Infinite Horizon Reinforcement Learning — Approximate Methods",
        17294,
        22246,
    ),
}


def pdftotext() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF.is_file():
        raise SystemExit(f"PDF not found: {PDF}")
    proc = subprocess.run(
        ["pdftotext", str(PDF), str(RAW)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or "pdftotext failed")
    return RAW.read_text(encoding="utf-8", errors="replace")


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("\f"):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if re.fullmatch(r"Chap\. \d+", s):
        return True
    if s in {"DRAFT", "Exact Dynamic Programming", "Approximation in Value Space",
             "Parametric Approximation", "Infinite Horizon Reinforcement Learning"}:
        return True
    if re.fullmatch(r"Sec\. \d+\.\d+ .*", s) and len(s) < 80:
        return True
    return False


def clean_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf = ""
    for raw in lines:
        line = raw.replace("\f", "").strip()
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if is_noise_line(line):
            continue
        # Skip isolated page numbers / running headers
        if re.fullmatch(r"Exact Dynamic Programming|Approximation in Value Space|"
                        r"Parametric Approximation|Infinite Horizon Reinforcement Learning", line):
            continue
        if buf:
            # Join hyphenated line breaks: "program-\nming" already merged by pdftotext often as "program ming"
            if buf.endswith("-") and line[0].islower():
                buf = buf[:-1] + line
                continue
            if not re.search(r"[.!?;:)\]\"']$", buf) and line and line[0].islower():
                buf = buf + " " + line
                continue
            out.append(buf)
            buf = line
        else:
            buf = line
    if buf:
        out.append(buf)
    return out


def split_sentences(paragraph: str) -> list[str]:
    """Split English prose; keep equations/display lines intact."""
    p = paragraph.strip()
    if not p:
        return []
    if re.search(r"^\(.*\)$|^x_\{|^J\s*\*|^Q\s*\*|^\\|^=\s*min|^=\s*\\sum", p):
        return [p]
    if len(p) < 120 and ("=" in p or "min" in p) and any(c in p for c in "()"):
        return [p]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", p)
    return [s.strip() for s in parts if s.strip()]


def paragraphs_from_lines(lines: list[str]) -> list[str]:
    return clean_lines(lines)


def extract_chapter(full_lines: list[str], start: int, end: int | None) -> list[str]:
    # 1-based line numbers in CHAPTER_RANGES
    sl = full_lines[start - 1 : end]
    return paragraphs_from_lines(sl)


def main() -> None:
    if not RAW.is_file():
        pdftotext()
    full_lines = RAW.read_text(encoding="utf-8", errors="replace").splitlines()
    for num, (title, start, end) in CHAPTER_RANGES.items():
        paras = extract_chapter(full_lines, start, end)
        path = OUT_DIR / f"ch{num:02d}_clean.txt"
        path.write_text("\n\n".join(paras) + "\n", encoding="utf-8")
        print(f"Wrote {path} ({len(paras)} paragraphs, {path.stat().st_size} bytes) — {title}")


if __name__ == "__main__":
    main()
