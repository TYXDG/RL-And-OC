#!/usr/bin/env python3
"""Extract text from the Zero-Order Robotics tutorial PDF."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "2025-An Introduction to Zero-Order Optimization Techniques for Robotics.pdf"
OUT = ROOT / "source" / "full.txt"


def main() -> None:
    if not PDF.is_file():
        raise SystemExit(f"PDF not found: {PDF}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["pdftotext", str(PDF), str(OUT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or "pdftotext failed")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
