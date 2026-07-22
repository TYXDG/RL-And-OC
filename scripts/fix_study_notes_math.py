#!/usr/bin/env python3
"""Convert \\(...\\) and \\[...\\] to $ / $$ in study-notes (Cursor markdown preview)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "study-notes"


def convert_latex_delimiters(text: str) -> str:
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\n\1\n$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def main() -> None:
    for md in sorted(NOTES.glob("ch*.md")):
        original = md.read_text(encoding="utf-8")
        updated = convert_latex_delimiters(original)
        if updated != original:
            md.write_text(updated, encoding="utf-8")
            print(f"updated {md.name}")
        else:
            print(f"ok {md.name}")


if __name__ == "__main__":
    main()
